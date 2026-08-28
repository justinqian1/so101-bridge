"""Teleop + camera capture to disk (§3, §9 step 3).

Per tick (target 30 Hz, jitter-tolerant): sync-read follower (observation.state),
sync-read leader (action), write leader position to follower, hand the sample to a
writer thread. The control loop never touches disk — a separate thread drains a queue
to joints.jsonl so SD/SSD write latency can't stall the servo loop. Cameras are ffmpeg
passthrough (§3.2); JPEGs never get decoded here (§0).

Pi-side. numpy + feetech-servo-sdk only.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

from common.calibration import calibration_hash, load_calibration
from common.cameras import FileRecorder
from common.schema import (
    CONTROL_HZ,
    Meta,
    episode_dir,
    next_episode_index,
    write_meta,
)
from common.servo import ServoBus


# ── writer thread: queue -> joints.jsonl (§3, keep disk off the control loop) ─

def _writer_thread(path: Path, q: "queue.Queue") -> None:
    with open(path, "w") as f:
        while True:
            item = q.get()
            if item is None:
                f.flush()
                return
            f.write(json.dumps(item) + "\n")
            if q.empty():
                f.flush()


# ── non-blocking single-key listener (operator control) ──────────────────────

class KeyListener:
    """Reads the tty in cbreak mode from whichever thread calls it."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def drain(self) -> str:
        """Consume every pending keystroke, in arrival order. Non-blocking."""
        out = ""
        while select.select([self._fd], [], [], 0)[0]:
            b = os.read(self._fd, 64)
            if not b:
                break
            out += b.decode(errors="ignore")
        return out

    def wait(self, keys: str) -> str:
        while True:
            for k in self.drain():
                if k in keys:
                    return k
            time.sleep(0.02)

    def prompt(self, msg: str) -> str:
        """Read a full line: cooked mode for echo and line editing, then back to cbreak."""
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        try:
            return input(msg).strip()
        finally:
            tty.setcbreak(self._fd)

    def restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# ── session provenance (§3.3) ────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _warn_if_throttled() -> None:
    # Undervoltage shows up as random USB dropouts that look like software bugs (§7).
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return
    if out.partition("=")[2] not in ("0x0", ""):
        print(f"[session] WARNING: {out} — undervoltage/throttling drops servo bus packets")


# ── one episode ──────────────────────────────────────────────────────────────

def record_episode(ep_dir: Path, task: str, follower: ServoBus, leader: ServoBus,
                   cameras: FileRecorder, keys: KeyListener) -> None:
    """Record one episode until [s]top."""
    ep_dir.mkdir(parents=True)
    period = 1.0 / CONTROL_HZ

    q: queue.Queue = queue.Queue()
    writer = threading.Thread(target=_writer_thread, args=(ep_dir / "joints.jsonl", q), daemon=True)
    writer.start()

    i = 0
    # ONE monotonic clock, in ONE process; everything else is an offset (§3.5).
    # Captured before cameras.start() so a crash mid-setup still has a t0 to write.
    t0_mono = time.monotonic()
    t0_unix = time.time()
    try:
        cameras.start(ep_dir)
        print(f"[{ep_dir.name}] recording — [s]top")

        stopped = False
        while not stopped:
            cameras.check_alive()
            state = follower.read_positions()
            action = leader.read_positions()
            follower.write_positions(action)
            q.put({"t": round(time.monotonic() - t0_mono, 4),
                   "state": state.tolist(), "action": action.tolist()})

            if "s" in keys.drain():
                stopped = True
                break

            i += 1
            target = t0_mono + i * period
            sleep = target - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except Exception as e:
        print(f"[{ep_dir.name}] crashed: {e!r} — discarding")
        write_meta(ep_dir, Meta(t0_monotonic=t0_mono, t0_unix=t0_unix,
                                fps=CONTROL_HZ, task=task, status="discard"))
        raise
    finally:
        # A servo dropout must not leave ffmpeg orphaned: it keeps appending to this
        # episode's .mkv forever and holds the device against every later episode.
        cameras.stop()
        q.put(None)
        writer.join()

    print(f"[{ep_dir.name}] stopped — [k]eep  [d]iscard")
    status = "keep" if keys.wait("kd") == "k" else "discard"

    write_meta(ep_dir, Meta(t0_monotonic=t0_mono, t0_unix=t0_unix, fps=CONTROL_HZ, task=task, status=status))
    print(f"[{ep_dir.name}] {i} ticks, status={status}")


RECONNECT_ATTEMPTS = 8
RECONNECT_DELAY_S = 0.5


def _reconnect(follower: ServoBus, leader: ServoBus) -> bool:
    """Close + reopen both buses after a crash. Most dropouts are a transient USB hiccup
    that clears within a second or two, so retry a handful of times before giving up."""
    for attempt in range(1, RECONNECT_ATTEMPTS + 1):
        try:
            follower.close()
            leader.close()
            follower.connect()
            follower.configure()
            follower.enable_torque()
            leader.connect()
            leader.disable_torque()
            return True
        except Exception as e:
            print(f"[session] reconnect attempt {attempt}/{RECONNECT_ATTEMPTS} failed: {e!r}")
            time.sleep(RECONNECT_DELAY_S)
    return False


def _ask_task(keys: KeyListener, msg: str) -> str:
    """Ask until a non-empty task: it is what convert.py writes into the dataset."""
    while True:
        task = keys.prompt(msg)
        if task:
            return task


def main():
    ap = argparse.ArgumentParser(description="SO-101 teleop capture.")
    ap.add_argument("--name", required=True, help="Session name, e.g. 'pickup_cube'")
    ap.add_argument("--out", default="sessions", help="Sessions root directory")
    ap.add_argument("--robot-id", default="so101")
    ap.add_argument("--follower-calib", default="calibration/so101_follower.json")
    ap.add_argument("--leader-calib", default="calibration/so101_leader.json")
    args = ap.parse_args()

    # Fixed device aliases from port-alias-setup.md — not configurable per-run.
    cameras_cfg = {
        "ext": "/dev/v4l/by-id/cam-ext",
        "wrist": "/dev/v4l/by-id/cam-wrist",
    }
    _warn_if_throttled()

    follower = ServoBus("/dev/so101-follower", load_calibration(args.follower_calib))
    leader = ServoBus("/dev/so101-leader", load_calibration(args.leader_calib))
    follower.connect()
    leader.connect()
    follower.configure()
    follower.enable_torque()
    leader.disable_torque()  # leader is moved by hand

    stamp = datetime.now().strftime("%Y-%m-%d")
    session_dir = Path(args.out) / f"{stamp}_{args.name}"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(json.dumps({
        "robot_id": args.robot_id,
        "git_sha": _git_sha(),
        "calibration_hash": {
            "follower": calibration_hash(args.follower_calib),
            "leader": calibration_hash(args.leader_calib),
        },
        "cameras": cameras_cfg,
        "fps": CONTROL_HZ,
    }, indent=2))
    print(f"[session] {session_dir}")

    cameras = FileRecorder(cameras_cfg)
    keys = KeyListener()
    try:
        task = _ask_task(keys, "[session] task description: ")
        while True:
            print(f"[session] task: {task} | [s]tart recording  [t]ask change  [q]uit")
            k = keys.wait("stq")
            if k == "q":
                break
            if k == "t":
                task = _ask_task(keys, "[session] new task description: ")
                continue
            idx = next_episode_index(session_dir)
            try:
                record_episode(episode_dir(session_dir, idx), task,
                               follower, leader, cameras, keys)
            except Exception:
                print("[session] attempting to reconnect...")
                if not _reconnect(follower, leader):
                    print("[session] reconnect failed — exiting")
                    raise
                print("[session] reconnected")
    finally:
        keys.restore()
        try:
            follower.disable_torque()
        except ConnectionError as e:
            # Bus is already dead; say so instead of masking why we got here.
            print(f"[session] follower torque NOT released: {e}")
        follower.close()
        leader.close()
    print("[session] done")


if __name__ == "__main__":
    main()
