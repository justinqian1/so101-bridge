"""Session dir -> LeRobotDataset (§4, §9 step 4).

Goes through the LeRobotDataset API (create/add_frame/save_episode) so upstream format
churn is a version bump, not a rewrite (§4) — this API boundary is the single
highest-leverage decision in the design.

The one piece of real logic is resampling: capture is variable-rate on both the joint
and camera paths, LeRobot wants fixed-fps aligned frames. Per control tick we pick the
nearest camera frame by timestamp and drop the rest, then log the alignment error.

Camera frames are pulled straight from the Matroska as JPEG *packets* (demux, no PyAV
decode) and run through common/preprocess.decode_frame — the exact PIL path serve.py
uses — so stored frames and live frames share one spatial transform (§1).

Desktop-only. Imports av, torch (via lerobot), PIL.
"""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from common.preprocess import IMAGE_HEIGHT, IMAGE_WIDTH, decode_frame
from common.schema import (
    ACTION,
    JOINTS,
    STATE,
    Meta,
    image_key,
    iter_kept_episodes,
    read_joints_jsonl,
    read_meta,
)


# PIL releases the GIL through JPEG decode and resize, so decoding scales on threads.
DECODE_THREADS = min(8, os.cpu_count() or 1)


def load_packets(mkv_path: Path) -> tuple[list[float], list[bytes]]:
    """Demux JPEG packets (no decode). Returns (unix_times, jpeg_bytes) sorted.

    PTS are unix-epoch seconds (record.py writes them with -use_wallclock_as_timestamps
    -copyts). Never rebase to the first frame: ffmpeg's first frame lands a few hundred
    ms after spawn, by a margin that differs per camera and per episode, so a
    first-frame-relative timeline has no fixed relationship to the joint log (§3.5).
    """
    times: list[float] = []
    data: list[bytes] = []
    with av.open(str(mkv_path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            times.append(float(packet.pts * stream.time_base))
            data.append(bytes(packet))
    return times, data


def nearest_index(times: list[float], target: float) -> int:
    i = bisect_left(times, target)
    if i == 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - target) < abs(times[i - 1] - target) else i - 1


def build_features(cams: list[str]) -> dict:
    vec = {"dtype": "float32", "shape": (len(JOINTS),), "names": JOINTS}
    features = {ACTION: dict(vec), STATE: dict(vec)}
    for cam in cams:
        features[image_key(cam)] = {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def convert_episode(dataset: LeRobotDataset, ep_dir: Path, cams: list[str],
                    meta: Meta, cam_offset: float, task: str) -> None:
    ticks = read_joints_jsonl(ep_dir)

    # Load each camera's JPEG packet timeline once.
    cam_packets = {c: load_packets(ep_dir / f"cam_{c}.mkv") for c in cams}
    for c, (times, data) in cam_packets.items():
        if not data:
            raise RuntimeError(f"{ep_dir.name}: camera '{c}' produced no frames")

    # Which clock -use_wallclock_as_timestamps lands on is an ffmpeg build detail: some
    # emit unix epoch, others CLOCK_MONOTONIC (av_gettime vs av_gettime_relative). Both
    # origins are in meta.json, so pick whichever actually sits next to the packets
    # rather than trusting either. Joint t is relative to that same origin.
    first_frame = min(times[0] for times, _ in cam_packets.values())
    t0 = min((meta.t0_unix, meta.t0_monotonic), key=lambda origin: abs(origin - first_frame))
    if abs(t0 - first_frame) > 60:
        raise RuntimeError(
            f"{ep_dir.name}: camera PTS start at {first_frame:.1f}s, which matches neither "
            f"t0_unix ({meta.t0_unix:.1f}) nor t0_monotonic ({meta.t0_monotonic:.1f}) — "
            f"the recording clock is not one this converter understands"
        )
    tick_times = [t0 + row["t"] for row in ticks]

    # Trim ticks to the span every camera actually covers. The cameras take a few
    # hundred ms to deliver their first frame, so the head of an episode has joint
    # data and no video; clamping those onto the first frame invents observations
    # that never happened. Drop them instead, and say how many (§4).
    lo = max(times[0] for times, _ in cam_packets.values()) - cam_offset
    hi = min(times[-1] for times, _ in cam_packets.values()) - cam_offset
    covered = [i for i, t in enumerate(tick_times) if lo <= t <= hi]
    if not covered:
        raise RuntimeError(f"{ep_dir.name}: no control tick falls inside camera coverage")
    ticks = ticks[covered[0]:covered[-1] + 1]
    tick_times = tick_times[covered[0]:covered[-1] + 1]

    # Resample: nearest camera frame per control tick; track alignment error (§4).
    errors: list[float] = []
    picks: list[dict[str, int]] = []
    for t_tick in tick_times:
        pick = {}
        for c in cams:
            times, _ = cam_packets[c]
            idx = nearest_index(times, t_tick + cam_offset)
            errors.append(abs(times[idx] - (t_tick + cam_offset)))
            pick[c] = idx
        picks.append(pick)

    # Decode every packet the resampling actually kept, once each, in parallel. Ticks
    # outnumber packets whenever a camera drops frames, so dedupe before decoding.
    keys = sorted({(c, pick[c]) for pick in picks for c in cams})
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        decoded = dict(zip(keys, pool.map(lambda k: decode_frame(cam_packets[k[0]][1][k[1]]), keys)))

    states, actions = [], []
    for row, pick in zip(ticks, picks):
        state = np.asarray(row["state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)
        assert state.shape == (len(JOINTS),), f"state has {state.shape[0]} joints, expected {len(JOINTS)}"
        assert not np.isnan(state).any() and not np.isnan(action).any(), "NaN in joint log"
        states.append(state)
        actions.append(action)

        frame = {ACTION: action, STATE: state, "task": task}
        for c in cams:
            frame[image_key(c)] = decoded[(c, pick[c])]
        dataset.add_frame(frame)

    _assert_episode(np.stack(states), np.stack(actions), len(ticks), errors, ep_dir)
    dataset.save_episode()

    max_err, med_err = max(errors) * 1000, float(np.median(errors)) * 1000
    print(f"[{ep_dir.name}] {len(ticks)} frames  align max={max_err:.1f}ms median={med_err:.1f}ms")


def _assert_episode(states: np.ndarray, actions: np.ndarray, n_ticks: int,
                    errors: list[float], ep_dir: Path) -> None:
    # action/observation.state must not be identical (catches a wiring/naming swap, §4).
    assert not np.array_equal(states, actions), (
        f"{ep_dir.name}: action == observation.state everywhere — leader/follower likely swapped"
    )
    # Normalised range: body ±100, gripper 0–100 (small tolerance for int rounding).
    body = np.concatenate([states[:, :-1], actions[:, :-1]])
    grip = np.concatenate([states[:, -1], actions[:, -1]])
    assert np.all(np.abs(body) <= 101), f"{ep_dir.name}: body joint outside ±100"
    assert np.all((grip >= -1) & (grip <= 101)), f"{ep_dir.name}: gripper outside 0–100"
    assert len(errors) % n_ticks == 0, "frame/tick count mismatch"


def main():
    ap = argparse.ArgumentParser(description="Convert a capture session to a LeRobotDataset.")
    ap.add_argument("--session", required=True, nargs="+",
                    help="One or more session dirs, sharing cams, fps, and robot_id.")
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/so101_pick_cube")
    ap.add_argument("--root", default=None, help="Local dataset dir (default: HF cache)")
    ap.add_argument("--task", default=None,
                    help="Override every episode's recorded task with this one string.")
    ap.add_argument("--cam-offset", type=float, default=0.0,
                    help="Seconds to shift camera timeline vs joints (verify empirically once, §3.5)")
    args = ap.parse_args()

    session_dirs = [Path(s) for s in args.session]
    sessions = [json.loads((d / "session.json").read_text()) for d in session_dirs]

    cams = list(sessions[0]["cameras"].keys())
    fps = sessions[0].get("fps", 30)
    robot_id = sessions[0].get("robot_id", "so101")
    for d, s in zip(session_dirs[1:], sessions[1:]):
        if list(s["cameras"].keys()) != cams:
            raise SystemExit(f"{d}: cameras {list(s['cameras'].keys())} != {cams} (from {session_dirs[0]})")
        if s.get("fps", 30) != fps:
            raise SystemExit(f"{d}: fps {s.get('fps', 30)} != {fps} (from {session_dirs[0]})")
        if s.get("robot_id", "so101") != robot_id:
            raise SystemExit(f"{d}: robot_id {s.get('robot_id', 'so101')} != {robot_id} (from {session_dirs[0]})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=build_features(cams),
        root=args.root, robot_type=robot_id, use_videos=True,
        # Feed frames straight to the encoder threads; the default path round-trips
        # every frame through a temporary PNG, which costs more than the encode itself.
        streaming_encoding=True,
    )

    for session_dir in session_dirs:
        for _, ep_dir in iter_kept_episodes(session_dir):
            meta = read_meta(ep_dir)
            # LeRobot indexes tasks per frame, so episodes in one dataset may differ (§4).
            task = args.task or meta.task
            if not task:
                raise SystemExit(f"{ep_dir}: no task in meta.json — pass --task")
            convert_episode(dataset, ep_dir, cams, meta, args.cam_offset, task)

    dataset.finalize()
    print(f"Done -> {dataset.root}")


if __name__ == "__main__":
    main()
