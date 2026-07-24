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
from bisect import bisect_left
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from common.preprocess import IMAGE_HEIGHT, IMAGE_WIDTH, decode_frame
from common.schema import (
    ACTION,
    JOINTS,
    STATE,
    image_key,
    iter_kept_episodes,
    read_joints_jsonl,
    read_meta,
)


def load_packets(mkv_path: Path) -> tuple[list[float], list[bytes]]:
    """Demux JPEG packets (no decode). Returns (relative_times, jpeg_bytes) sorted."""
    times: list[float] = []
    data: list[bytes] = []
    with av.open(str(mkv_path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            times.append(float(packet.pts * stream.time_base))
            data.append(bytes(packet))
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]  # relative to first frame; align on relative timelines (§3.5)
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
                    task: str, cam_offset: float) -> None:
    ticks = read_joints_jsonl(ep_dir)
    tick_times = [row["t"] for row in ticks]

    # Load each camera's JPEG packet timeline once.
    cam_packets = {c: load_packets(ep_dir / f"cam_{c}.mkv") for c in cams}
    for c, (times, data) in cam_packets.items():
        if not data:
            raise RuntimeError(f"{ep_dir.name}: camera '{c}' produced no frames")

    # Resample: nearest camera frame per control tick; track alignment error (§4).
    errors: list[float] = []
    decode_cache: dict[tuple[str, int], np.ndarray] = {}
    states, actions = [], []

    for row, t_tick in zip(ticks, tick_times):
        state = np.asarray(row["state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)
        assert state.shape == (len(JOINTS),), f"state has {state.shape[0]} joints, expected {len(JOINTS)}"
        assert not np.isnan(state).any() and not np.isnan(action).any(), "NaN in joint log"
        states.append(state)
        actions.append(action)

        frame = {ACTION: action, STATE: state, "task": task}
        for c in cams:
            times, data = cam_packets[c]
            idx = nearest_index(times, t_tick + cam_offset)
            errors.append(abs(times[idx] - (t_tick + cam_offset)))
            key = (c, idx)
            if key not in decode_cache:
                decode_cache[key] = decode_frame(data[idx])
            frame[image_key(c)] = decode_cache[key]
        dataset.add_frame(frame)

    _assert_episode(np.stack(states), np.stack(actions), len(ticks), errors, ep_dir)
    dataset.save_episode()

    max_err, med_err = max(errors) * 1000, float(np.median(errors)) * 1000
    flag = "  <-- EXCEEDS half-frame (16ms), investigate before recording more" if max_err > 16 else ""
    print(f"[{ep_dir.name}] {len(ticks)} frames  align max={max_err:.1f}ms median={med_err:.1f}ms{flag}")


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
    ap.add_argument("--session", required=True, help="Session dir, e.g. sessions/2026-07-23_pick_cube")
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/so101_pick_cube")
    ap.add_argument("--root", default=None, help="Local dataset dir (default: HF cache)")
    ap.add_argument("--cam-offset", type=float, default=0.0,
                    help="Seconds to shift camera timeline vs joints (verify empirically once, §3.5)")
    args = ap.parse_args()

    session_dir = Path(args.session)
    session = json.loads((session_dir / "session.json").read_text())
    cams = list(session["cameras"].keys())
    fps = session.get("fps", 30)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=build_features(cams),
        root=args.root, robot_type=session.get("robot_id", "so101"), use_videos=True,
    )

    for _, ep_dir in iter_kept_episodes(session_dir):
        convert_episode(dataset, ep_dir, cams, read_meta(ep_dir).task, args.cam_offset)

    dataset.finalize()
    print(f"Done -> {dataset.root}")


if __name__ == "__main__":
    main()
