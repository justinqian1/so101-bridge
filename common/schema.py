"""Frozen conventions: joint order, rates, feature keys, and on-disk layout.

These are borrowed from LeRobot deliberately (§2). Matching them lets us fine-tune
from community SO-101 checkpoints and hand data off without explanation. Freeze on
day one — they have been far more stable than LeRobot's APIs.

Pure stdlib. Safe to import on the Pi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Fixed joint order. Defined once, imported everywhere.
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
GRIPPER = "gripper"

# Normalisation mode per joint: body joints ±100, gripper 0–100 (§2).
NORM_MODES = {j: ("range_0_100" if j == GRIPPER else "range_m100_100") for j in JOINTS}

# Feetech STS3215 servo ids, in JOINTS order.
MOTOR_IDS = {j: i + 1 for i, j in enumerate(JOINTS)}
MOTOR_MODEL = "sts3215"

# Rates (§2). Control loop 30 Hz, policy query 10 Hz.
CONTROL_HZ = 30
POLICY_HZ = 10

# Camera capture format (§3.2 / §7 — verify native MJPEG with v4l2-ctl first).
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Feature keys (§2). Getting action/observation.state backwards trains without
# error and produces a policy that does nothing useful — assert on it in convert.py.
ACTION = "action"                       # leader arm position (what was commanded)
STATE = "observation.state"             # follower arm position (what happened)
IMAGE_PREFIX = "observation.images."    # + camera name, e.g. observation.images.top


def image_key(cam_name: str) -> str:
    return f"{IMAGE_PREFIX}{cam_name}"


# ── On-disk capture layout (§3.3) ────────────────────────────────────────────


@dataclass
class Meta:
    """episode_NNN/meta.json"""

    t0_monotonic: float
    t0_unix: float
    fps: int
    task: str
    status: str  # "keep" | "discard"


def episode_dir(session_dir: Path, index: int) -> Path:
    return Path(session_dir) / f"episode_{index:03d}"


def next_episode_index(session_dir: Path) -> int:
    existing = sorted(Path(session_dir).glob("episode_*"))
    if not existing:
        return 0
    return int(existing[-1].name.split("_")[1]) + 1


def write_meta(ep_dir: Path, meta: Meta) -> None:
    (Path(ep_dir) / "meta.json").write_text(json.dumps(meta.__dict__, indent=2))


def read_meta(ep_dir: Path) -> Meta:
    d = json.loads((Path(ep_dir) / "meta.json").read_text())
    return Meta(**d)


def iter_kept_episodes(session_dir: Path):
    """Yield (index, ep_dir) for episodes marked kept, in order."""
    for ep_dir in sorted(Path(session_dir).glob("episode_*")):
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        if json.loads(meta_path.read_text()).get("status") == "kept":
            index = int(ep_dir.name.split("_")[1])
            yield index, ep_dir


def read_joints_jsonl(ep_dir: Path) -> list[dict]:
    """Load joints.jsonl as a list of {t, state, action} dicts."""
    lines = (Path(ep_dir) / "joints.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]
