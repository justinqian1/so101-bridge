"""Calibration arithmetic, vendored from LeRobot (§2, §8).

Copied rather than imported: it's small, self-contained, and a vendored copy can't
break under an upstream refactor. Matches lerobot.motors.motors_bus normalisation and
lerobot.motors.encoding_utils sign-magnitude coding byte-for-byte, so calibration.json
files are interchangeable with stock LeRobot SO-101 tooling.

Pure stdlib. Safe to import on the Pi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from common.schema import JOINTS, NORM_MODES

# STS3215 encoder resolution (counts per turn).
RESOLUTION = 4096

# Sign-magnitude sign-bit index per register (feetech tables.py).
HOMING_OFFSET_SIGN_BIT = 11
POSITION_SIGN_BIT = 15


@dataclass
class MotorCalibration:
    """One motor's calibration. JSON field order matches LeRobot's dataclass."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


# ── sign-magnitude coding (encoding_utils.py) ────────────────────────────────


def encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"Magnitude {magnitude} exceeds {max_magnitude} for sign bit {sign_bit_index}")
    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int) -> int:
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude = encoded_value & ((1 << sign_bit_index) - 1)
    return -magnitude if direction_bit else magnitude


# ── normalisation (motors_bus.py _normalize / _unnormalize) ──────────────────


def normalize(joint: str, raw: int, cal: MotorCalibration) -> float:
    """Raw (homed) encoder counts -> ±100 (body) or 0–100 (gripper)."""
    lo, hi = cal.range_min, cal.range_max
    if hi == lo:
        raise ValueError(f"Invalid calibration for '{joint}': min and max are equal.")
    bounded = min(hi, max(lo, raw))
    if NORM_MODES[joint] == "range_m100_100":
        return ((bounded - lo) / (hi - lo)) * 200 - 100
    return ((bounded - lo) / (hi - lo)) * 100  # range_0_100


def unnormalize(joint: str, value: float, cal: MotorCalibration) -> int:
    """±100 / 0–100 -> raw (homed) encoder counts."""
    lo, hi = cal.range_min, cal.range_max
    if hi == lo:
        raise ValueError(f"Invalid calibration for '{joint}': min and max are equal.")
    if NORM_MODES[joint] == "range_m100_100":
        bounded = min(100.0, max(-100.0, value))
        return int(((bounded + 100) / 200) * (hi - lo) + lo)
    bounded = min(100.0, max(0.0, value))  # range_0_100
    return int((bounded / 100) * (hi - lo) + lo)


def half_turn_homing_offset(raw_middle: int) -> int:
    """Homing offset that centres the range: Present = Actual - Homing_Offset (§ feetech)."""
    return raw_middle - (RESOLUTION - 1) // 2


# ── persistence (matches robot.py _save/_load_calibration JSON shape) ────────


def load_calibration(path: str | Path) -> dict[str, MotorCalibration]:
    data = json.loads(Path(path).read_text())
    return {name: MotorCalibration(**fields) for name, fields in data.items()}


def save_calibration(path: str | Path, cal: dict[str, MotorCalibration]) -> None:
    data = {name: c.__dict__ for name, c in cal.items()}
    Path(path).write_text(json.dumps(data, indent=4))


def calibration_hash(path: str | Path) -> str:
    """Short sha256 of a calibration file, for provenance in session.json."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def assert_joint_order(cal: dict[str, MotorCalibration]) -> None:
    if list(cal.keys()) != JOINTS:
        raise ValueError(f"Calibration joints {list(cal.keys())} != expected order {JOINTS}")
