"""One-off calibration for one SO-101 arm -> calibration/so101_<arm>.json (§9 step 1).

Procedure vendored from LeRobot's SO-101 calibrate flow (§2): centre each joint (half-turn
homing offset), then sweep the ranges of motion. wrist_roll is a full-turn joint on the
leader, so its range is fixed to the full encoder span instead of swept. The written JSON
is interchangeable with stock LeRobot tooling.

Pi-side. Talks to scservo_sdk directly (one-off, sequential reads — speed irrelevant).
"""

from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

import scservo_sdk as scs

from common.calibration import (
    HOMING_OFFSET_SIGN_BIT,
    POSITION_SIGN_BIT,
    RESOLUTION,
    MotorCalibration,
    decode_sign_magnitude,
    encode_sign_magnitude,
    half_turn_homing_offset,
    save_calibration,
)
from common.schema import JOINTS, MOTOR_IDS
from common.servo import (
    DEFAULT_BAUDRATE,
    HOMING_OFFSET,
    LOCK,
    MAX_POSITION_LIMIT,
    MIN_POSITION_LIMIT,
    OPERATING_MODE,
    POSITION_MODE,
    PRESENT_POSITION,
    TORQUE_ENABLE,
)


def enter_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0.0)[0]) and sys.stdin.readline() is not None


class _RawBus:
    """Minimal direct servo access for the calibration procedure only."""

    def __init__(self, port: str):
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(0)
        if not self.port.openPort() or not self.port.setBaudRate(DEFAULT_BAUDRATE):
            raise ConnectionError(f"Could not open {port} at {DEFAULT_BAUDRATE} baud")

    def w2(self, addr, id_, value):
        self.packet.write2ByteTxRx(self.port, id_, addr, value)

    def w1(self, addr, id_, value):
        self.packet.write1ByteTxRx(self.port, id_, addr, value)

    def present(self, id_) -> int:
        raw, _, _ = self.packet.read2ByteTxRx(self.port, id_, PRESENT_POSITION[0])
        return decode_sign_magnitude(raw, POSITION_SIGN_BIT)


def calibrate(port: str, full_turn_motor: str | None) -> dict[str, MotorCalibration]:
    bus = _RawBus(port)
    ids = MOTOR_IDS

    # Disable torque, position mode, reset any prior calibration to raw span.
    for id_ in ids.values():
        bus.w1(TORQUE_ENABLE[0], id_, 0)
        bus.w1(LOCK[0], id_, 0)
        bus.w1(OPERATING_MODE[0], id_, POSITION_MODE)
        bus.w2(HOMING_OFFSET[0], id_, 0)
        bus.w2(MIN_POSITION_LIMIT[0], id_, 0)
        bus.w2(MAX_POSITION_LIMIT[0], id_, RESOLUTION - 1)

    input("Move the arm to the MIDDLE of every joint's range, then press ENTER...")
    homing = {}
    for joint, id_ in ids.items():
        offset = half_turn_homing_offset(bus.present(id_))
        homing[joint] = offset
        bus.w2(HOMING_OFFSET[0], id_, encode_sign_magnitude(offset, HOMING_OFFSET_SIGN_BIT))

    sweep = [j for j in JOINTS if j != full_turn_motor]
    print(f"\nMove {', '.join(sweep)} through their FULL range of motion.")
    print("Recording min/max. Press ENTER to stop...")
    mins = {j: bus.present(ids[j]) for j in sweep}
    maxes = dict(mins)
    while not enter_pressed():
        for j in sweep:
            p = bus.present(ids[j])
            mins[j] = min(mins[j], p)
            maxes[j] = max(maxes[j], p)
        print("  " + "  ".join(f"{j}:[{mins[j]},{maxes[j]}]" for j in sweep), end="\r", flush=True)
    print()

    if full_turn_motor:
        mins[full_turn_motor] = 0
        maxes[full_turn_motor] = RESOLUTION - 1

    cal = {}
    for joint, id_ in ids.items():
        cal[joint] = MotorCalibration(
            id=id_, drive_mode=0, homing_offset=homing[joint],
            range_min=mins[joint], range_max=maxes[joint],
        )
        # Persist limits to the motor so the on-motor state matches the file.
        bus.w2(MIN_POSITION_LIMIT[0], id_, mins[joint])
        bus.w2(MAX_POSITION_LIMIT[0], id_, maxes[joint])
    bus.port.closePort()
    return cal


def main():
    ap = argparse.ArgumentParser(description="Calibrate one SO-101 arm.")
    ap.add_argument("--arm", required=True, choices=["follower", "leader"])
    ap.add_argument("--out", default="calibration", help="Output directory")
    # Leader wrist_roll spins freely (full turn); follower's is range-limited.
    ap.add_argument("--full-turn", default=None,
                    help="Joint to treat as full-turn (default: wrist_roll for leader, none for follower)")
    args = ap.parse_args()

    full_turn = args.full_turn or ("wrist_roll" if args.arm == "leader" else None)
    cal = calibrate(f"/dev/so101-{args.arm}", full_turn)

    out_path = Path(args.out) / f"so101_{args.arm}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(out_path, cal)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
