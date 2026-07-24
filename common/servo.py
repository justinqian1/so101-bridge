"""Feetech STS3215 bus: connect, apply calibration, sync-read/write joints.

Vendored register map + sync-read/write pattern from LeRobot's FeetechMotorsBus (§8),
trimmed to what capture and inference need. Talks to `scservo_sdk` directly, applies our
own calibration (common/calibration.py), and exposes normalised numpy arrays in JOINTS
order.

Pi-side. Depends only on numpy + feetech-servo-sdk (`scservo_sdk`). No torch, no lerobot.
"""

from __future__ import annotations

import numpy as np
import scservo_sdk as scs

from common import calibration as cal
from common.calibration import MotorCalibration
from common.schema import JOINTS, MOTOR_IDS

# STS3215 control table (feetech tables.py): name -> (address, size_bytes).
TORQUE_ENABLE = (40, 1)
LOCK = (55, 1)
GOAL_POSITION = (42, 2)
PRESENT_POSITION = (56, 2)
HOMING_OFFSET = (31, 2)
MIN_POSITION_LIMIT = (9, 2)
MAX_POSITION_LIMIT = (11, 2)
OPERATING_MODE = (33, 1)
RETURN_DELAY_TIME = (7, 1)
ACCELERATION = (41, 1)
MAX_ACCELERATION = (85, 1)
P_COEFFICIENT = (21, 1)
I_COEFFICIENT = (23, 1)
D_COEFFICIENT = (22, 1)
PHASE = (18, 1)

DEFAULT_BAUDRATE = 1_000_000
POSITION_MODE = 0


class ServoBus:
    """One SO-101 arm (6× STS3215) on a USB-TTL adapter."""

    def __init__(self, port: str, calibration: dict[str, MotorCalibration], baudrate: int = DEFAULT_BAUDRATE):
        cal.assert_joint_order(calibration)
        self.port = port
        self.baudrate = baudrate
        self.calibration = calibration
        self.ids = [MOTOR_IDS[j] for j in JOINTS]
        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)  # protocol 0 for STS/SMS series
        self._reader = scs.GroupSyncRead(self._port, self._packet, *PRESENT_POSITION)
        self._writer = scs.GroupSyncWrite(self._port, self._packet, *GOAL_POSITION)
        for id_ in self.ids:
            self._reader.addParam(id_)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self, apply_calibration: bool = True) -> None:
        if not self._port.openPort():
            raise ConnectionError(f"Failed to open servo port {self.port}")
        if not self._port.setBaudRate(self.baudrate):
            raise ConnectionError(f"Failed to set baudrate {self.baudrate} on {self.port}")
        if apply_calibration:
            self.write_calibration()

    def close(self) -> None:
        self._port.closePort()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disable_torque()
        self.close()

    # ── register helpers ─────────────────────────────────────────────────────

    def _write_reg(self, reg: tuple[int, int], id_: int, value: int) -> None:
        addr, size = reg
        if size == 1:
            comm, err = self._packet.write1ByteTxRx(self._port, id_, addr, value)
        else:
            comm, err = self._packet.write2ByteTxRx(self._port, id_, addr, value)
        if comm != scs.COMM_SUCCESS or err != 0:
            raise ConnectionError(f"write reg {reg} id={id_}: {self._packet.getTxRxResult(comm)}")

    def _read_reg(self, reg: tuple[int, int], id_: int) -> int:
        addr, size = reg
        if size == 1:
            value, comm, err = self._packet.read1ByteTxRx(self._port, id_, addr)
        else:
            value, comm, err = self._packet.read2ByteTxRx(self._port, id_, addr)
        if comm != scs.COMM_SUCCESS or err != 0:
            raise ConnectionError(f"read reg {reg} id={id_}: {self._packet.getTxRxResult(comm)}")
        return value

    # ── configuration ────────────────────────────────────────────────────────

    def write_calibration(self) -> None:
        """Push homing offset + position limits to each motor so Present_Position is
        already homed. Then reads normalise against range_min/range_max directly."""
        for joint, id_ in zip(JOINTS, self.ids):
            c = self.calibration[joint]
            self._write_reg(HOMING_OFFSET, id_, cal.encode_sign_magnitude(c.homing_offset, cal.HOMING_OFFSET_SIGN_BIT))
            self._write_reg(MIN_POSITION_LIMIT, id_, c.range_min)
            self._write_reg(MAX_POSITION_LIMIT, id_, c.range_max)

    def configure(self) -> None:
        """One-time servo setup: position mode, fast accel, PID (matches so_follower)."""
        self.disable_torque()
        for joint, id_ in zip(JOINTS, self.ids):
            # Force angle feedback mode 0 (clear Phase bit 4) so positions stay in [0, res-1].
            phase = self._read_reg(PHASE, id_)
            if phase & 0x10:
                self._write_reg(PHASE, id_, phase & ~0x10)
            self._write_reg(RETURN_DELAY_TIME, id_, 0)
            self._write_reg(MAX_ACCELERATION, id_, 254)
            self._write_reg(ACCELERATION, id_, 254)
            self._write_reg(OPERATING_MODE, id_, POSITION_MODE)
            self._write_reg(P_COEFFICIENT, id_, 16)
            self._write_reg(I_COEFFICIENT, id_, 0)
            self._write_reg(D_COEFFICIENT, id_, 32)

    def enable_torque(self) -> None:
        for id_ in self.ids:
            self._write_reg(TORQUE_ENABLE, id_, 1)
            self._write_reg(LOCK, id_, 1)

    def disable_torque(self) -> None:
        for id_ in self.ids:
            self._write_reg(TORQUE_ENABLE, id_, 0)
            self._write_reg(LOCK, id_, 0)

    # ── hot path ─────────────────────────────────────────────────────────────

    def read_positions(self, num_retry: int = 1) -> np.ndarray:
        """Sync-read all joints -> normalised float array in JOINTS order."""
        for attempt in range(num_retry + 1):
            comm = self._reader.txRxPacket()
            if comm == scs.COMM_SUCCESS:
                break
        else:
            raise ConnectionError(f"sync read failed: {self._packet.getTxRxResult(comm)}")

        out = np.empty(len(JOINTS), dtype=np.float32)
        addr, size = PRESENT_POSITION
        for i, (joint, id_) in enumerate(zip(JOINTS, self.ids)):
            raw = self._reader.getData(id_, addr, size)
            raw = cal.decode_sign_magnitude(raw, cal.POSITION_SIGN_BIT)
            out[i] = cal.normalize(joint, raw, self.calibration[joint])
        return out

    def write_positions(self, values: np.ndarray) -> None:
        """Write normalised goal positions (JOINTS order) via sync-write."""
        self._writer.clearParam()
        for joint, id_, value in zip(JOINTS, self.ids, values):
            raw = cal.unnormalize(joint, float(value), self.calibration[joint])
            raw = cal.encode_sign_magnitude(raw, cal.POSITION_SIGN_BIT)
            self._writer.addParam(id_, [scs.SCS_LOBYTE(raw), scs.SCS_HIBYTE(raw)])
        comm = self._writer.txPacket()
        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(f"sync write failed: {self._packet.getTxRxResult(comm)}")
