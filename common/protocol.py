"""ZMQ REQ/REP wire format shared by the Pi client and desktop server (§5.3).

  Pi  -> desktop:  (jpeg_bytes[], joint_state, seq, t_monotonic)
  desktop -> Pi:   action chunk, seq echoed

Multipart framing: a small JSON header frame + raw binary frames (float32 state /
actions, concatenated JPEGs). Kept in one file so both sides encode identically.

Stdlib + numpy only — safe on the Pi. No image decode, no torch.
"""

from __future__ import annotations

import json

import numpy as np


def pack_request(seq: int, t_monotonic: float, state: np.ndarray, frames: dict[str, bytes]) -> list[bytes]:
    cams = list(frames.keys())
    header = {"seq": seq, "t": t_monotonic, "cams": cams, "n_state": int(state.shape[0])}
    parts = [json.dumps(header).encode(), np.asarray(state, dtype=np.float32).tobytes()]
    parts += [frames[name] for name in cams]
    return parts


def unpack_request(parts: list[bytes]) -> tuple[int, float, np.ndarray, dict[str, bytes]]:
    header = json.loads(parts[0])
    state = np.frombuffer(parts[1], dtype=np.float32)
    frames = {name: parts[2 + i] for i, name in enumerate(header["cams"])}
    return header["seq"], header["t"], state, frames


def pack_reply(seq: int, actions: np.ndarray) -> list[bytes]:
    actions = np.asarray(actions, dtype=np.float32)
    header = {"seq": seq, "shape": list(actions.shape)}
    return [json.dumps(header).encode(), actions.tobytes()]


def unpack_reply(parts: list[bytes]) -> tuple[int, np.ndarray]:
    header = json.loads(parts[0])
    actions = np.frombuffer(parts[1], dtype=np.float32).reshape(header["shape"])
    return header["seq"], actions
