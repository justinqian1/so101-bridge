"""ZMQ DEALER/ROUTER wire format shared by the Pi client and desktop server (§5.3).

  Pi  -> desktop:  TimedObservation (jpegs, joint state, timestep, must_go, delay)
  desktop -> Pi:   TimedChunk (action chunk + the absolute timestep of its first action)

The Pi streams observations whenever its action queue drains past a threshold and receives
chunks on a background thread, so a request can be in flight while the arm is still executing
the previous chunk. DEALER/ROUTER rather than REQ/REP because REQ forbids that overlap.

`timestep` is an absolute action index, counted by the Pi from the start of the run and
shared by both sides. It is the only thing keeping the actions the Pi has executed and
the RTC prefix the desktop holds aligned, so it is on every message in both directions.

Observations are streamed, not requested: the Pi sends one on every tick once its queue
has drained past the threshold, and the desktop replies only to the ones it actually
runs. Silence means the observation was filtered as redundant or displaced by a newer
one, which costs nothing — a fresher observation is already on its way (lerobot
`async_inference`).

Multipart framing: a small JSON header frame + raw binary frames (float32 state /
actions, concatenated JPEGs). Kept in one file so both sides encode identically.

Stdlib + numpy only — safe on the Pi. No image decode, no torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimedObservation:
    """One observation, tagged with where the Pi is in its action queue."""

    seq: int
    t_monotonic: float
    timestep: int                 # absolute index of the next action to be executed
    state: np.ndarray
    frames: dict[str, bytes] = field(default_factory=dict)
    must_go: bool = False         # queue is empty — run inference on this one regardless
    delay: int = 0                # actions the Pi expects to execute before the reply lands


@dataclass
class TimedChunk:
    """One action chunk, tagged with the absolute timestep of its first action."""

    seq: int
    timestep: int
    actions: np.ndarray           # (T, action_dim), float32
    rtc: bool = False             # server ran Real-Time Chunking; merge by replacement


def pack_request(obs: TimedObservation) -> list[bytes]:
    cams = list(obs.frames.keys())
    header = {
        "seq": obs.seq,
        "t": obs.t_monotonic,
        "timestep": obs.timestep,
        "must_go": obs.must_go,
        "delay": obs.delay,
        "cams": cams,
        "n_state": int(obs.state.shape[0]),
    }
    parts = [json.dumps(header).encode(), np.asarray(obs.state, dtype=np.float32).tobytes()]
    parts += [obs.frames[name] for name in cams]
    return parts


def unpack_request(parts: list[bytes]) -> TimedObservation:
    header = json.loads(parts[0])
    return TimedObservation(
        seq=header["seq"],
        t_monotonic=header["t"],
        timestep=header["timestep"],
        state=np.frombuffer(parts[1], dtype=np.float32),
        frames={name: parts[2 + i] for i, name in enumerate(header["cams"])},
        must_go=header["must_go"],
        delay=header["delay"],
    )


def pack_reply(chunk: TimedChunk) -> list[bytes]:
    actions = np.asarray(chunk.actions, dtype=np.float32)
    header = {"seq": chunk.seq, "timestep": chunk.timestep, "shape": list(actions.shape), "rtc": chunk.rtc}
    return [json.dumps(header).encode(), actions.tobytes()]


def unpack_reply(parts: list[bytes]) -> TimedChunk:
    header = json.loads(parts[0])
    return TimedChunk(
        seq=header["seq"],
        timestep=header["timestep"],
        actions=np.frombuffer(parts[1], dtype=np.float32).reshape(header["shape"]),
        rtc=header["rtc"],
    )
