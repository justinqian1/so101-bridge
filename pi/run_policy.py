"""Inference client: stream observations to the desktop, execute action chunks (§5.4).

Async, not lockstep. A network thread streams observations up and receives chunks as
they land; the control loop pops one action per tick from a shared queue and never
blocks on the round trip. Ported from LeRobot's `async_inference.robot_client` and the
client half of `rollout.inference.rtc.RTCInferenceEngine`, with the queue reimplemented
on numpy — the Pi imports no torch and no lerobot (§0).

The two merge modes are LeRobot's, chosen by the server and announced on every chunk:

  rtc=True   Real-Time Chunking. The server already blended the new chunk against the
             tail we had not executed yet, so the queue is replaced outright.
  rtc=False  Plain async chunking. Overlapping timesteps are blended here instead,
             weighted towards the newer chunk.

Either way `timestep` — the absolute action index — decides what overlaps what, so a
late chunk is spliced in at the right place rather than restarting motion mid-stroke.

Observations stream on every tick once the queue drains past the threshold, and the
server answers only the ones it runs — silence just means a fresher observation is
already on its way, so there is no timeout to wait out.

Safety behaviour around the loop:
  - queue starved  -> hold the last commanded position, never continue a stale chunk
  - queue empty    -> the observation goes out as must-go and bypasses the server filter
  - exit (any path) -> disable torque

Pi-side. numpy + pyzmq + feetech-servo-sdk. No decode, no torch (§0).
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from collections import deque
from queue import Empty, Full, Queue

import numpy as np
import zmq

from common.calibration import load_calibration
from common.cameras import open_streams
from common.protocol import TimedChunk, TimedObservation, pack_request, unpack_reply
from common.schema import CONTROL_HZ
from common.servo import ServoBus

# Fixed device aliases from port-alias-setup.md — not configurable per-run.
CAMERAS = {
    "ext": "/dev/v4l/by-id/cam-ext",
    "wrist": "/dev/v4l/by-id/cam-wrist",
}

# Non-RTC blend of overlapping actions (lerobot's default `weighted_average`).
AGGREGATE_OLD, AGGREGATE_NEW = 0.3, 0.7
# Joint units. Jumps larger than this across a chunk boundary get logged (§5.4).
DISCONTINUITY_ATOL = 5.0
# Socket thread poll timeout.
POLL_MS = 5


class Rate:
    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._next = time.monotonic()

    def sleep(self):
        self._next += self.period
        dt = self._next - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        else:
            self._next = time.monotonic()  # fell behind; resync


class LatencyTracker:
    """Sliding window of round-trip times (lerobot policies/rtc/latency_tracker.py)."""

    def __init__(self, maxlen: int = 100):
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, latency: float) -> None:
        if latency >= 0:
            self._values.append(float(latency))

    def p95(self) -> float:
        """95th percentile, not the max: one Wi-Fi stall should not inflate the estimate
        for the rest of the run (lerobot runs in-process, where max() is safe)."""
        if not self._values:
            return 0.0
        return float(np.quantile(np.asarray(self._values, dtype=np.float32), 0.95))


class ActionQueue:
    """Thread-safe action queue indexed by absolute timestep.

    numpy port of lerobot policies/rtc/action_queue.py. The one change: actions carry
    an absolute timestep rather than a queue-relative index, because the chunk arrives
    from another machine and has to be spliced in against actions already executed.
    """

    def __init__(self):
        self.queue: np.ndarray | None = None   # (T, action_dim), postprocessed
        self.start = 0                         # absolute timestep of queue[0]
        self.last_index = 0
        self.chunk_size = 1                    # largest chunk seen, for the send threshold
        self.lock = threading.Lock()

    def get(self) -> np.ndarray | None:
        """Pop the next action, or None if the queue is drained."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        return self.qsize() <= 0

    def next_timestep(self) -> int:
        """Absolute timestep of the action that will be executed next."""
        with self.lock:
            return self.start + self.last_index

    def merge(self, chunk: TimedChunk) -> None:
        """Splice an incoming chunk into the queue at its absolute timestep."""
        with self.lock:
            self.chunk_size = max(self.chunk_size, len(chunk.actions))
            next_timestep = self.start + self.last_index
            # Actions executed since the observation behind this chunk was captured.
            delay = max(0, next_timestep - chunk.timestep)
            if delay >= len(chunk.actions):
                print(f"[stale] chunk #{chunk.timestep} overtaken by {delay} actions, dropped")
                return

            incoming = chunk.actions[delay:]
            tail = self.queue[self.last_index :] if self.queue is not None else None

            if chunk.rtc or tail is None or len(tail) == 0:
                # RTC: the server already blended against `tail`; replacing is the point.
                merged = incoming
            else:
                overlap = min(len(tail), len(incoming))
                blended = AGGREGATE_OLD * tail[:overlap] + AGGREGATE_NEW * incoming[:overlap]
                remainder = incoming[overlap:] if len(incoming) > overlap else tail[overlap:]
                merged = np.concatenate([blended, remainder])

            if tail is not None and len(tail) and len(merged):
                jump = float(np.max(np.abs(merged[0] - tail[0])))
                if jump > DISCONTINUITY_ATOL:
                    print(f"[jump] {jump:.1f} across chunk boundary at #{next_timestep}")

            self.queue = np.ascontiguousarray(merged, dtype=np.float32)
            self.start = next_timestep
            self.last_index = 0


class PolicyClient(threading.Thread):
    """Owns the DEALER socket. Sends whatever the control loop hands it, merges replies.

    All socket traffic is on this thread: ZMQ sockets are not thread-safe, and the
    control loop must not block on the network.
    """

    def __init__(self, endpoint: str, queue: ActionQueue, chunk_size_threshold: float):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.action_queue = queue
        self.chunk_size_threshold = chunk_size_threshold
        self.outbox: Queue = Queue(maxsize=1)
        self.latency = LatencyTracker()
        self.shutdown_event = threading.Event()
        self._seq = 0
        # timestep -> monotonic send time, for round-trip timing. Several observations
        # can be outstanding at once, so this is a map rather than a single slot.
        self._sent_at: dict[int, float] = {}
        self._sent_lock = threading.Lock()

    # ── control-loop side ──────────────────────────────────────────────────────

    def ready_to_send(self) -> bool:
        """True once the queue has drained past the threshold.

        lerobot's `_ready_to_send_observation`, and nothing more: from that point an
        observation goes out on every tick. Holding one back to wait for a reply would
        only add dead air, since the server keeps just the newest observation anyway and
        answers whichever one it happens to be holding when the GPU frees up.
        """
        return self.action_queue.qsize() / self.action_queue.chunk_size <= self.chunk_size_threshold

    def send(self, state: np.ndarray, frames: dict[str, bytes]) -> None:
        """Hand an observation to the socket thread. Never blocks the control loop."""
        self._seq += 1
        timestep = self.action_queue.next_timestep()
        obs = TimedObservation(
            seq=self._seq,
            t_monotonic=time.monotonic(),
            timestep=timestep,
            state=state,
            frames=frames,
            # Empty queue: the arm is holding position, so this one has to be run whatever
            # the server's redundancy filter thinks. lerobot additionally gates this on an
            # event cleared per must-go and re-armed only when a chunk arrives — dropped
            # here, because a chunk that never arrives then leaves the event cleared, the
            # timestep frozen, and every later observation filtered out for good. The
            # server's single-slot observation queue already keeps this from becoming a
            # stream of forced inferences.
            must_go=self.action_queue.empty(),
            delay=math.ceil(self.latency.p95() * CONTROL_HZ),
        )
        try:
            self.outbox.put_nowait(obs)
        except Full:
            return  # socket thread has not drained the last one yet; skip this tick
        with self._sent_lock:
            self._sent_at[timestep] = time.monotonic()

    # ── socket thread ──────────────────────────────────────────────────────────

    def run(self) -> None:
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self.shutdown_event.is_set():
                try:
                    obs = self.outbox.get_nowait()
                    sock.send_multipart(pack_request(obs))
                except Empty:
                    pass
                if poller.poll(POLL_MS):
                    self._on_chunk(unpack_reply(sock.recv_multipart()))
        finally:
            sock.close(0)

    def _on_chunk(self, chunk: TimedChunk) -> None:
        with self._sent_lock:
            sent = self._sent_at.pop(chunk.timestep, None)
            # Older observations were filtered or displaced and will never be answered:
            # the server only ever runs the newest one it holds.
            for stale in [t for t in self._sent_at if t < chunk.timestep]:
                del self._sent_at[stale]
        if sent is not None:
            self.latency.add(time.monotonic() - sent)
        self.action_queue.merge(chunk)


def main():
    ap = argparse.ArgumentParser(description="SO-101 remote inference client.")
    ap.add_argument("--server", required=True, help="host:port of desktop policy server")
    ap.add_argument("--calib", default="calibration/so101_follower.json")
    ap.add_argument("--chunk-size-threshold", type=float, default=0.5,
                    help="Send the next observation once the queue drains below this fraction")
    args = ap.parse_args()

    streams = open_streams(CAMERAS)
    bus = ServoBus("/dev/so101-follower", load_calibration(args.calib))
    bus.connect()
    bus.configure()
    bus.enable_torque()

    action_queue = ActionQueue()
    client = PolicyClient(f"tcp://{args.server}", action_queue, args.chunk_size_threshold)
    client.start()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # SIGINT already raises

    print(f"[run] streaming to {args.server}, executing at {CONTROL_HZ}Hz")
    rate = Rate(CONTROL_HZ)
    hold: np.ndarray | None = None
    starved = False
    try:
        while True:
            action = action_queue.get()
            if action is not None:
                bus.write_positions(action)
                hold = action
                starved = False
            else:
                if not starved:
                    print("[starved] queue empty, holding position")
                    starved = True
                if hold is None:
                    hold = bus.read_positions(num_retry=1)
                bus.write_positions(hold)

            if client.ready_to_send():
                frames = {n: s.latest() for n, s in streams.items()}
                if all(f is not None for f in frames.values()):  # else cameras still warming up
                    client.send(bus.read_positions(num_retry=1), frames)

            rate.sleep()
    finally:
        client.shutdown_event.set()
        client.join(timeout=1.0)
        bus.disable_torque()
        bus.close()
        for s in streams.values():
            s.stop()


if __name__ == "__main__":
    main()
