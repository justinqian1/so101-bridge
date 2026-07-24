"""Inference client: stream observations to the desktop, execute action chunks (§5.4).

Query at ~POLICY_HZ, receive a chunk of actions, execute at CONTROL_HZ, and re-query
before the chunk drains — this amortises the round trip over half a second of motion,
which is why network latency mostly stops mattering.

The irreducible custom code is the timing loop and the safety behaviour around it:
  - query timeout   -> hold position, never continue a stale chunk
  - servo read fail -> retry once, then halt
  - SIGINT/SIGTERM  -> disable torque cleanly
  - chunk boundary discontinuity -> log (policy may be seeing a stale observation)

Pi-side. numpy + pyzmq + feetech-servo-sdk. No decode, no torch (§0).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time

import numpy as np
import zmq

from common.calibration import load_calibration
from common.cameras import open_streams
from common.protocol import pack_request, unpack_reply
from common.schema import CONTROL_HZ, POLICY_HZ, JOINTS
from common.servo import ServoBus


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


class PolicyClient:
    def __init__(self, endpoint: str, timeout_ms: int):
        self._ctx = zmq.Context.instance()
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._sock = self._connect()
        self._seq = 0

    def _connect(self) -> "zmq.Socket":
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        return sock

    def query(self, frames: dict[str, bytes], state: np.ndarray) -> np.ndarray | None:
        """Blocking request/response. Returns action chunk, or None on timeout."""
        self._seq += 1
        self._sock.send_multipart(pack_request(self._seq, time.monotonic(), state, frames))
        try:
            seq, actions = unpack_reply(self._sock.recv_multipart())
            return actions
        except zmq.Again:
            # REQ socket is stuck after a missed reply; reset it for the next query.
            self._sock.close(0)
            self._sock = self._connect()
            return None


def check_tailscale_direct(host: str) -> None:
    # A DERP relay can route across a distant region and add 40ms silently (§5.4).
    try:
        out = subprocess.check_output(["tailscale", "status"], text=True, timeout=5)
    except Exception:
        print("[warn] could not run 'tailscale status' — cannot confirm direct path")
        return
    for line in out.splitlines():
        if host in line:
            if "relay" in line:
                print(f"[warn] Tailscale peer '{host}' is via RELAY, not direct — expect +40ms")
            else:
                print(f"[tailscale] {host}: direct")
            return


def read_state_or_halt(bus: ServoBus) -> np.ndarray:
    try:
        return bus.read_positions(num_retry=1)
    except ConnectionError as e:
        raise SystemExit(f"[halt] servo read failed after retry: {e}")


def main():
    ap = argparse.ArgumentParser(description="SO-101 remote inference client.")
    ap.add_argument("--server", required=True, help="host:port of desktop policy server")
    ap.add_argument("--cam", action="append", default=[], help="name=/dev/v4l/by-id/... (repeatable)")
    ap.add_argument("--calib", default="calibration/so101_follower.json")
    ap.add_argument("--timeout-ms", type=int, default=500)
    ap.add_argument("--requery-at", type=int, default=15,
                    help="Actions to execute before re-querying (before the chunk drains)")
    args = ap.parse_args()

    host = args.server.split(":")[0]
    check_tailscale_direct(host)

    cameras = dict(c.split("=", 1) for c in args.cam)
    streams = open_streams(cameras)
    bus = ServoBus("/dev/so101-follower", load_calibration(args.calib))
    bus.connect()
    bus.configure()
    bus.enable_torque()

    client = PolicyClient(f"tcp://{args.server}", args.timeout_ms)

    def shutdown(*_):
        bus.disable_torque()
        bus.close()
        for s in streams.values():
            s.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"[run] querying {args.server} at ~{POLICY_HZ}Hz, executing at {CONTROL_HZ}Hz")
    rate = Rate(CONTROL_HZ)
    last_action: np.ndarray | None = None
    try:
        while True:
            frames = {n: s.latest() for n, s in streams.items()}
            if any(f is None for f in frames.values()):
                time.sleep(0.05)  # cameras not warmed up yet
                continue

            state = read_state_or_halt(bus)
            chunk = client.query(frames, state)
            if chunk is None:
                print("[timeout] holding position")
                bus.write_positions(state)  # hold, never execute a stale chunk
                continue

            # Chunk boundary discontinuity: large jump => policy may see a stale obs (§5.4).
            if last_action is not None:
                jump = float(np.max(np.abs(chunk[0] - last_action)))
                if jump > 20:
                    print(f"[warn] chunk boundary jump {jump:.1f} (normalised units)")

            for action in chunk[:args.requery_at]:
                bus.write_positions(action)
                last_action = action
                rate.sleep()
    except SystemExit:
        raise
    finally:
        shutdown()


if __name__ == "__main__":
    main()
