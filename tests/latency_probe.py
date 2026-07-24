#!/usr/bin/env python3
"""
latency_probe.py -- go/no-go test for the "dumb bridge" design.

Question it answers
-------------------
If the Pi is nothing but a socat serial bridge and stock LeRobot runs on the
desktop, the network sits *inside* the servo control loop: every 33ms tick is a
sync-read plus a sync-write across campus. The Feetech SDK's timeouts assume
local latency. Does the bridged path fit inside the budget, including the tail?

This measures one sync-read transaction, repeatedly, and reports the
distribution. Run it locally on the Pi for a baseline, then on the desktop
through the socat bridge, and compare.

Usage
-----
    # Pi, local baseline
    python latency_probe.py --port /dev/ttyACM0 --ids 1,2,3,4,5,6 --duration 60

    # Desktop, through the bridge
    python latency_probe.py --port /tmp/ttyV0 --ids 1,2,3,4,5,6 --duration 600

Bridge setup
------------
    # Pi
    socat tcp-listen:5000,reuseaddr,nodelay file:/dev/ttyACM0,raw,echo=0,b1000000
    # desktop
    socat pty,link=/tmp/ttyV0,raw,echo=0 tcp:pi-hostname:5000

Requires only feetech-servo-sdk. No torch, no lerobot, no opencv. Runs on
either machine.

Safety: read-only. Never writes to a servo, never enables torque.
"""

import argparse
import json
import statistics
import sys
import time

try:
    from scservo_sdk import (
        PortHandler,
        PacketHandler,
        GroupSyncRead,
        COMM_SUCCESS,
    )
except ImportError:
    sys.exit("Need feetech-servo-sdk:  pip install feetech-servo-sdk")


# STS3215 register map. Verify against your servo's datasheet -- Feetech
# reuses model names across firmware revisions and these do occasionally move.
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2
PROTOCOL_END = 0  # STS/SMS series is little-endian

# Control loop budget. A real tick is a sync-read AND a sync-write, so a single
# transaction gets roughly a third of the period once you leave headroom for
# policy/logging work.
TICK_PERIOD_MS = 1000.0 / 30.0
SINGLE_TXN_BUDGET_MS = 12.0
COMFORTABLE_MS = 8.0


def percentile(sorted_vals, q):
    """Nearest-rank percentile. Avoids a numpy dependency."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round(q / 100.0 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[k]


def histogram(vals, bins=(1, 2, 3, 5, 8, 12, 20, 33, 100)):
    lines = []
    total = len(vals)
    if not total:
        return "  (no samples)"
    prev = 0.0
    counts = []
    for b in bins:
        counts.append(sum(1 for v in vals if prev <= v < b))
        prev = b
    counts.append(sum(1 for v in vals if v >= bins[-1]))
    labels = []
    prev = 0.0
    for b in bins:
        labels.append(f"{prev:>5.0f}-{b:<5.0f}ms")
        prev = b
    labels.append(f"{bins[-1]:>5.0f}+     ms")

    width = 48
    peak = max(counts) or 1
    for label, c in zip(labels, counts):
        bar = "#" * int(width * c / peak)
        pct = 100.0 * c / total
        lines.append(f"  {label} | {bar:<{width}} {c:>7d}  {pct:5.1f}%")
    return "\n".join(lines)


def run(port_name, baud, ids, duration, rate_hz, warmup):
    port = PortHandler(port_name)
    packet = PacketHandler(PROTOCOL_END)

    if not port.openPort():
        sys.exit(f"Failed to open {port_name}")
    if not port.setBaudRate(baud):
        sys.exit(f"Failed to set baud {baud}")

    # Keep the SDK's own timeout well under the tick budget so a stall shows up
    # as a counted failure rather than as a multi-second freeze.
    port.setPacketTimeoutMillis(20)

    sync = GroupSyncRead(port, packet, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for sid in ids:
        if not sync.addParam(sid):
            sys.exit(f"addParam failed for servo id {sid}")

    print(f"port         {port_name}")
    print(f"baud         {baud}")
    print(f"servo ids    {ids}")
    print(f"transaction  sync-read present position, {len(ids)} servos")
    print(f"rate         {rate_hz} Hz")
    print(f"duration     {duration}s  (+{warmup}s warmup, discarded)")
    print()

    latencies = []
    comm_failures = 0
    missing_data = 0
    period = 1.0 / rate_hz

    t_start = time.monotonic()
    t_measure_from = t_start + warmup
    next_tick = t_start
    last_report = t_start

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t_start
            if elapsed >= duration + warmup:
                break

            t0 = time.perf_counter()
            comm = sync.txRxPacket()
            t1 = time.perf_counter()

            measuring = now >= t_measure_from
            dt_ms = (t1 - t0) * 1000.0

            if comm != COMM_SUCCESS:
                if measuring:
                    comm_failures += 1
            else:
                incomplete = any(
                    not sync.isAvailable(sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    for sid in ids
                )
                if incomplete:
                    if measuring:
                        missing_data += 1
                elif measuring:
                    latencies.append(dt_ms)

            # Progress line once a second, so a 10-minute run isn't a blank screen.
            if now - last_report >= 1.0:
                last_report = now
                if latencies:
                    recent = latencies[-int(rate_hz) :]
                    tag = "measuring" if measuring else "  warmup "
                    print(
                        f"\r  [{tag}] {elapsed:6.1f}s  "
                        f"n={len(latencies):<7d} "
                        f"last-1s mean={statistics.mean(recent):5.2f}ms "
                        f"max={max(recent):6.2f}ms  "
                        f"fail={comm_failures + missing_data}",
                        end="",
                        flush=True,
                    )

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind. Resync rather than accumulating debt.
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        port.closePort()

    print("\n")
    return latencies, comm_failures, missing_data


def report(latencies, comm_failures, missing_data, port_name, ids, duration, rate_hz):
    n = len(latencies)
    failures = comm_failures + missing_data
    attempted = n + failures

    if n == 0:
        print("No successful transactions. Check wiring, ids, baud rate, and that")
        print("socat is running on both ends.")
        return 2

    s = sorted(latencies)
    p50, p95, p99, p999 = (percentile(s, q) for q in (50, 95, 99, 99.9))
    mx = s[-1]

    print("=" * 68)
    print("LATENCY  (single sync-read transaction)")
    print("=" * 68)
    print(f"  samples      {n}")
    print(f"  failures     {failures}  ({100.0*failures/max(attempted,1):.3f}%)"
          f"   [comm={comm_failures} incomplete={missing_data}]")
    print()
    print(f"  min          {s[0]:7.2f} ms")
    print(f"  mean         {statistics.mean(s):7.2f} ms")
    print(f"  p50          {p50:7.2f} ms")
    print(f"  p95          {p95:7.2f} ms")
    print(f"  p99          {p99:7.2f} ms")
    print(f"  p99.9        {p999:7.2f} ms")
    print(f"  max          {mx:7.2f} ms")
    print()
    print(histogram(s))
    print()

    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"  Tick period at 30 Hz:            {TICK_PERIOD_MS:.1f} ms")
    print(f"  A real tick = read + write, so one transaction should stay")
    print(f"  under ~{SINGLE_TXN_BUDGET_MS:.0f} ms, comfortably under ~{COMFORTABLE_MS:.0f} ms.")
    print()

    if failures > 0:
        verdict = "FAIL"
        note = (f"{failures} failed transaction(s). Any dropout kills the dumb-bridge\n"
                "  design -- the failure mode is a servo timeout mid-motion.\n"
                "  Write pi/run_policy.py.")
        code = 1
    elif p99 > SINGLE_TXN_BUDGET_MS:
        verdict = "FAIL"
        note = ("p99 exceeds the single-transaction budget. The network does not\n"
                "  belong inside the servo loop. Write pi/run_policy.py.")
        code = 1
    elif p99 > COMFORTABLE_MS:
        verdict = "MARGINAL"
        note = ("Fits on a good day with no margin for a retransmit. Write\n"
                "  pi/run_policy.py -- the 70 lines are cheaper than debugging a\n"
                "  mid-motion timeout later.")
        code = 1
    else:
        verdict = "PASS"
        note = ("Dumb bridge looks viable on this run. Next: run stock\n"
                "  lerobot-teleoperate against this port for a full session and watch\n"
                "  the loop-time histogram -- this probe only tests reads, and the\n"
                "  real access pattern includes writes and larger packets.\n"
                "  Re-run at a busy time of day before trusting it.")
        code = 0

    print(f"  ==> {verdict}")
    print(f"  {note}")
    print()

    if duration < 300:
        print("  NOTE: short run. The tail is the whole point here -- repeat with")
        print("  --duration 600 or more before making a decision.")
        print()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = f"latency_probe_{stamp}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "port": port_name,
                "ids": ids,
                "rate_hz": rate_hz,
                "duration_s": duration,
                "n": n,
                "comm_failures": comm_failures,
                "missing_data": missing_data,
                "min_ms": s[0],
                "mean_ms": statistics.mean(s),
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "p999_ms": p999,
                "max_ms": mx,
                "verdict": verdict,
                "samples_ms": [round(v, 4) for v in latencies],
            },
            f,
        )
    print(f"  raw samples -> {out}")
    print("  (compare the Pi-local baseline against the bridged run)")
    return code


def main():
    ap = argparse.ArgumentParser(
        description="Measure Feetech sync-read latency, locally or through a socat bridge."
    )
    ap.add_argument("--port", required=True, help="/dev/ttyACM0 on the Pi, /tmp/ttyV0 on the desktop")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="1,2,3,4,5,6", help="comma-separated servo ids")
    ap.add_argument("--duration", type=float, default=60.0, help="measured seconds")
    ap.add_argument("--warmup", type=float, default=3.0, help="discarded seconds")
    ap.add_argument("--rate", type=float, default=30.0, help="probe rate in Hz")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    latencies, comm_failures, missing_data = run(
        args.port, args.baud, ids, args.duration, args.rate, args.warmup
    )
    sys.exit(
        report(latencies, comm_failures, missing_data, args.port, ids, args.duration, args.rate)
    )


if __name__ == "__main__":
    main()