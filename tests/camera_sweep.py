#!/usr/bin/env python3
"""Sweep one v4l2 control and save a frame per value, for picking locked settings.

CONTROLS in common/cameras.py carries two values that can't be derived from a datasheet
— ext's focus_absolute and wrist's exposure_time_absolute. Both are scene-dependent, and
both read back `inactive` while their automatic is still running, so there is nothing to
copy off a converged camera. This holds every other control at its locked value, steps
the one under test across its range, and writes one JPEG per step to pick from by eye.

JPEG size doubles as an objective focus metric: a defocused 640x480 frame loses its high
frequencies and compresses to roughly a quarter of its sharp size, so for focus_absolute
the largest file is the sharpest. Exposure has no such shortcut — look at the images.

Runs one ffmpeg for the whole sweep rather than one per value: a fresh stream costs
0.3–1.1s of startup latency, which would dominate.

Pi-side. Stdlib only; JPEGs are copied byte-for-byte, never decoded (§0).

Usage
-----
    python tests/camera_sweep.py --camera ext --control focus_absolute
    python tests/camera_sweep.py --camera wrist --control exposure_time_absolute \
        --values 20,40,60,80,120,160,220,300,400,550,750,1000
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from common.cameras import CONTROLS, FrameStream


def _v4l2(device: str, *args: str) -> str:
    result = subprocess.run(["v4l2-ctl", "-d", device, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"v4l2-ctl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def control_range(device: str, control: str) -> tuple[int, int, int]:
    """(min, max, step) as the device reports them."""
    for line in _v4l2(device, "--list-ctrls").splitlines():
        if line.strip().startswith(control + " "):
            found = {k: int(v) for k, v in re.findall(r"(min|max|step)=(-?\d+)", line)}
            if "min" in found and "max" in found:
                return found["min"], found["max"], found.get("step", 1)
    raise SystemExit(f"{device} has no control '{control}'")


def read_control(device: str, control: str) -> int:
    return int(_v4l2(device, "-C", control).split(":")[1])


def sweep_values(lo: int, hi: int, step: int, count: int) -> list[int]:
    """`count` values across [lo, hi], snapped to the control's step, deduplicated."""
    span = hi - lo
    raw = [lo + round(span * i / max(count - 1, 1)) for i in range(count)]
    snapped = [lo + round((v - lo) / step) * step for v in raw]
    return sorted(set(min(hi, max(lo, v)) for v in snapped))


def main():
    ap = argparse.ArgumentParser(description="Sweep a v4l2 control, one saved frame per value.")
    ap.add_argument("--camera", required=True, choices=sorted(CONTROLS), help="Camera name")
    ap.add_argument("--control", required=True, help="Control to sweep, e.g. focus_absolute")
    ap.add_argument("--values", help="Explicit comma-separated values (overrides --count)")
    ap.add_argument("--count", type=int, default=16, help="Samples across the device's range")
    ap.add_argument("--settle", type=float, default=0.6,
                    help="Seconds to wait after each change (focus motors need the time)")
    ap.add_argument("--out", default="sweeps", help="Output directory")
    args = ap.parse_args()

    device = f"/dev/v4l/by-id/cam-{args.camera}"
    lo, hi, step = control_range(device, args.control)
    if args.values:
        values = [int(v) for v in args.values.split(",")]
        out_of_range = [v for v in values if not lo <= v <= hi]
        if out_of_range:
            raise SystemExit(f"{args.control} accepts {lo}..{hi}; out of range: {out_of_range}")
    else:
        values = sweep_values(lo, hi, step, args.count)

    out_dir = Path(args.out) / f"{args.camera}_{args.control}"
    out_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(max(values)))

    print(f"{args.camera} ({device})")
    print(f"{args.control}: device range {lo}..{hi} step {step}")
    print(f"sweeping {len(values)} values, {args.settle}s settle -> {out_dir}\n")

    # FrameStream.start() applies the locked CONTROLS table first, so everything except
    # the control under test is pinned to what recording will actually use.
    stream = FrameStream(args.camera, device)
    stream.start()
    results: list[tuple[int, int]] = []
    try:
        deadline = time.monotonic() + 10
        while stream.latest() is None:
            if time.monotonic() > deadline:
                raise SystemExit(f"no frames from {device} after 10s")
            time.sleep(0.05)

        for value in values:
            _v4l2(device, "-c", f"{args.control}={value}")
            time.sleep(args.settle)
            frame = stream.latest()
            actual = read_control(device, args.control)

            path = out_dir / f"{args.control}_{value:0{width}d}.jpg"
            path.write_bytes(frame)
            results.append((value, len(frame)))

            # A camera that resets controls on STREAMON, or snaps to its own grid, makes
            # every frame in the sweep a duplicate of one setting. Say so per value.
            note = "" if actual == value else f"  <-- device reports {actual}"
            print(f"  {args.control}={value:>{width}}  {len(frame)/1024:6.1f} KB  {path.name}{note}")
    finally:
        stream.stop()

    if len(set(size for _, size in results)) == 1:
        print("\nEvery frame is the same size — the control is probably not taking effect.")
        return
    best, best_size = max(results, key=lambda r: r[1])
    print(f"\nLargest frame: {args.control}={best} at {best_size/1024:.1f} KB")
    print("For focus_absolute that is the sharpest setting. For exposure it only means "
          "brightest/noisiest — inspect the images.")
    print(f"\n  scp -r pi:{out_dir} .   then set the winner in CONTROLS (common/cameras.py)")


if __name__ == "__main__":
    main()
