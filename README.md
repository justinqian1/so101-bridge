# so101-bridge

Data capture and remote VLA inference for an SO-101 arm. A Raspberry Pi 5 is the
robot-side machine; a GPU desktop does dataset writing, training, and inference. See
[SPEC.md](SPEC.md) for the full design.

**Governing rule (§0): the Pi never decodes an image and never imports torch.** JPEG
bytes go straight from camera to disk (capture) or onto the wire (inference). No
`opencv`, `lerobot`, `torch`, or `av` in the `[pi]` dependency set.

## Install

```bash
pip install -e .[pi]        # on the Raspberry Pi
pip install -e .[desktop]   # on the GPU desktop
```

## Layout

| Path | Runs on | Purpose |
|---|---|---|
| `common/schema.py` | both | joint order, rates, feature keys, on-disk layout |
| `common/calibration.py` | both | vendored homing/normalisation math + sign-magnitude |
| `common/servo.py` | Pi | Feetech STS3215 bus: read/write joints |
| `common/cameras.py` | Pi | ffmpeg passthrough — file sink (capture) + pipe sink (inference) |
| `common/preprocess.py` | desktop | JPEG → policy input; the one file that must not drift |
| `common/protocol.py` | both | ZMQ REQ/REP wire format |
| `pi/calibrate.py` | Pi | one-off, writes `calibration/so101_<arm>.json` |
| `pi/record.py` | Pi | teleop + capture to disk |
| `pi/run_policy.py` | Pi | inference client |
| `desktop/convert.py` | desktop | session dir → LeRobotDataset |
| `desktop/serve.py` | desktop | policy server |
| `tests/latency_probe.py` | either | go/no-go for the dumb-bridge design |

## Before you build (§7 hazards)

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext   # confirm native MJPEG @ 640x480/30
vcgencmd get_throttled                              # 0x0 == healthy power
```

Record to a **USB SSD**, not the boot SD card. Address cameras by
`/dev/v4l/by-id/...`, which is stable across reboots.

## Build / run order (§9)

```bash
# 1. Calibrate each arm (overwrites the placeholder JSONs in calibration/).
python -m pi.calibrate --arm follower --port /dev/ttyACM0
python -m pi.calibrate --arm leader   --port /dev/ttyACM1

# 2. Latency probe BEFORE writing/using inference — answers a design question cheaply.
python tests/latency_probe.py --port /dev/ttyACM0 --ids 1,2,3,4,5,6 --duration 60

# 3. Record teleop episodes.
python -m pi.record --task pick_cube \
    --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
    --cam top=/dev/v4l/by-id/CAM_TOP --cam wrist=/dev/v4l/by-id/CAM_WRIST
#   Per episode, press:  g = keep (good)   d = discard   q = keep + quit

# 4. Convert on the desktop and CHECK alignment before recording 200 more.
python -m desktop.convert --session sessions/2026-07-23_pick_cube \
    --repo-id you/so101_pick_cube --root ./data/so101_pick_cube

# 5. Train with stock LeRobot (desktop).

# 6. Inference (last).
python -m desktop.serve  --checkpoint ./checkpoints/pick_cube --device cuda   # desktop
python -m pi.run_policy  --server DESKTOP_TS_IP:5555 --port /dev/ttyACM0 \
    --cam top=/dev/v4l/by-id/CAM_TOP --cam wrist=/dev/v4l/by-id/CAM_WRIST     # Pi
```

The `calibration/*.json` files committed here are **placeholders** (identity ranges).
Run `pi/calibrate.py` to generate real ones before recording.

## Conventions frozen on day one (§2)

- `action` = **leader** position (commanded), `observation.state` = **follower**
  position (what happened). Swapping them trains without error and yields a useless
  policy — `convert.py` asserts they differ.
- Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- Body joints normalise to ±100, gripper to 0–100.
- Control loop 30 Hz, policy query 10 Hz.
