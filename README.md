# so101-bridge

Data capture and remote VLA inference for an SO-101 arm. A Raspberry Pi is the
robot-side machine; a GPU desktop does dataset writing, training, and inference. This repo is a workaround to the Pi having insufficient RAM to load the LeRobot module.

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

## Build / run order

First do the [port alias setup](port-alias-setup.md) (one time) — downstream scripts rely on those aliases.

```bash
# 1. Calibrate each arm (overwrites the placeholder JSONs in calibration/).
python -m pi.calibrate --arm follower
python -m pi.calibrate --arm leader

# 2. Record teleop episodes.
python -m pi.record --task pick_cube
#   Press: s = start or stop recording; k = keep; d = discard; q = quit

# 3. Convert on the desktop and CHECK alignment before recording 200 more.
python -m desktop.convert --session sessions/2026-07-23_pick_cube \
    --repo-id you/so101_pick_cube --root ./data/so101_pick_cube

# 4. Train with stock LeRobot (desktop).

# 5. Inference (last).
python -m desktop.serve  --checkpoint ./checkpoints/pick_cube --device cuda   # desktop
python -m pi.run_policy  --server DESKTOP_TS_IP:5555                         # Pi
```

The `calibration/*.json` files committed here are **placeholders** (identity ranges).
Run `pi/calibrate.py` to generate real ones before recording.

## Conventions frozen

- `action` = **leader** position (commanded), `observation.state` = **follower**
  position (what happened). Swapping them trains without error and yields a useless
  policy — `convert.py` asserts they differ.
- Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- Body joints normalise to ±100, gripper to 0–100.
- Control loop 30 Hz. Observations stream out whenever the action queue drains
  past `--chunk-size-threshold` (async inference).
