# so101-bridge — Software Design Spec

Data collection and remote VLA inference for an SO-101 arm. Robot-side host is a Raspberry Pi 5 (1GB); training and inference run on a GPU desktop.

NOTE: This was the original design doc, retained for reference. Expect some divergence from the current implementations.

---

## 0. Governing constraint

**The Pi never decodes an image and never imports torch.**

JPEG bytes leave the camera and go straight to disk (capture) or straight onto the wire (inference). Everything that needs to understand pixels or tensors runs on the desktop.

Corollary: no `opencv`, no `lerobot`, no `torch`, no `torchvision` in the `[pi]` dependency set. If a dependency pulls any of them transitively, it doesn't go on the Pi.

This single rule determines most of what follows, and it's what lets one repo serve both capture and inference.

### Division of labour

| | Pi | Desktop |
|---|---|---|
| Teleop loop | ✅ | |
| Camera capture | ✅ (passthrough, no decode) | |
| Dataset writing | | ✅ |
| Video re-encode | | ✅ |
| Training | | ✅ |
| Policy inference | | ✅ |
| Action execution | ✅ | |

---

## 1. Repo layout

```
so101-bridge/
  pyproject.toml
  common/
    __init__.py
    servo.py           # bus init, calibration apply, read/write joints
    cameras.py         # ffmpeg subprocess mgmt (file sink + pipe sink)
    schema.py          # episode dirs, jsonl format, joint names/order, rates
    calibration.py     # homing offsets, range normalisation (vendored)
    preprocess.py      # jpeg bytes -> policy input  [desktop-only at runtime]
  pi/
    calibrate.py       # one-off, writes calibration/*.json
    record.py          # teleop + capture to disk
    run_policy.py      # inference client
  desktop/
    convert.py         # session dir -> LeRobotDataset
    serve.py           # policy server
  calibration/
    so101_follower.json
    so101_leader.json
```

### Dependency extras

```toml
[project.optional-dependencies]
pi      = ["pyserial", "feetech-servo-sdk", "pyzmq", "numpy"]
desktop = ["lerobot[feetech]==<PINNED>", "torch", "av", "pyzmq", "numpy"]
```

`pip install -e .[pi]` on the Pi, `pip install -e .[desktop]` on the desktop. Pin LeRobot to an exact version and upgrade deliberately.

### `common/preprocess.py`

Defines the transform from raw JPEG bytes to whatever the policy consumes — resize, crop, channel order, dtype, value range. Imported by both `desktop/convert.py` and `desktop/serve.py`, and it must be byte-identical across the two. Divergence gives you silent train/serve skew: a policy that trains cleanly, behaves badly, and reports no error anywhere. One file is the cheapest insurance against that.

It runs on the desktop in both cases. The Pi imports `common/` for `servo.py`, `cameras.py`, `schema.py`, and `calibration.py` only. Put the torch/PIL imports at module scope in `preprocess.py` so a stray Pi-side import fails at import time rather than mid-run.

---

## 2. Conventions

Borrowed from LeRobot deliberately: matching them means you can fine-tune from community SO-101 checkpoints, compare against public datasets, and hand the data to someone else without explanation. They've also been considerably more stable than LeRobot's APIs.

**Feature keys**
- `action` — **leader** arm position (commanded)
- `observation.state` — **follower** arm position (actual)
- `observation.images.<name>` — e.g. `observation.images.top`, `observation.images.wrist`

Swapping `action` and `observation.state` produces a dataset that trains without error and a policy that does nothing useful. Assert against it in `convert.py`.

**Joint order** — fixed, defined once in `schema.py`:
```python
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
```

**Calibration** — homing offsets plus range normalisation; joints to ±100, gripper 0–100. Stored per-arm in `calibration/*.json`.

**Rates** — control loop 30 Hz, policy query 10 Hz. Constants in `schema.py`.

---

## 3. Capture — `pi/record.py`

### 3.1 Teleop loop

`feetech-servo-sdk` is pure Python. Per tick: sync-read leader, write to follower, append both to the joint log. Target 30 Hz; the capture format tolerates jitter (§3.4). Roughly 80 lines including CLI and episode management.

### 3.2 Cameras — file sink

One ffmpeg subprocess per camera, no decode:

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 \
       -use_wallclock_as_timestamps 1 -i /dev/video0 \
       -c:v copy -f matroska cam_top.mkv
```

- `-c:v copy` — JPEGs pass through untouched; negligible CPU, ~30MB RSS per process
- Matroska — handles variable frame timing honestly, unlike mp4
- `-use_wallclock_as_timestamps 1` — PTS you can align against the joint log

Address cameras by `/dev/v4l/by-id/...`; `/dev/videoN` reorders across reboots.

### 3.3 Episode layout

```
sessions/<session_name>/
  session.json                 # robot id, calibration hash, camera config, git sha
  episode_000/
    meta.json                  # t0_monotonic, t0_unix, fps, task, status
    joints.jsonl               # one object per tick
    cam_top.mkv
    cam_wrist.mkv
```

`joints.jsonl`:
```json
{"t": 0.0333, "state": [1.2, -30.4, ...], "action": [1.3, -30.1, ...]}
```

JSONL is crash-safe on append and trivially inspectable; ~5MB per half hour at 30 Hz, irrelevant next to video.

`session.json` recording the git sha and calibration hash costs nothing and answers "why does this session look different" six months later.

**Episode status.** Mark `status: "good" | "discard"` in `meta.json` at episode end via operator keypress. `convert.py` reads only `good`. Cheaper and less regret-prone than deleting directories.

### 3.4 Timestamps

Take **one** `time.monotonic()` at episode start, in **one** process, and record everything as offsets from it. Don't try to align wall-clock across processes after the fact.

ffmpeg is a separate process, so its PTS are wall-clock; record `t0_unix` alongside `t0_monotonic` to bridge the two. Verify the offset empirically once — wave a hand in front of the camera while jogging a joint — rather than trusting it.

---

## 4. Conversion — `desktop/convert.py`

**Don't hand-write parquet.** Go through `LeRobotDataset.create()` / `add_frame()` / `save_episode()`. The on-disk format has churned repeatedly (v2.0 → v2.1 → v3.0; parquet chunking and video layout both changed) and hand-rolled writers rot. Through the API, a format bump is a version bump and maybe a signature fix.

This is the highest-leverage decision in the design: it's the boundary that absorbs upstream churn.

### The one piece of real logic: resampling

LeRobot wants fixed-fps aligned frames; your capture is variable-rate on both paths. Per episode:

1. Build the control-tick timeline from `joints.jsonl`
2. Decode each camera stream with **PyAV**, walking frames in order
3. For each control tick, select the nearest camera frame by timestamp; drop the rest
4. Pass numpy arrays to `add_frame()`; let LeRobot re-encode video
5. Log max and median alignment error per episode

Max alignment error above half a frame interval (~16ms at 30fps) means a capture bug worth finding before recording 200 more episodes. Print it; don't bury it in a log file.

### Assertions

- joint count matches `JOINTS`
- `action` and `observation.state` are not identical (catches the swap, and catches a leader that wasn't actually being read)
- no NaNs, no values outside the calibrated range
- frame count matches tick count

---

## 5. Inference

### 5.1 Marginal code

Given a working `record.py`:

| Component | Status |
|---|---|
| `common/servo.py` | unchanged |
| `common/preprocess.py` | unchanged |
| `common/cameras.py` | **+1 mode** (pipe sink) — the only new logic |
| `pi/run_policy.py` | ~70 lines |
| `desktop/serve.py` | ~60 lines |

### 5.2 Cameras — pipe sink

Same subprocess, different sink. At inference you want the newest frame in memory, not a file:

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 \
       -i /dev/video0 -c:v copy -f image2pipe -
```

Concatenated JPEGs on stdout. A reader thread splits on `FFD8` / `FFD9` markers and keeps only the latest — ~30 lines, still no decode.

**Discard, don't queue.** A backlog of stale frames silently converts network jitter into a growing observation lag that presents as a policy problem.

### 5.3 Protocol

Pi → desktop: `(jpeg_bytes[], joint_state, seq, t_monotonic)`
Desktop → Pi: action chunk, `seq` echoed

ZMQ REQ/REP with `RCVTIMEO` set. This is lockstep request/response; DEALER/ROUTER buys nothing until there are multiple robots.

LeRobot ships an async inference stack (policy server + robot client, gRPC). The server half is the valuable one — chunk management and aggregation — and it runs on the desktop where torch is fine. If you want it, generate gRPC stubs from their `.proto` files (`grpcio-tools`, generate on the desktop, ship `_pb2.py` to the Pi) rather than forking the package to de-torch the client.

Recommendation: start with your own ZMQ REQ/REP. You'll own the timeout semantics completely, which is the part that matters most here, and their async stack is the newest and least settled part of the codebase.

### 5.4 Control loop

```python
while True:
    frames = {n: c.latest() for n, c in cams.items()}
    state  = bus.read_positions()
    chunk  = client.query(frames, state)        # blocks, RCVTIMEO set
    for action in chunk[:REQUERY_AT]:
        bus.write_positions(action)
        rate.sleep()
```

Query at ~10 Hz, receive 20–50 actions, execute at 30 Hz, re-query before the chunk drains. This amortises the round trip over half a second of motion, which is why network latency mostly stops mattering.

**The irreducible custom code** is this timing loop and the safety behaviour around it — ~30 of the 70 lines. No upstream package hands you this in a form you'd trust:

- **Query timeout** — hold position, or ramp to zero velocity. Never continue executing a stale chunk indefinitely.
- **Servo read failure** — retry once, then halt.
- **SIGINT / SIGTERM** — disable torque cleanly. Wire this before the first real run.
- **Chunk boundary discontinuity** — log it. Large jumps between the end of one chunk and the start of the next indicate the policy is acting on a stale observation.

---

## 6. What to reuse from LeRobot

Everything on the desktop is a real dependency: dataset, training, policies, visualisation. On the Pi nothing is importable, but two things are worth **vendoring** — copying the logic into `common/`, not importing it. Both are small, self-contained, and immune to upstream change once copied.

Paths below are for the pinned version. The tree was flattened at one point (`lerobot.common.*` → `lerobot.*`), so grep rather than trusting them verbatim.

| What | Where in LeRobot | How to use it |
|---|---|---|
| Dataset writing API | `lerobot/datasets/lerobot_dataset.py` — `LeRobotDataset.create` / `add_frame` / `save_episode` | **Import.** The churn boundary (§4). |
| Feature/dtype spec for `create()` | `lerobot/datasets/utils.py` | Import; read it to get the `features` dict shape right. |
| Feetech register map, sync-read/write patterns | `lerobot/motors/feetech/` | **Vendor** into `common/servo.py`. |
| Calibration arithmetic (homing offsets, range norm) | `lerobot/motors/motors_bus.py`, `lerobot/robots/so101_follower/` | **Vendor** into `common/calibration.py`. Read the `calibrate()` path. |
| Joint names, bus wiring, gripper handling | `lerobot/robots/so101_follower/`, `lerobot/teleoperators/so101_leader/` | Reference for `schema.py`. |
| Teleop/record loop structure | the record and teleoperate entry points under `lerobot/scripts/` | Reference only; yours is much smaller without dataset writing inline. |
| Policy load + `select_action` | `lerobot/policies/factory.py` | Import in `serve.py`. |
| Async inference protocol | the async/server subpackage — `.proto` files | Optional (§5.3). Generate stubs; don't fork. |
| Dataset visualisation | `lerobot/scripts/visualize_dataset*.py` | Run as-is to check conversion output. |

---

## 7. Passthrough probe — separate, later

There's a version of this system with no custom Pi code at all: `socat` bridges the serial bus, cameras stream MJPEG, and stock LeRobot robot classes run on the desktop against what looks like local hardware.

The concern is that this puts the network *inside* the servo loop — two round trips per 33ms tick, against SDK timeouts that assume local latency. Whether that holds is an empirical question about the real lab-to-desktop link, so it isn't answerable from the same building and isn't answerable yet.

Treat it as an experiment outside this repo. `latency_probe.py` (provided separately; depends only on `feetech-servo-sdk`) measures single-transaction sync-read latency. Run it once the robot is in the lab, both locally on the Pi as a baseline and on the desktop through the bridge. Rough reading: p99 under ~8ms with no dropouts over a long run means passthrough is worth pursuing; above ~12ms, or any dropout, means it isn't.

**Build the full spec above regardless.** If the probe later comes back clean, you delete `pi/run_policy.py` and the pipe-sink mode in `common/cameras.py` — a cheap deletion. The reverse, discovering mid-project that passthrough doesn't hold and needing to build the client then, is not cheap. Nothing else in the repo depends on the outcome either way.

---

## 8. Build order

1. `common/servo.py`, `common/calibration.py`, `pi/calibrate.py` — arms moving, calibration written and verified against LeRobot's convention.
2. `common/schema.py`, `common/cameras.py` (file sink), `pi/record.py` — record 5 throwaway episodes.
3. `desktop/convert.py` — convert those 5 and **visualise them**. Confirm alignment error is sane and `action`/`observation.state` aren't swapped, before recording at volume.
4. Record a real dataset. Train.
5. `common/preprocess.py` freezes the moment step 4 begins — any change after that invalidates the trained policy.
6. `desktop/serve.py`, `common/cameras.py` (pipe sink), `pi/run_policy.py`.

Steps 2–3 are the load-bearing pair. Get them right and step 6 is a socket, a frame buffer, and a timing loop.