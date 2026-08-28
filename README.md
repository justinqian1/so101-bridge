# so101-bridge

This repo provides utils for data capture and VLA inference for an SO-101 arm (async inference and RTC on by default).
This repo is a workaround for Raspberry Pi's not having enough RAM to load the LeRobot module.
Workflow: Pi for data capture and port forwarding; GPU desktop for dataset writing, training, and inference.

## Install

```bash
pip install -e .[pi]        # on the Raspberry Pi
pip install -e .[desktop]   # on the GPU desktop
```

## Pi-side setup

**One time:** do the [port alias setup](port-alias-setup.md) to set aliases for the arms and cameras; 
downstream scripts rely on these aliases. Then calibrate the arms:

```bash
python -m pi.calibrate --arm follower
python -m pi.calibrate --arm leader
```

**Every time:** connect the cameras to the blue (USB 3.0) ports, and the arms to the gray (USB 2.0) ports.
Before recording, verify all externals are connected:
```bash
ls -l /dev/so101-leader /dev/so101-follower /dev/v4l/by-id/cam-ext /dev/v4l/by-id/cam-wrist
# each should resolve to a distinct ttyACM*/video* node
```

## Usage

```bash
# 1. Record teleop episodes, Pi-only; task(s) are set inside this script.
python -m pi.record --name pick_cube
#   Press: s = start/stop recording; t = change task; k = keep; d = discard; q = quit

# 2. Convert to LeRobotDataset on the desktop.
python -m desktop.convert --session sessions/2026-07-23_pick_cube --repo-id your_name/so101_pick_cube

# 3. Train VLA.
lerobot-train --policy.path=lerobot/smolvla_base \
	--dataset.repo_id=your_name/dataset_name \
	--dataset.image_transforms.enable=true \
	--dataset.image_transforms.max_num_transforms=3 \
	--batch_size=64 --steps=20000 \
	--output_dir=/path/to/output \
	--job_name=job_name \
	--policy.device=cuda \
	--policy.push_to_hub=false \
	--wandb.enable=true \
	--save_freq=2500 \
	--rename_map='{"observation.images.ext": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# 4. Inference.
python -m desktop.serve  --checkpoint ./checkpoints/pick_cube --device cuda   # desktop
python -m pi.run_policy  --server DESKTOP_TS_IP:5555 --dry-run   # dry run: you teleop,
                                                                 # and chunks are printed, not executed
python -m pi.run_policy  --server DESKTOP_TS_IP:5555             # for real; the policy drives the arm
```
