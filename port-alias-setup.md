# Port alias setup (one-time, on the Pi)

Without permanent aliases, ports can be reassigned without warning. Fix them permanently with this script; downstream scripts rely on this convention.

| Device | Path |
|---|---|
| Leader arm | `/dev/so101-leader` |
| Follower arm | `/dev/so101-follower` |
| External camera | `/dev/v4l/by-id/cam-ext` |
| Wrist camera | `/dev/v4l/by-id/cam-wrist` |

## Arms

```bash
# connect leader arm to the pi, then identify it
udevadm info -q property -n /dev/ttyACM0
# note ID_VENDOR_ID, ID_MODEL_ID, ID_SERIAL_SHORT

# disconnect, connect follower arm, identify it the same way (serial will differ)
udevadm info -q property -n /dev/ttyACM0
```

```bash
sudo nano /etc/udev/rules.d/99-so101.rules
```

Paste, filling in `<>` from the values above for each arm:

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="<ID_VENDOR_ID>", ATTRS{idProduct}=="<ID_MODEL_ID>", ATTRS{serial}=="<ID_SERIAL_SHORT>", SYMLINK+="so101-leader"
SUBSYSTEM=="tty", ATTRS{idVendor}=="<ID_VENDOR_ID>", ATTRS{idProduct}=="<ID_MODEL_ID>", ATTRS{serial}=="<ID_SERIAL_SHORT>", SYMLINK+="so101-follower"
```

## Cameras

```bash
# connect external cam only, identify it
udevadm info -q property -n /dev/video0 | grep ID_SERIAL_SHORT

# disconnect, connect wrist cam, identify it
udevadm info -q property -n /dev/video0 | grep ID_SERIAL_SHORT
```

```bash
sudo nano /etc/udev/rules.d/99-so101-cameras.rules
```

Paste, filling in `<>` from the values above for each camera:

```
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="<ID_SERIAL_SHORT>", ATTR{index}=="0", SYMLINK+="v4l/by-id/cam-ext"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="<ID_SERIAL_SHORT>", ATTR{index}=="0", SYMLINK+="v4l/by-id/cam-wrist"
```

## Apply and verify

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

# with both arms and both cameras attached:
ls -l /dev/so101-leader /dev/so101-follower /dev/v4l/by-id/cam-ext /dev/v4l/by-id/cam-wrist
# each should resolve to a distinct ttyACM*/video* node
```
