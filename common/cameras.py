"""Camera capture via ffmpeg subprocesses — JPEG bytes only, never decoded (§0).

Two sinks, same subprocess and same `-c:v copy` passthrough:

  FileRecorder  — one ffmpeg per camera writing native MJPEG into Matroska (§3.2).
  FrameStream   — one ffmpeg per camera piping concatenated JPEGs to stdout; a reader
                  thread keeps only the newest frame in memory (§5.2).

Pi-side. Stdlib + numpy-free. The Pi never imports opencv/PIL; decoding happens on the
desktop. Address cameras by /dev/v4l/by-id/... — /dev/videoN reorders across reboots (§3.2).
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from common.schema import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH

SOI = b"\xff\xd8"  # JPEG start-of-image
EOI = b"\xff\xd9"  # JPEG end-of-image


# Locked capture controls, applied before every stream start (§3.2).
#
# Every automatic algorithm on these cameras converges over seconds and re-hunts
# mid-episode. ext's continuous autofocus takes ~3.2s to settle — a defocused frame
# compresses to ~23% of its sharp size — and its re-hunts produce the oversized frame
# that precedes every dropped-frame event. wrist's auto-exposure takes 2.4–5.3s. Both
# ship with exposure_dynamic_framerate=1 (non-default), which lets the camera lower its
# frame rate to buy exposure time. Locked, there is nothing to converge: frame 1 matches
# frame 500, and capture matches inference.
#
# Applied one control per call: v4l2-ctl batches a single invocation into one
# VIDIOC_S_EXT_CTRLS, where a dependent control can be evaluated before the automatic
# it depends on is off. Order matters — each `*_automatic` must precede its value.
CONTROLS = {
    "ext": [
        ("exposure_dynamic_framerate", 0),
        ("power_line_frequency", 2),          # 2 = 60 Hz
        ("focus_automatic_continuous", 0),
        ("focus_absolute", 0),                # MEASURE: see README, scene-dependent
        ("white_balance_automatic", 0),
        ("white_balance_temperature", 4000),
        ("auto_exposure", 1),                 # 1 = manual
        ("exposure_time_absolute", 250),
        ("gain", 0),
    ],
    "wrist": [
        ("exposure_dynamic_framerate", 0),
        ("power_line_frequency", 2),
        ("white_balance_automatic", 0),
        ("white_balance_temperature", 4600),
        ("auto_exposure", 1),
        ("exposure_time_absolute", 157),      # MEASURE: see README, scene-dependent
        ("gain", 0),
    ],
}


def apply_controls(name: str, device: str) -> None:
    """Freeze exposure/focus/white balance so nothing converges mid-episode."""
    for control, value in CONTROLS[name]:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", f"{control}={value}"],
            capture_output=True, text=True,
        )
        # An unlocked camera silently produces a different image distribution than the
        # one the policy trained on, so a rejected control is fatal, not a warning.
        if result.returncode != 0:
            raise RuntimeError(
                f"camera '{name}': {control}={value} rejected: {result.stderr.strip()}"
            )


def _base_input_args(device: str) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        "-framerate", str(CAMERA_FPS),
        "-i", device,
    ]


# ── file sink (capture) ──────────────────────────────────────────────────────


class FileRecorder:
    """Records each camera to episode_dir/cam_<name>.mkv, one ffmpeg per camera."""

    def __init__(self, cameras: dict[str, str]):
        """cameras: {name: /dev/v4l/by-id/...}"""
        self.cameras = cameras
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, episode_dir: Path) -> dict[str, str]:
        """Spawn recorders. Returns {name: output_path}."""
        outputs = {}
        for name, device in self.cameras.items():
            apply_controls(name, device)
            out = str(Path(episode_dir) / f"cam_{name}.mkv")
            cmd = _base_input_args(device) + [
                # Unix-epoch PTS to align against the joint log after the fact (§3.5).
                # -copyts is load-bearing: without it the Matroska muxer rebases the
                # first PTS to zero and the absolute origin — the only thing tying the
                # two timelines together — is lost.
                "-use_wallclock_as_timestamps", "1",
                "-copyts",
                "-c:v", "copy",
                "-f", "matroska",
                out,
            ]
            self._procs[name] = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
            outputs[name] = out
        return outputs

    def check_alive(self) -> None:
        """Raise if any recorder has exited. Cheap enough to call every control tick.

        A busy device (e.g. an orphaned recorder from a crashed run) makes ffmpeg exit
        at once; without this the session records joints and no video, silently.
        """
        dead = sorted(n for n, p in self._procs.items() if p.poll() is not None)
        if dead:
            raise RuntimeError(f"camera recorder exited: {', '.join(dead)}")

    def stop(self) -> None:
        """SIGTERM: ffmpeg writes the Matroska trailer on it; kill if it hangs."""
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self._procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._procs.clear()


# ── pipe sink (inference) ────────────────────────────────────────────────────


class FrameStream:
    """Streams one camera's JPEGs; keeps only the latest frame.

    Discard, don't queue (§5.2): a backlog of stale frames silently converts network
    jitter into growing observation lag that looks like a policy problem.
    """

    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self) -> None:
        apply_controls(self.name, self.device)
        cmd = _base_input_args(self.device) + ["-c:v", "copy", "-f", "image2pipe", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        buf = bytearray()
        stdout = self._proc.stdout
        while not self._stop.is_set():
            chunk = stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            # Emit the last complete JPEG in the buffer; drop everything before it.
            end = buf.rfind(EOI)
            if end == -1:
                continue
            start = buf.rfind(SOI, 0, end)
            if start == -1:
                continue
            with self._lock:
                self._latest = bytes(buf[start:end + 2])
            del buf[:end + 2]

    def latest(self) -> bytes | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)


def open_streams(cameras: dict[str, str]) -> dict[str, FrameStream]:
    """Start a FrameStream per camera. {name: device} -> {name: FrameStream}."""
    streams = {name: FrameStream(name, device) for name, device in cameras.items()}
    for s in streams.values():
        s.start()
    return streams
