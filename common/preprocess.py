"""JPEG bytes -> policy input. THE one file that must not drift (§1).

Imported by both desktop/convert.py (dataset writing) and desktop/serve.py (live
inference). The spatial transform must be byte-identical across the two paths or you get
silent train/serve skew — a policy that trains cleanly and behaves badly, with no error
anywhere. Keeping it here, in one file, is the cheapest available insurance.

DESKTOP-ONLY. torch/PIL are imported at module load so a stray Pi-side import fails
loudly and immediately (§1) rather than mid-run.
"""

from __future__ import annotations

try:
    import numpy as np
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "common/preprocess.py is desktop-only (needs PIL/torch). "
        "The Pi must never import it — it decodes pixels, which violates SPEC §0."
    ) from e

import io

# ── the freeze point ─────────────────────────────────────────────────────────
# Resolution and channel/dtype/range the policy consumes. Changing any of these
# invalidates train/serve compatibility for existing checkpoints — bump deliberately.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
RESAMPLE = Image.BILINEAR


def decode_frame(jpeg_bytes: bytes) -> "np.ndarray":
    """JPEG bytes -> resized uint8 HWC RGB array at policy resolution.

    Used by convert.py to fill the dataset and by serve.py before inference, so stored
    frames and live frames go through exactly the same spatial transform.
    """
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), RESAMPLE)
    return np.array(img, dtype=np.uint8)  # copy -> writable (LeRobot/torch prefer writable)


def to_policy_tensor(frame: "np.ndarray"):
    """uint8 HWC RGB [0,255] -> float32 CHW [0,1] torch tensor (serve.py).

    add_frame() in convert.py stores the uint8 HWC array directly; LeRobot applies the
    same CHW/float normalisation at train time, so this defines the serve-side match.
    """
    import torch

    t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).contiguous()
    return t.to(torch.float32) / 255.0
