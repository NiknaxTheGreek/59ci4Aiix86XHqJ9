from __future__ import annotations

import numpy as np

KR = 0.2126
KG = 0.7152
KB = 0.0722


def rec709_luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return KR * r + KG * g + KB * b


def rec709_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8-like input to Rec.709 grayscale, repeated to 3 channels."""
    y = rec709_luma(rgb)
    q = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return np.repeat(q[..., None], 3, axis=-1)
