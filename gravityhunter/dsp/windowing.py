from __future__ import annotations

import numpy as np


def make_window(name: str, n: int) -> np.ndarray:
    """
    Supported teaching windows: rectangular, hann, hamming.
    """
    name = name.lower()

    if name in {"rect", "rectangular", "boxcar"}:
        return np.ones(n, dtype=float)
    if name in {"hann", "hanning"}:
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)

    raise ValueError(f"Unknown window: {name}")


def apply_window(x: np.ndarray, name: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    w = make_window(name, x.size)
    return x * w, w
