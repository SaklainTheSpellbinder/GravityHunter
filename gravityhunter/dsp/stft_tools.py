from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .windowing import make_window


def manual_stft(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Educational STFT:
        segment -> window -> rFFT -> shift by hop -> repeat

    Returns
    -------
    f : (F,)
    t : (T,) frame-center times
    Z : (F, T) complex
    """
    x = np.asarray(x, dtype=np.float64)

    if not (0 <= noverlap < nperseg <= x.size):
        raise ValueError("Need 0 <= noverlap < nperseg <= len(x).")

    hop = nperseg - noverlap
    w = make_window(window, nperseg)

    cols = []
    centers = []

    for start in range(0, x.size - nperseg + 1, hop):
        seg = x[start:start + nperseg] * w
        cols.append(np.fft.rfft(seg))
        centers.append((start + (nperseg - 1) / 2.0) / fs)

    Z = np.stack(cols, axis=1)  # (F, T)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    t = np.asarray(centers)
    return f, t, Z


def scipy_stft(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SciPy STFT, no boundary padding so timing stays easy to interpret.
    """
    x = np.asarray(x, dtype=np.float64)
    f, t, Z = stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    return f, t, Z


def spectrogram_power(Z: np.ndarray) -> np.ndarray:
    return np.abs(Z) ** 2


def power_to_db(power: np.ndarray, *, floor_db: float = -120.0) -> np.ndarray:
    power = np.asarray(power, dtype=float)
    ref = np.max(power)
    if ref <= 0:
        return np.full_like(power, floor_db)
    db = 10.0 * np.log10(np.maximum(power / ref, 10 ** (floor_db / 10.0)))
    return db
