from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .windowing import make_window


def manual_welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Educational Welch PSD implementation.

    Steps:
        split -> overlap -> window -> rFFT -> |FFT|^2 -> density scaling -> average

    Returns a one-sided PSD with units input_unit^2 / Hz.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    if nperseg <= 1 or nperseg > n:
        raise ValueError("Need 1 < nperseg <= len(x).")

    if noverlap is None:
        noverlap = nperseg // 2

    if not (0 <= noverlap < nperseg):
        raise ValueError("Need 0 <= noverlap < nperseg.")

    hop = nperseg - noverlap
    w = make_window(window, nperseg)
    window_power = np.sum(w**2)

    psds = []
    for start in range(0, n - nperseg + 1, hop):
        seg = x[start:start + nperseg]
        segw = seg * w

        X = np.fft.rfft(segw)
        P = (np.abs(X) ** 2) / (fs * window_power)

        # Convert two-sided power to a one-sided density.
        if nperseg % 2 == 0:
            P[1:-1] *= 2.0
        else:
            P[1:] *= 2.0

        psds.append(P)

    if not psds:
        raise ValueError("No complete Welch segments were produced.")

    psd = np.mean(np.vstack(psds), axis=0)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, psd


def welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    SciPy Welch PSD. This should closely match manual_welch_psd()
    when the same segmentation/scaling choices are used.
    """
    x = np.asarray(x, dtype=np.float64)

    if noverlap is None:
        noverlap = nperseg // 2

    f, pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    return f, pxx


def asd_from_psd(psd: np.ndarray) -> np.ndarray:
    psd = np.asarray(psd, dtype=np.float64)
    return np.sqrt(np.maximum(psd, 0.0))
