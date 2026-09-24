from __future__ import annotations

import numpy as np
from scipy.signal import get_window, welch


def manual_welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Educational Welch PSD using SciPy-compatible density scaling.

    Each segment is mean-detrended, multiplied by a DFT-even/periodic window,
    converted to a one-sided modified periodogram, then averaged.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    if nperseg <= 1 or nperseg > n:
        raise ValueError("Need 1 < nperseg <= len(x).")
    if noverlap is None:
        noverlap = nperseg // 2
    if not (0 <= noverlap < nperseg):
        raise ValueError("Need 0 <= noverlap < nperseg.")

    hop = nperseg - noverlap
    # SciPy's string windows use the FFT-bin/periodic convention for Welch.
    w = get_window(window, nperseg, fftbins=True)
    window_power = float(np.sum(np.abs(w) ** 2))

    psds: list[np.ndarray] = []
    for start in range(0, n - nperseg + 1, hop):
        seg = x[start:start + nperseg]
        seg = seg - np.mean(seg)  # detrend='constant'
        X = np.fft.rfft(seg * w)
        P = (np.abs(X) ** 2) / (fs * window_power)
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
    average: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD in input-unit²/Hz."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a non-empty 1-D signal")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    nperseg = int(min(max(2, nperseg), x.size))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(max(0, noverlap), nperseg - 1))
    if average not in {"mean", "median"}:
        raise ValueError("average must be 'mean' or 'median'")

    f, pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average=average,
    )
    return f, pxx


def asd_from_psd(psd: np.ndarray) -> np.ndarray:
    psd = np.asarray(psd, dtype=np.float64)
    return np.sqrt(np.maximum(psd, 0.0))
