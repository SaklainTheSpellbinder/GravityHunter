from __future__ import annotations

import numpy as np


def interpolate_psd(
    target_freq: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    floor_ratio: float = 1e-12,
) -> np.ndarray:
    """Interpolate a one-sided PSD onto a target frequency grid."""
    target_freq = np.asarray(target_freq, dtype=float)
    psd_freq = np.asarray(psd_freq, dtype=float)
    psd = np.asarray(psd, dtype=float)

    if psd_freq.ndim != 1 or psd.ndim != 1 or psd_freq.size != psd.size:
        raise ValueError("psd_freq and psd must be same-length 1-D arrays.")
    if psd_freq.size < 2 or np.any(np.diff(psd_freq) <= 0):
        raise ValueError("psd_freq must be strictly increasing")
    if not np.all(np.isfinite(psd_freq)) or not np.all(np.isfinite(psd)):
        raise ValueError("PSD arrays contain NaN/Inf")

    interp = np.interp(target_freq, psd_freq, psd, left=psd[0], right=psd[-1])
    positive = interp[interp > 0]
    if positive.size == 0:
        raise ValueError("PSD must contain positive values.")
    floor = max(np.median(positive) * floor_ratio, np.finfo(float).tiny)
    return np.maximum(interp, floor)


def whiten(
    x: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Whiten a real signal using a one-sided PSD density.

    For NumPy's unnormalized rFFT and a one-sided PSD S_n(f), the discrete
    normalization follows the standard GWOSC tutorial convention:

        X_white[k] = X[k] / sqrt(S_n(f_k) * fs / 2)

    (equivalently divide by sqrt(S_n / dt / 2)).  Frequencies outside the
    requested analysis band are set to zero.  `standardize=True` applies only
    a final scalar centering/rescaling and therefore does not change the
    spectral whitening shape.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a non-empty 1-D signal")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")

    n = x.size
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x - np.mean(x))
    psd_i = interpolate_psd(f, psd_freq, psd)

    # GWOSC normalization for a one-sided PSD density.
    denom = np.sqrt(psd_i * fs / 2.0)
    Xw = X / denom

    mask = np.ones_like(f, dtype=bool)
    mask[0] = False
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    Xw = np.where(mask, Xw, 0.0)

    xw = np.fft.irfft(Xw, n=n)
    if standardize:
        xw = xw - np.mean(xw)
        std = float(np.std(xw))
        if std > 0 and np.isfinite(std):
            xw = xw / std

    return xw, f, Xw, psd_i
