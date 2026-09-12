from __future__ import annotations

import numpy as np


def interpolate_psd(
    target_freq: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    floor_ratio: float = 1e-12,
) -> np.ndarray:
    """
    Interpolate a one-sided PSD onto another one-sided frequency grid.

    A small positive floor prevents division by zero.
    """
    target_freq = np.asarray(target_freq, dtype=float)
    psd_freq = np.asarray(psd_freq, dtype=float)
    psd = np.asarray(psd, dtype=float)

    if psd_freq.ndim != 1 or psd.ndim != 1 or psd_freq.size != psd.size:
        raise ValueError("psd_freq and psd must be same-length 1-D arrays.")

    interp = np.interp(
        target_freq,
        psd_freq,
        psd,
        left=psd[0],
        right=psd[-1],
    )

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
    """
    Educational PSD-based whitening.

        X_white(f) = X(f) / sqrt(S_n(f))

    Optionally zero frequencies outside [fmin, fmax].

    Returns
    -------
    x_white      : (N,)
    f            : rFFT frequency axis
    X_white      : (N//2+1,) complex for even N
    psd_on_grid  : same shape as X_white

    Notes
    -----
    Absolute whitening normalization depends on Fourier/PSD conventions.
    `standardize=True` rescales the time series to zero mean, unit std,
    while preserving the spectral flattening.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)

    psd_i = interpolate_psd(f, psd_freq, psd)
    Xw = X / np.sqrt(psd_i)

    mask = np.ones_like(f, dtype=bool)
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    Xw = np.where(mask, Xw, 0.0)

    xw = np.fft.irfft(Xw, n=n)

    if standardize:
        xw = xw - np.mean(xw)
        std = np.std(xw)
        if std > 0:
            xw = xw / std

    return xw, f, Xw, psd_i
