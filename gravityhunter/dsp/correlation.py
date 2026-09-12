from __future__ import annotations

import numpy as np


def full_lags(n_data: int, n_template: int) -> np.ndarray:
    """
    Lags corresponding to np.correlate(data, template, mode="full"):
        -(M-1), ..., 0, ..., N-1
    """
    return np.arange(-(n_template - 1), n_data)


def direct_correlation(
    data: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    corr = np.correlate(data, template, mode="full")
    lags = full_lags(data.size, template.size)
    return lags, corr


def fft_linear_correlation(
    data: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full linear correlation via FFT.

    The raw IFFT ordering is:
        lag 0, +1, +2, ..., negative lags at the end

    We reorder it to match np.correlate(..., mode="full").
    """
    data = np.asarray(data, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    n = data.size
    m = template.size
    L = n + m - 1

    X = np.fft.rfft(data, n=L)
    S = np.fft.rfft(template, n=L)
    circular_order = np.fft.irfft(X * np.conj(S), n=L)

    if m > 1:
        corr = np.concatenate((circular_order[-(m - 1):], circular_order[:n]))
    else:
        corr = circular_order[:n]

    lags = full_lags(n, m)
    return lags, corr


def normalized_correlation(
    data: np.ndarray,
    template: np.ndarray,
    *,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Teaching implementation of overlap-normalized correlation.

    O(NM), so use on short arrays or as a correctness oracle.
    """
    data = np.asarray(data, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    n = data.size
    m = template.size
    lags = full_lags(n, m)
    out = np.zeros(lags.size, dtype=float)

    for i, lag in enumerate(lags):
        # data index j corresponds to template index j-lag
        data_start = max(0, lag)
        data_stop = min(n, lag + m)

        if data_stop <= data_start:
            continue

        temp_start = data_start - lag
        temp_stop = temp_start + (data_stop - data_start)

        xd = data[data_start:data_stop]
        st = template[temp_start:temp_stop]

        denom = np.linalg.norm(xd) * np.linalg.norm(st)
        out[i] = 0.0 if denom < eps else float(np.dot(xd, st) / denom)

    return lags, out
