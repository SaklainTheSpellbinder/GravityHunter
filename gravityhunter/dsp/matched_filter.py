from __future__ import annotations

import numpy as np

from .correlation import full_lags
from .whitening import interpolate_psd


def matched_filter_snr(
    data: np.ndarray,
    template: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-sided PSD-weighted matched filter.

    Continuous idea:
        z(t) ~ 4 Re ∫ X(f) S*(f) / S_n(f) exp(j 2πft) df

    This function computes a full linear lag series by zero-padding to
        L = N + M - 1.

    Returns
    -------
    lags : (L,)
        Lags in samples, matching np.correlate convention.
    snr : (L,)
        Noise-weighted normalized matched-filter time series.
    raw : (L,)
        Unnormalized PSD-weighted correlation.

    Notes
    -----
    Exact absolute SNR conventions vary with PSD/FFT normalization.
    This implementation is internally consistent and suitable for the
    educational GravityHunter pipeline. Validate it on synthetic injections.
    """
    data = np.asarray(data, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)

    n = data.size
    m = template.size
    L = n + m - 1

    f = np.fft.rfftfreq(L, d=1.0 / fs)
    X = np.fft.rfft(data, n=L)
    S = np.fft.rfft(template, n=L)

    Sn = interpolate_psd(f, psd_freq, psd)

    mask = np.ones_like(f, dtype=bool)
    mask[0] = False  # ignore DC in the detector statistic
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax

    q = np.zeros_like(X)
    q[mask] = X[mask] * np.conj(S[mask]) / Sn[mask]

    df = fs / L
    dt = 1.0 / fs

    # np.fft.rfft is an unnormalized discrete sum. The continuous Fourier
    # transform approximation is dt * FFT(x). Including dt^2 here keeps the
    # PSD-density units consistent with scipy.signal.welch.
    #
    # irfft contributes roughly (2/L) Re sum over positive bins, so
    # 4*df*dt^2*Re(sum) = irfft(q) * (2*L*df*dt^2) = irfft(q)*(2/fs).
    circular_raw = np.fft.irfft(q, n=L) * (2.0 / fs)

    if m > 1:
        raw = np.concatenate(
            (circular_raw[-(m - 1):], circular_raw[:n])
        )
    else:
        raw = circular_raw[:n]

    sigma2 = 4.0 * df * (dt ** 2) * np.sum(
        (np.abs(S[mask]) ** 2) / Sn[mask]
    )
    sigma = np.sqrt(max(float(sigma2), np.finfo(float).tiny))

    snr = raw / sigma
    lags = full_lags(n, m)
    return lags, snr, raw


def expected_template_snr(
    template: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    nfft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """
    Noise-weighted norm of a template:
        sigma^2 = 4 ∫ |S(f)|^2 / S_n(f) df
    """
    template = np.asarray(template, dtype=np.float64)

    if nfft is None:
        nfft = template.size
    if nfft < template.size:
        raise ValueError("nfft must be >= template length.")

    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    S = np.fft.rfft(template, n=nfft)
    Sn = interpolate_psd(f, psd_freq, psd)

    mask = np.ones_like(f, dtype=bool)
    mask[0] = False
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax

    df = fs / nfft
    dt = 1.0 / fs
    sigma2 = 4.0 * df * (dt ** 2) * np.sum(
        (np.abs(S[mask]) ** 2) / Sn[mask]
    )
    return float(np.sqrt(max(sigma2, 0.0)))


def noise_weighted_template_correlation(
    template_a: np.ndarray,
    template_b: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    *,
    nfft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """Return the normalized PSD-weighted overlap between two real templates.

    This is the correlation coefficient of the two template basis functions under
    the same noise-weighted inner product used by the matched filter.  It is used
    when combining two non-orthogonal quadratures; assuming orthogonality when
    they are correlated can overstate the combined statistic.
    """
    a = np.asarray(template_a, dtype=np.float64)
    b = np.asarray(template_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size < 2:
        raise ValueError("Templates must be same-length non-empty 1-D arrays")
    if nfft is None:
        nfft = a.size
    if nfft < a.size:
        raise ValueError("nfft must be >= template length")

    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    Sn = interpolate_psd(f, psd_freq, psd)
    mask = np.ones_like(f, dtype=bool)
    mask[0] = False
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    if not np.any(mask):
        raise ValueError("No frequency bins remain in the requested analysis band")

    df = fs / nfft
    dt = 1.0 / fs
    scale = 4.0 * df * (dt ** 2)
    aa = scale * np.sum((np.abs(A[mask]) ** 2) / Sn[mask])
    bb = scale * np.sum((np.abs(B[mask]) ** 2) / Sn[mask])
    ab = scale * np.sum(np.real(A[mask] * np.conj(B[mask])) / Sn[mask])
    denom = np.sqrt(max(float(aa), 0.0) * max(float(bb), 0.0))
    if denom <= np.finfo(float).tiny:
        return 0.0
    return float(np.clip(ab / denom, -1.0, 1.0))


def combine_two_basis_snr(
    snr_a: np.ndarray,
    snr_b: np.ndarray,
    correlation: float,
    *,
    singular_tol: float = 1e-6,
) -> np.ndarray:
    """Maximize a two-template matched-filter statistic over linear amplitudes.

    If the two normalized template basis functions have overlap ``c``, then

        rho^2 = (rho_a^2 - 2 c rho_a rho_b + rho_b^2) / (1-c^2).

    This reduces to ``hypot(rho_a, rho_b)`` for orthogonal quadratures.
    """
    a = np.asarray(snr_a, dtype=float)
    b = np.asarray(snr_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("SNR arrays must have the same shape")
    c = float(np.clip(correlation, -1.0, 1.0))
    denom = 1.0 - c * c
    if denom <= singular_tol:
        # Nearly redundant basis: keep the stronger one rather than numerically
        # amplifying a nearly singular Gram matrix.
        return np.maximum(np.abs(a), np.abs(b))
    rho2 = (a * a - 2.0 * c * a * b + b * b) / denom
    return np.sqrt(np.maximum(rho2, 0.0))
