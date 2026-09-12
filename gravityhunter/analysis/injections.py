from __future__ import annotations

import numpy as np

from gravityhunter.dsp.matched_filter import expected_template_snr


def inject_signal(
    noise: np.ndarray,
    signal: np.ndarray,
    *,
    start_index: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    x[n] = n[n] + alpha*s[n] inserted at start_index.
    """
    noise = np.asarray(noise, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if start_index < 0 or start_index + signal.size > noise.size:
        raise ValueError("Injection does not fit inside noise array.")

    out = noise.copy()
    out[start_index:start_index + signal.size] += alpha * signal
    return out


def scale_template_to_target_snr(
    template: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    target_snr: float,
    *,
    nfft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Scale a template so its expected noise-weighted norm is target_snr.

    Returns (scaled_template, alpha).
    """
    rho0 = expected_template_snr(
        template,
        fs,
        psd_freq,
        psd,
        nfft=nfft,
        fmin=fmin,
        fmax=fmax,
    )

    if rho0 <= 0:
        raise ValueError("Template has zero expected SNR.")

    alpha = target_snr / rho0
    return np.asarray(template, dtype=float) * alpha, float(alpha)
