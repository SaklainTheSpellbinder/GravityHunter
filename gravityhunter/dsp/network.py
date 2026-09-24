from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlation import fft_linear_correlation


@dataclass
class DelayEstimate:
    best_lag_samples: int
    delay_seconds: float
    correlation_peak: float
    lags: np.ndarray
    correlation: np.ndarray


@dataclass
class CoincidenceResult:
    h1_detected: bool
    l1_detected: bool
    time_compatible: bool
    coincident: bool
    absolute_time_difference_s: float | None


def estimate_delay(
    h1: np.ndarray,
    l1: np.ndarray,
    fs: float,
    *,
    max_abs_delay_s: float | None = None,
    use_absolute_peak: bool = True,
) -> DelayEstimate:
    """
    Estimate the lag that best aligns H1 with L1 using linear cross-correlation.

    Sign convention for estimate_delay(H1, L1):
      delay > 0  -> H1 is delayed relative to L1 (L1 arrived first)
      delay < 0  -> H1 leads L1 (H1 arrived first)

    This convention is regression-tested with known synthetic shifts and the
    bundled GWOSC event records.
    """
    lags, corr = fft_linear_correlation(h1, l1)

    if max_abs_delay_s is not None:
        max_lag = int(round(max_abs_delay_s * fs))
        valid = np.abs(lags) <= max_lag
        if not np.any(valid):
            raise ValueError("No lags fall inside max_abs_delay_s.")
        lags_eval = lags[valid]
        corr_eval = corr[valid]
    else:
        lags_eval = lags
        corr_eval = corr

    score = np.abs(corr_eval) if use_absolute_peak else corr_eval
    i = int(np.argmax(score))
    lag = int(lags_eval[i])

    return DelayEstimate(
        best_lag_samples=lag,
        delay_seconds=lag / fs,
        correlation_peak=float(corr_eval[i]),
        lags=lags,
        correlation=corr,
    )


def coincidence_check(
    *,
    h1_detected: bool,
    l1_detected: bool,
    h1_time_s: float | None,
    l1_time_s: float | None,
    max_time_difference_s: float,
) -> CoincidenceResult:
    if (
        not h1_detected
        or not l1_detected
        or h1_time_s is None
        or l1_time_s is None
    ):
        return CoincidenceResult(
            h1_detected=h1_detected,
            l1_detected=l1_detected,
            time_compatible=False,
            coincident=False,
            absolute_time_difference_s=None,
        )

    dt = abs(h1_time_s - l1_time_s)
    compatible = dt <= max_time_difference_s

    return CoincidenceResult(
        h1_detected=h1_detected,
        l1_detected=l1_detected,
        time_compatible=compatible,
        coincident=bool(compatible),
        absolute_time_difference_s=float(dt),
    )
