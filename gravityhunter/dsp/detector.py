from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class DetectionResult:
    detected: bool
    peak_snr: float
    peak_index: int | None
    peak_lag_samples: int | None
    peak_time_s: float | None
    threshold: float


def find_snr_peaks(
    snr: np.ndarray,
    *,
    threshold: float,
    min_distance_samples: int | None = None,
    prominence: float | None = None,
) -> np.ndarray:
    snr = np.asarray(snr, dtype=float)

    kwargs = {"height": threshold}
    if min_distance_samples is not None:
        kwargs["distance"] = min_distance_samples
    if prominence is not None:
        kwargs["prominence"] = prominence

    peaks, _ = find_peaks(snr, **kwargs)
    return peaks


def detect_best_candidate(
    snr: np.ndarray,
    lags: np.ndarray,
    fs: float,
    *,
    threshold: float,
    template_reference_index: int = 0,
    min_distance_samples: int | None = None,
    prominence: float | None = None,
) -> DetectionResult:
    """
    Convert an SNR-vs-lag series into one best educational candidate.

    Candidate time is interpreted as:
        (lag + template_reference_index) / fs

    If your template reference time is its merger sample, pass that index.
    """
    snr = np.asarray(snr, dtype=float)
    lags = np.asarray(lags, dtype=int)

    if snr.shape != lags.shape:
        raise ValueError("snr and lags must have the same shape.")

    peaks = find_snr_peaks(
        snr,
        threshold=threshold,
        min_distance_samples=min_distance_samples,
        prominence=prominence,
    )

    if peaks.size == 0:
        return DetectionResult(
            detected=False,
            peak_snr=float(np.max(snr)) if snr.size else float("-inf"),
            peak_index=None,
            peak_lag_samples=None,
            peak_time_s=None,
            threshold=threshold,
        )

    best_idx = int(peaks[np.argmax(snr[peaks])])
    lag = int(lags[best_idx])
    candidate_sample = lag + int(template_reference_index)

    return DetectionResult(
        detected=True,
        peak_snr=float(snr[best_idx]),
        peak_index=best_idx,
        peak_lag_samples=lag,
        peak_time_s=candidate_sample / fs,
        threshold=threshold,
    )
