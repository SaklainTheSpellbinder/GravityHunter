from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import tukey

from gravityhunter.catalog import EventSpec, get_event_spec
from gravityhunter.dsp.detector import DetectionResult, detect_best_candidate
from gravityhunter.dsp.filters import apply_bandpass_zero_phase, apply_multiple_notches
from gravityhunter.dsp.loader import StrainRecord, load_common_gwosc_hdf5, slice_by_time
from gravityhunter.dsp.matched_filter import (
    combine_two_basis_snr,
    matched_filter_snr,
    noise_weighted_template_correlation,
)
from gravityhunter.dsp.network import CoincidenceResult, DelayEstimate, coincidence_check, estimate_delay
from gravityhunter.dsp.psd import welch_psd
from gravityhunter.dsp.stft_tools import scipy_stft, spectrogram_power
from gravityhunter.dsp.templates import TemplateRecord, load_losc_template
from gravityhunter.dsp.whitening import whiten


@dataclass
class DetectorResult:
    record: StrainRecord
    raw: np.ndarray
    filtered: np.ndarray
    white: np.ndarray
    psd_freq: np.ndarray
    psd: np.ndarray
    stft_freq: np.ndarray
    stft_time: np.ndarray
    stft_power: np.ndarray
    lags: np.ndarray
    snr: np.ndarray
    detection: DetectionResult


@dataclass
class RealCaseResult:
    spec: EventSpec
    template: TemplateRecord
    template_plus_compact: np.ndarray
    template_cross_compact: np.ndarray
    template_reference_index: int
    h1: DetectorResult
    l1: DetectorResult
    event_time_s: float | None
    network_delay: DelayEstimate | None
    coincidence: CoincidenceResult
    phase_intervals_absolute: dict[str, tuple[float, float]]


def chirp_mass(m1: float, m2: float) -> float:
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def _with_metadata_fallback(record: StrainRecord, spec: EventSpec, detector: str) -> StrainRecord:
    """Fill metadata missing from a local HDF5 variant without changing samples."""
    if record.start_time is not None and record.detector is not None:
        return record
    return StrainRecord(
        strain=record.strain,
        fs=record.fs,
        start_time=record.start_time if record.start_time is not None else spec.record_gps_start,
        detector=record.detector if record.detector is not None else detector,
    )


def _event_time_in_record(record: StrainRecord, spec: EventSpec) -> float | None:
    if spec.gps_event is None:
        return None
    if record.start_time is not None:
        t = float(spec.gps_event - record.start_time)
        if -1e-6 <= t <= record.duration + 1e-6:
            return float(np.clip(t, 0.0, record.duration))
    return None


def _prepare_record(record: StrainRecord, spec: EventSpec) -> tuple[StrainRecord, float | None]:
    if spec.kind == "quiet":
        if spec.quiet_window_s is None:
            raise ValueError("Quiet case is missing quiet_window_s")
        return slice_by_time(record, *spec.quiet_window_s), None
    return record, _event_time_in_record(record, spec)


def _noise_reference(x: np.ndarray, fs: float, event_t: float | None) -> np.ndarray:
    """Return off-source samples for PSD estimation.

    The event neighborhood is excluded when a reference event time is known.
    This is used only for the noise model, never to limit the record-wide candidate
    search itself.
    """
    n = x.size
    if event_t is None:
        return x
    guard = 3.0
    i0 = max(0, int(np.floor((event_t - guard) * fs)))
    i1 = min(n, int(np.ceil((event_t + guard) * fs)))
    parts: list[np.ndarray] = []
    if i0 >= int(2 * fs):
        parts.append(x[:i0])
    if n - i1 >= int(2 * fs):
        parts.append(x[i1:])
    return np.concatenate(parts) if parts else x


def _psd_for_full_record(record: StrainRecord, spec: EventSpec) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a robust PSD from the full local detector file.

    Quiet negative-control windows deliberately do not estimate their PSD from
    the same 8 s test slice.  Instead we use the parent 32 s record and exclude
    the known source event when that parent record contains one.
    """
    event_t = None
    if spec.kind == "real_event":
        event_t = _event_time_in_record(record, spec)
    elif spec.kind == "quiet":
        try:
            source_spec = get_event_spec(spec.source_event)
            event_t = _event_time_in_record(record, source_spec)
        except Exception:
            event_t = None

    noise = _noise_reference(record.strain, record.fs, event_t)
    if noise.size < 128:
        raise ValueError("Not enough off-source samples for a PSD estimate")

    target = min(4096, noise.size)
    nperseg = 2 ** int(np.floor(np.log2(max(128, target))))
    nperseg = min(nperseg, noise.size)
    if nperseg < 128:
        nperseg = noise.size
    return welch_psd(
        noise,
        record.fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        average="median",
    )


def _filter_signal(x: np.ndarray, fs: float, spec: EventSpec) -> np.ndarray:
    valid_notches = [f for f in spec.notches_hz if 0 < f < fs / 2]
    y = apply_multiple_notches(x, fs, valid_notches, q=30.0) if valid_notches else np.asarray(x, dtype=float).copy()
    low, high = spec.fband
    high = min(high, fs / 2 - 1.0)
    y, _ = apply_bandpass_zero_phase(y, fs, low, high, order=4)
    return y


def _analysis_taper(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return x.copy()
    return (x - np.mean(x)) * tukey(x.size, alpha=alpha)


def _matched_quadrature(
    data: np.ndarray,
    tplus: np.ndarray,
    tcross: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    fband: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """PSD-weighted phase-maximized template score.

    Matched filtering is performed on tapered *unfiltered* strain/template data
    while the frequency-domain mask and measured PSD provide the weighting.
    This avoids double-weighting a band-pass/notch transfer function in both the
    numerator and an unfiltered noise PSD.
    """
    kwargs = dict(fmin=fband[0], fmax=min(fband[1], fs / 2 - 1.0))
    d = _analysis_taper(data, alpha=0.04)
    tp = _analysis_taper(tplus, alpha=0.08)
    tc = _analysis_taper(tcross, alpha=0.08)
    lags_p, snr_p, _ = matched_filter_snr(d, tp, fs, psd_freq, psd, **kwargs)
    lags_c, snr_c, _ = matched_filter_snr(d, tc, fs, psd_freq, psd, **kwargs)
    if not np.array_equal(lags_p, lags_c):
        raise RuntimeError("Quadrature matched-filter lag axes differ")

    # The public plus/cross basis is not assumed to be exactly orthogonal under
    # the measured detector PSD.  Maximize over the two linear response
    # amplitudes using their noise-weighted Gram matrix instead of blindly
    # taking hypot(), which is only exact for orthogonal normalized bases.
    nfft = d.size + tp.size - 1
    overlap = noise_weighted_template_correlation(
        tp, tc, fs, psd_freq, psd, nfft=nfft, **kwargs
    )
    return lags_p, combine_two_basis_snr(snr_p, snr_c, overlap)


def _valid_snr_and_detection(
    snr: np.ndarray,
    lags: np.ndarray,
    fs: float,
    record_n: int,
    template_n: int,
    ref_idx: int,
    threshold: float,
) -> tuple[np.ndarray, DetectionResult]:
    """Mask partial-overlap lags and detect the strongest valid local peak."""
    candidate_t = (lags + ref_idx) / fs
    full_overlap = (lags >= 0) & (lags <= max(0, record_n - template_n))
    inside_time = (candidate_t >= 0.0) & (candidate_t < record_n / fs)
    valid = full_overlap & inside_time & np.isfinite(snr)

    for_detection = np.full_like(snr, -np.inf, dtype=float)
    for_detection[valid] = snr[valid]
    det = detect_best_candidate(
        for_detection,
        lags,
        fs,
        threshold=threshold,
        template_reference_index=ref_idx,
        prominence=0.5,
    )
    for_display = np.full_like(snr, np.nan, dtype=float)
    for_display[valid] = snr[valid]
    return for_display, det


def _analyze_detector(
    record: StrainRecord,
    spec: EventSpec,
    template_plus: np.ndarray,
    template_cross: np.ndarray,
    template_ref_idx: int,
    threshold: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
) -> DetectorResult:
    x = record.strain.astype(np.float64, copy=False)
    fs = record.fs

    # Conditioning is retained for visualization/STFT/sonification.
    filtered = _filter_signal(x, fs, spec)
    white, _, _, _ = whiten(
        filtered,
        fs,
        psd_freq,
        psd,
        fmin=spec.fband[0],
        fmax=min(spec.fband[1], fs / 2 - 1.0),
        standardize=True,
    )

    stft_n = 256 if fs <= 4096 else 512
    stft_n = min(stft_n, white.size)
    stft_overlap = min(int(stft_n * 0.75), stft_n - 1)
    f_stft, t_stft, Z = scipy_stft(
        white,
        fs,
        nperseg=stft_n,
        noverlap=stft_overlap,
        window="hann",
    )

    # Detection uses the measured PSD directly rather than a separately filtered
    # noise model. Narrow lines are naturally down-weighted by 1/S_n(f).
    lags, snr_full = _matched_quadrature(
        x,
        template_plus,
        template_cross,
        fs,
        psd_freq,
        psd,
        spec.fband,
    )
    snr, det = _valid_snr_and_detection(
        snr_full, lags, fs, x.size, template_plus.size, template_ref_idx, threshold
    )

    return DetectorResult(
        record=record,
        raw=x,
        filtered=filtered,
        white=white,
        psd_freq=psd_freq,
        psd=psd,
        stft_freq=f_stft,
        stft_time=t_stft,
        stft_power=spectrogram_power(Z),
        lags=lags,
        snr=snr,
        detection=det,
    )


def _candidate_gps(det: DetectorResult) -> float | None:
    if not det.detection.detected or det.detection.peak_time_s is None:
        return None
    if det.record.start_time is None:
        return det.detection.peak_time_s
    return det.record.start_time + det.detection.peak_time_s


def _estimate_network_delay_from_candidates(
    h1: DetectorResult,
    l1: DetectorResult,
    *,
    max_abs_delay_s: float = 0.012,
) -> DelayEstimate | None:
    """Cross-correlate a common absolute-time window around detected candidates."""
    h1_gps = _candidate_gps(h1)
    l1_gps = _candidate_gps(l1)
    if h1_gps is None or l1_gps is None:
        return None
    if abs(h1_gps - l1_gps) > 0.05:
        return None
    if h1.record.start_time is None or l1.record.start_time is None:
        return None

    center_abs = 0.5 * (h1_gps + l1_gps)
    center_h = center_abs - h1.record.start_time
    center_l = center_abs - l1.record.start_time
    half = 0.50
    fs = h1.record.fs

    def extract(d: DetectorResult, center: float) -> np.ndarray | None:
        i0 = int(round((center - half) * fs))
        i1 = int(round((center + half) * fs))
        if i0 < 0 or i1 > d.white.size or i1 - i0 < int(0.1 * fs):
            return None
        return d.white[i0:i1]

    xh = extract(h1, center_h)
    xl = extract(l1, center_l)
    if xh is None or xl is None:
        return None
    n = min(xh.size, xl.size)
    return estimate_delay(
        xh[:n], xl[:n], fs,
        max_abs_delay_s=max_abs_delay_s,
        use_absolute_peak=True,
    )


def analyze_real_case(spec: EventSpec, *, threshold: float | None = None) -> RealCaseResult:
    if spec.kind == "synthetic":
        raise ValueError("analyze_real_case only accepts real/quiet cases")
    if spec.h1_file is None or spec.l1_file is None or spec.template_file is None:
        raise ValueError("Case is missing local file paths")

    h1_full = _with_metadata_fallback(load_common_gwosc_hdf5(spec.h1_file), spec, "H1")
    l1_full = _with_metadata_fallback(load_common_gwosc_hdf5(spec.l1_file), spec, "L1")
    if not np.isclose(h1_full.fs, l1_full.fs):
        raise ValueError("H1 and L1 sample rates differ")
    fs = h1_full.fs

    # PSDs come from the full detector files, before any quiet-window slicing.
    h1_psd_freq, h1_psd = _psd_for_full_record(h1_full, spec)
    l1_psd_freq, l1_psd = _psd_for_full_record(l1_full, spec)

    h1_record, h1_event_t = _prepare_record(h1_full, spec)
    l1_record, l1_event_t = _prepare_record(l1_full, spec)
    if spec.expected_signal and h1_event_t is None:
        raise ValueError("Catalog event GPS does not fall inside the selected H1 record")
    if spec.expected_signal and l1_event_t is None:
        raise ValueError("Catalog event GPS does not fall inside the selected L1 record")
    event_t = h1_event_t if spec.expected_signal else None

    template = load_losc_template(spec.template_file, fs=fs)
    compact = template.compact_slice()
    # Detection templates remain unfiltered; the PSD/frequency mask performs the
    # noise weighting. Filtering is only part of the displayed conditioned data.
    tplus = template.plus[compact].copy()
    tcross = template.cross[compact].copy()
    ref_idx = template.peak_index - compact.start

    th = float(spec.threshold if threshold is None else threshold)
    h1 = _analyze_detector(
        h1_record, spec, tplus, tcross, ref_idx, th, h1_psd_freq, h1_psd
    )
    l1 = _analyze_detector(
        l1_record, spec, tplus, tcross, ref_idx, th, l1_psd_freq, l1_psd
    )

    # Coincidence uses absolute detector timestamps where available.
    h1_time = _candidate_gps(h1)
    l1_time = _candidate_gps(l1)
    coincidence = coincidence_check(
        h1_detected=h1.detection.detected,
        l1_detected=l1.detection.detected,
        h1_time_s=h1_time,
        l1_time_s=l1_time,
        max_time_difference_s=0.012,
    )
    delay = _estimate_network_delay_from_candidates(h1, l1, max_abs_delay_s=0.012)

    phase_rel = template.phase_boundaries_relative()
    phase_abs: dict[str, tuple[float, float]] = {}
    if event_t is not None:
        for name, (a, b) in phase_rel.items():
            phase_abs[name] = (event_t + a, event_t + b)

    return RealCaseResult(
        spec=spec,
        template=template,
        template_plus_compact=tplus,
        template_cross_compact=tcross,
        template_reference_index=ref_idx,
        h1=h1,
        l1=l1,
        event_time_s=event_t,
        network_delay=delay,
        coincidence=coincidence,
        phase_intervals_absolute=phase_abs,
    )
