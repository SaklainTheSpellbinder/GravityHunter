from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gravityhunter.catalog import EventSpec
from gravityhunter.dsp.detector import DetectionResult, detect_best_candidate
from gravityhunter.dsp.filters import apply_bandpass_zero_phase, apply_multiple_notches
from gravityhunter.dsp.loader import StrainRecord, load_common_gwosc_hdf5, slice_by_time
from gravityhunter.dsp.matched_filter import matched_filter_snr
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


def _event_time_in_record(record: StrainRecord, spec: EventSpec) -> float | None:
    if spec.gps_event is None:
        return None
    if record.start_time is not None:
        return float(spec.gps_event - record.start_time)
    # All bundled event files are 32-second event-centered releases. Fallback
    # only when metadata does not carry GPSstart.
    return record.duration / 2.0


def _prepare_record(record: StrainRecord, spec: EventSpec) -> tuple[StrainRecord, float | None]:
    if spec.kind == "quiet":
        assert spec.quiet_window_s is not None
        return slice_by_time(record, *spec.quiet_window_s), None
    return record, _event_time_in_record(record, spec)


def _noise_reference(x: np.ndarray, fs: float, event_t: float | None) -> np.ndarray:
    n = x.size
    if event_t is None:
        return x
    # Use off-source data from both sides; exclude a generous neighborhood of
    # the known event only for PSD estimation, not for the detector search.
    guard = 3.0
    i0 = max(0, int((event_t - guard) * fs))
    i1 = min(n, int((event_t + guard) * fs))
    parts = []
    if i0 >= int(2 * fs):
        parts.append(x[:i0])
    if n - i1 >= int(2 * fs):
        parts.append(x[i1:])
    if not parts:
        return x
    return np.concatenate(parts)


def _filter_signal(x: np.ndarray, fs: float, spec: EventSpec) -> np.ndarray:
    valid_notches = [f for f in spec.notches_hz if 0 < f < fs / 2]
    y = apply_multiple_notches(x, fs, valid_notches, q=30.0) if valid_notches else x.copy()
    low, high = spec.fband
    high = min(high, fs / 2 - 1.0)
    y, _ = apply_bandpass_zero_phase(y, fs, low, high, order=4)
    return y


def _matched_quadrature(
    data: np.ndarray,
    tplus: np.ndarray,
    tcross: np.ndarray,
    fs: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    fband: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = dict(fmin=fband[0], fmax=min(fband[1], fs / 2 - 1.0))
    lags_p, snr_p, _ = matched_filter_snr(data, tplus, fs, psd_freq, psd, **kwargs)
    lags_c, snr_c, _ = matched_filter_snr(data, tcross, fs, psd_freq, psd, **kwargs)
    if not np.array_equal(lags_p, lags_c):
        raise RuntimeError("Plus/cross matched-filter lag axes differ")
    return lags_p, np.sqrt(snr_p**2 + snr_c**2)


def _valid_detection(
    snr: np.ndarray,
    lags: np.ndarray,
    fs: float,
    record_duration: float,
    ref_idx: int,
    threshold: float,
) -> DetectionResult:
    candidate_t = (lags + ref_idx) / fs
    valid = (candidate_t >= 0.0) & (candidate_t <= record_duration)
    masked = np.full_like(snr, -np.inf, dtype=float)
    masked[valid] = snr[valid]
    return detect_best_candidate(
        masked,
        lags,
        fs,
        threshold=threshold,
        template_reference_index=ref_idx,
        prominence=0.5,
    )


def _analyze_detector(
    record: StrainRecord,
    spec: EventSpec,
    event_t: float | None,
    template_plus: np.ndarray,
    template_cross: np.ndarray,
    template_ref_idx: int,
    threshold: float,
) -> DetectorResult:
    x = record.strain.astype(np.float64, copy=False)
    fs = record.fs

    noise = _noise_reference(x, fs, event_t)
    nperseg = int(min(4096, max(256, 2 ** int(np.floor(np.log2(max(256, noise.size // 4)))))))
    nperseg = min(nperseg, noise.size)
    if nperseg % 2:
        nperseg -= 1
    nperseg = max(128, nperseg)
    f_psd, psd = welch_psd(noise, fs, nperseg=nperseg, noverlap=nperseg // 2)

    filtered = _filter_signal(x, fs, spec)
    white, _, _, _ = whiten(
        filtered,
        fs,
        f_psd,
        psd,
        fmin=spec.fband[0],
        fmax=min(spec.fband[1], fs / 2 - 1.0),
        standardize=True,
    )

    # STFT of the whitened full selected record.
    stft_n = 256 if fs <= 4096 else 512
    stft_overlap = int(stft_n * 0.75)
    f_stft, t_stft, Z = scipy_stft(
        white,
        fs,
        nperseg=stft_n,
        noverlap=stft_overlap,
        window="hann",
    )

    lags, snr = _matched_quadrature(
        filtered,
        template_plus,
        template_cross,
        fs,
        f_psd,
        psd,
        spec.fband,
    )
    det = _valid_detection(snr, lags, fs, record.duration, template_ref_idx, threshold)

    return DetectorResult(
        record=record,
        raw=x,
        filtered=filtered,
        white=white,
        psd_freq=f_psd,
        psd=psd,
        stft_freq=f_stft,
        stft_time=t_stft,
        stft_power=spectrogram_power(Z),
        lags=lags,
        snr=snr,
        detection=det,
    )


def analyze_real_case(spec: EventSpec, *, threshold: float | None = None) -> RealCaseResult:
    if spec.kind == "synthetic":
        raise ValueError("analyze_real_case only accepts real/quiet cases")
    if spec.h1_file is None or spec.l1_file is None or spec.template_file is None:
        raise ValueError("Case is missing local file paths")

    h1_full = load_common_gwosc_hdf5(spec.h1_file)
    l1_full = load_common_gwosc_hdf5(spec.l1_file)
    if not np.isclose(h1_full.fs, l1_full.fs):
        raise ValueError("H1 and L1 sample rates differ")
    fs = h1_full.fs

    h1_record, h1_event_t = _prepare_record(h1_full, spec)
    l1_record, l1_event_t = _prepare_record(l1_full, spec)
    event_t = h1_event_t if spec.expected_signal else None

    template = load_losc_template(spec.template_file, fs=fs)
    full_plus = _filter_signal(template.plus, fs, spec)
    full_cross = _filter_signal(template.cross, fs, spec)
    compact = template.compact_slice()
    tplus = full_plus[compact]
    tcross = full_cross[compact]
    ref_idx = template.peak_index - compact.start

    th = float(spec.threshold if threshold is None else threshold)
    h1 = _analyze_detector(h1_record, spec, h1_event_t, tplus, tcross, ref_idx, th)
    l1 = _analyze_detector(l1_record, spec, l1_event_t, tplus, tcross, ref_idx, th)

    delay = None
    if spec.expected_signal and event_t is not None:
        # Estimate detector-to-detector lag only around the real coalescence.
        half = 0.20
        i0_h = max(0, int((h1_event_t - half) * fs))
        i1_h = min(h1.white.size, int((h1_event_t + half) * fs))
        i0_l = max(0, int((l1_event_t - half) * fs))
        i1_l = min(l1.white.size, int((l1_event_t + half) * fs))
        n = min(i1_h - i0_h, i1_l - i0_l)
        if n > int(0.1 * fs):
            delay = estimate_delay(
                h1.white[i0_h:i0_h+n],
                l1.white[i0_l:i0_l+n],
                fs,
                max_abs_delay_s=0.02,
                use_absolute_peak=True,
            )

    coincidence = coincidence_check(
        h1_detected=h1.detection.detected,
        l1_detected=l1.detection.detected,
        h1_time_s=h1.detection.peak_time_s,
        l1_time_s=l1.detection.peak_time_s,
        max_time_difference_s=0.02,
    )

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
