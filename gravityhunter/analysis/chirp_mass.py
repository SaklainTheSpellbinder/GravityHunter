from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from gravityhunter.catalog import EventSpec
from gravityhunter.dsp.stft_tools import scipy_stft, spectrogram_power
from gravityhunter.dsp.templates import TemplateRecord, instantaneous_frequency

G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30


@dataclass
class ChirpMassEstimate:
    detector: str
    success: bool
    message: str
    ridge_time_s: np.ndarray
    ridge_frequency_hz: np.ndarray
    smooth_frequency_hz: np.ndarray
    dfdt_hz_per_s: np.ndarray
    ridge_confidence: np.ndarray
    pointwise_mass_solar: np.ndarray
    estimated_mass_solar: float | None
    template_reference_mass_solar: float | None
    catalog_source_mass_solar: float | None
    fit_r2: float | None
    monotonic_fraction: float | None
    stft_freq: np.ndarray
    stft_time: np.ndarray
    stft_power: np.ndarray
    inspiral_interval_s: tuple[float, float] | None


def chirp_mass_from_components(m1: float, m2: float) -> float:
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def chirp_mass_from_f_dfdt(f_hz: np.ndarray, dfdt_hz_per_s: np.ndarray) -> np.ndarray:
    """Leading-order detector-frame chirp-mass estimate from observed f and df/dt.

    M = c^3/G * [(5/96) pi^(-8/3) f^(-11/3) df/dt]^(3/5)
    """
    f = np.asarray(f_hz, dtype=float)
    dfdt = np.asarray(dfdt_hz_per_s, dtype=float)
    out = np.full(np.broadcast_shapes(f.shape, dfdt.shape), np.nan, dtype=float)
    f, dfdt = np.broadcast_arrays(f, dfdt)
    valid = np.isfinite(f) & np.isfinite(dfdt) & (f > 0) & (dfdt > 0)
    if np.any(valid):
        inside = (
            (5.0 / 96.0)
            * np.pi ** (-8.0 / 3.0)
            * f[valid] ** (-11.0 / 3.0)
            * dfdt[valid]
        )
        out[valid] = (C_SI**3 / G_SI) * inside ** (3.0 / 5.0) / M_SUN_KG
    return out


def _fit_mass_from_ridge(
    t: np.ndarray,
    f: np.ndarray,
    confidence: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray, float | None, float | None]:
    """Robustly estimate chirp mass from the integrated inspiral law.

    Instead of fitting noisy numerical derivatives directly, fit

        t = t_c - A f^(-8/3)

    where

        A = (5/256) * pi^(-8/3) * (G M_c / c^3)^(-5/3).

    This is much more stable on real spectrogram ridges. We still compute
    df/dt for visualization and pointwise educational mass estimates.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if confidence is None:
        confidence = np.ones_like(f)
    else:
        confidence = np.asarray(confidence, dtype=float)

    finite = np.isfinite(t) & np.isfinite(f) & np.isfinite(confidence) & (f > 0)
    t, f, confidence = t[finite], f[finite], confidence[finite]
    if t.size < 6:
        return None, np.full_like(f, np.nan), None, None

    order = np.argsort(t)
    t, f, confidence = t[order], f[order], confidence[order]

    win = min(11, t.size if t.size % 2 else t.size - 1)
    win = max(win, 5)
    if win >= t.size:
        win = t.size if t.size % 2 else t.size - 1
    if win >= 5:
        smooth = savgol_filter(f, window_length=win, polyorder=min(2, win - 2), mode="interp")
    else:
        smooth = f.copy()

    # Diagnose the measured ridge *before* enforcing monotonicity; otherwise the
    # monotonic-fraction quality metric would be artificially close to one.
    raw_dfdt = np.gradient(smooth, t)
    monotonic_fraction = float(np.mean(raw_dfdt > 0))
    # The leading-order inspiral fit assumes increasing frequency. Suppress only
    # downward STFT-bin jitter for the fit/derivative display.
    smooth = np.maximum.accumulate(smooth)
    dfdt = np.gradient(smooth, t)

    # Reject weak/local-noise ridge points. Confidence is peak/median power in
    # the template-guided search corridor, so values near 1 carry little evidence.
    conf_floor = max(1.15, float(np.nanpercentile(confidence, 20)))
    valid = (
        np.isfinite(smooth)
        & (smooth > 20.0)
        & np.isfinite(confidence)
        & (confidence >= conf_floor)
    )
    if valid.size >= 4:
        valid[:2] = False
        valid[-2:] = False
    if np.sum(valid) < 5:
        # Fall back to all finite ridge points rather than returning a derivative
        # failure merely because the confidence distribution is flat.
        valid = np.isfinite(smooth) & (smooth > 20.0) & np.isfinite(confidence)
        if valid.size >= 4:
            valid[:2] = False
            valid[-2:] = False
    if np.sum(valid) < 5:
        return None, dfdt, None, monotonic_fraction

    tv = t[valid]
    fv = smooth[valid]
    w = np.clip(confidence[valid], 0.25, 30.0)
    x = fv ** (-8.0 / 3.0)

    # Weighted linear model t = intercept + slope*x. Physical inspiral requires
    # slope < 0, with A = -slope > 0.
    X = np.column_stack([np.ones_like(x), x])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], tv * sw, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    pred = intercept + slope * x
    resid = tv - pred

    # Robust MAD gate and second fit.
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad > 0:
        keep = np.abs(resid - med) <= 3.5 * 1.4826 * mad
        if np.sum(keep) >= 5:
            x2, t2, w2 = x[keep], tv[keep], w[keep]
            X2 = np.column_stack([np.ones_like(x2), x2])
            sw2 = np.sqrt(w2)
            beta, *_ = np.linalg.lstsq(X2 * sw2[:, None], t2 * sw2, rcond=None)
            intercept, slope = float(beta[0]), float(beta[1])
            x, tv, w = x2, t2, w2
            pred = intercept + slope * x

    A = -slope
    if not np.isfinite(A) or A <= 0:
        return None, dfdt, None, monotonic_fraction

    # A = (5/256) pi^(-8/3) (G M/c^3)^(-5/3)
    inside = (5.0 / (256.0 * A)) * np.pi ** (-8.0 / 3.0)
    if not np.isfinite(inside) or inside <= 0:
        return None, dfdt, None, monotonic_fraction
    mass_kg = (C_SI**3 / G_SI) * inside ** (3.0 / 5.0)
    mass_solar = float(mass_kg / M_SUN_KG)

    ybar = np.average(tv, weights=w)
    ss_res = float(np.sum(w * (tv - pred) ** 2))
    ss_tot = float(np.sum(w * (tv - ybar) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return mass_solar, dfdt, r2, monotonic_fraction

def estimate_chirp_mass_from_ridge(
    ridge_time_s: np.ndarray,
    ridge_frequency_hz: np.ndarray,
    confidence: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray, np.ndarray, np.ndarray, float | None, float | None]:
    """Public helper used by tests and educational notebooks.

    Returns (mass, smooth_f, dfdt, pointwise_mass, r2, monotonic_fraction).
    """
    t = np.asarray(ridge_time_s, dtype=float)
    f = np.asarray(ridge_frequency_hz, dtype=float)
    c = np.ones_like(f) if confidence is None else np.asarray(confidence, dtype=float)
    finite = np.isfinite(t) & np.isfinite(f) & np.isfinite(c) & (f > 0)
    t2, f2, c2 = t[finite], f[finite], c[finite]
    if t2.size < 6:
        nan = np.full_like(f2, np.nan)
        return None, f2.copy(), nan, nan, None, None

    order = np.argsort(t2)
    t2, f2, c2 = t2[order], f2[order], c2[order]
    win = min(11, t2.size if t2.size % 2 else t2.size - 1)
    if win >= 5:
        smooth = savgol_filter(f2, win, min(2, win - 2), mode="interp")
    else:
        smooth = f2.copy()
    smooth = np.maximum.accumulate(smooth)
    mass, dfdt, r2, mono = _fit_mass_from_ridge(t2, f2, c2)
    pointwise = chirp_mass_from_f_dfdt(smooth, dfdt)
    return mass, smooth, dfdt, pointwise, r2, mono


def _dedicated_stft_params(fs: float, inspiral_duration_s: float, n_samples: int) -> tuple[int, int]:
    if inspiral_duration_s <= 0.45:
        target = 512
    elif inspiral_duration_s <= 1.5:
        target = 1024
    else:
        target = 2048
    nperseg = min(target, n_samples)
    # Keep power-of-two where practical.
    if nperseg >= 256:
        nperseg = 2 ** int(np.floor(np.log2(nperseg)))
    nperseg = max(128, nperseg)
    noverlap = int(round(0.875 * nperseg))
    noverlap = min(noverlap, nperseg - 1)
    return nperseg, noverlap


def estimate_chirp_mass_from_detector(
    *,
    white_strain: np.ndarray,
    fs: float,
    template: TemplateRecord,
    event_time_s: float,
    phase_intervals_absolute: dict[str, tuple[float, float]],
    spec: EventSpec,
    detector: str,
) -> ChirpMassEstimate:
    """Extract a template-guided ridge from the *real* detector spectrogram.

    The reference waveform supplies only a local search corridor. The frequency
    sample selected at each time frame comes from the detector STFT power.
    This is intentionally an educational estimator, not LIGO parameter inference.
    """
    insp = phase_intervals_absolute.get("Inspiral")
    template_ref_mass = None
    if template.m1 is not None and template.m2 is not None:
        template_ref_mass = chirp_mass_from_components(template.m1, template.m2)

    if insp is None or insp[1] <= insp[0]:
        return ChirpMassEstimate(
            detector, False, "No usable inspiral interval was defined.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            np.array([]), np.array([]), np.empty((0, 0)), None,
        )

    duration = insp[1] - insp[0]
    nperseg, noverlap = _dedicated_stft_params(fs, duration, len(white_strain))
    f_stft, t_stft, Z = scipy_stft(
        white_strain,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )
    power = spectrogram_power(Z)

    # Keep only the inspiral before the merger-neighborhood boundary.
    tm = (t_stft >= insp[0]) & (t_stft <= insp[1])
    low = max(20.0, float(spec.fband[0]))
    high = min(float(spec.fband[1]), fs / 2 - 1.0, 600.0)
    fm = (f_stft >= low) & (f_stft <= high)
    if np.sum(tm) < 6 or np.sum(fm) < 4:
        return ChirpMassEstimate(
            detector, False, "The event window is too short for a stable spectrogram ridge estimate.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    frame_times = t_stft[tm]
    band_freq = f_stft[fm]
    band_power = power[fm][:, tm]

    # Expected ridge is used only as a corridor so persistent detector lines do
    # not win the argmax. Selected ridge points still come from real STFT power.
    tf = template.time_from_peak
    ref_f = instantaneous_frequency(template)
    good_ref = np.isfinite(ref_f) & (ref_f >= low) & (ref_f <= high)
    if np.sum(good_ref) < 4:
        return ChirpMassEstimate(
            detector, False, "Reference waveform does not provide enough inspiral-frequency support.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    expected = np.interp(
        frame_times - event_time_s,
        tf[good_ref],
        ref_f[good_ref],
        left=np.nan,
        right=np.nan,
    )

    ridge_t: list[float] = []
    ridge_f: list[float] = []
    ridge_c: list[float] = []
    for j, (tj, f0) in enumerate(zip(frame_times, expected)):
        if not np.isfinite(f0):
            continue
        tol = max(14.0, 0.22 * float(f0))
        local = np.abs(band_freq - f0) <= tol
        if np.sum(local) < 2:
            continue
        p = band_power[local, j]
        ff = band_freq[local]
        idx = int(np.argmax(p))
        peak_power = float(p[idx])
        background = float(np.median(p)) + np.finfo(float).tiny
        confidence = peak_power / background

        # A local power-weighted centroid is less jittery than taking one STFT
        # bin's argmax, while every selected frequency still comes from the real
        # detector spectrogram inside the template-guided corridor.
        excess = np.maximum(p - background, 0.0)
        if np.sum(excess) > 0:
            weights = excess**2
            f_pick = float(np.sum(ff * weights) / np.sum(weights))
        else:
            f_pick = float(ff[idx])
        ridge_t.append(float(tj))
        ridge_f.append(f_pick)
        ridge_c.append(float(confidence))

    t = np.asarray(ridge_t)
    f = np.asarray(ridge_f)
    c = np.asarray(ridge_c)
    if t.size < 6:
        return ChirpMassEstimate(
            detector, False, "Too few detector ridge points survived the inspiral search corridor.",
            t, f, f.copy(), np.full_like(f, np.nan), c, np.full_like(f, np.nan),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    mass, smooth, dfdt, pointwise, r2, mono = estimate_chirp_mass_from_ridge(t, f, c)
    if mass is None or not np.isfinite(mass) or mass <= 0 or mass > 300 or (r2 is not None and r2 < 0.15):
        success = False
        message = "A ridge was extracted, but the integrated leading-order inspiral fit was not stable enough for a credible chirp-mass estimate."
        mass = None
    else:
        success = True
        message = "Estimated from the real detector ridge by fitting the integrated leading-order inspiral law t = tc - A f^(-8/3)."

    return ChirpMassEstimate(
        detector=detector,
        success=success,
        message=message,
        ridge_time_s=t,
        ridge_frequency_hz=f,
        smooth_frequency_hz=smooth,
        dfdt_hz_per_s=dfdt,
        ridge_confidence=c,
        pointwise_mass_solar=pointwise,
        estimated_mass_solar=mass,
        template_reference_mass_solar=template_ref_mass,
        catalog_source_mass_solar=spec.chirp_mass_source,
        fit_r2=r2,
        monotonic_fraction=mono,
        stft_freq=f_stft,
        stft_time=t_stft,
        stft_power=power,
        inspiral_interval_s=insp,
    )


def estimate_chirp_mass_from_network(
    *,
    h1_white: np.ndarray,
    l1_white: np.ndarray,
    fs: float,
    template: TemplateRecord,
    event_time_s: float,
    phase_intervals_absolute: dict[str, tuple[float, float]],
    spec: EventSpec,
) -> ChirpMassEstimate:
    """Estimate chirp mass from a combined H1+L1 time-frequency ridge.

    Each detector STFT is normalized by its own median background power at each
    frequency, then the two relative-power maps are averaged.  This keeps the
    inference data-driven while making a weak ridge that is coherent across the
    detector network easier to trace than either single-detector spectrogram.
    The public template is used only to define a local frequency corridor.
    """
    insp = phase_intervals_absolute.get("Inspiral")
    template_ref_mass = None
    if template.m1 is not None and template.m2 is not None:
        template_ref_mass = chirp_mass_from_components(template.m1, template.m2)

    if insp is None or insp[1] <= insp[0]:
        return ChirpMassEstimate(
            "H1+L1", False, "No usable inspiral interval was defined.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            np.array([]), np.array([]), np.empty((0, 0)), None,
        )

    if len(h1_white) != len(l1_white):
        n = min(len(h1_white), len(l1_white))
        h1_white = np.asarray(h1_white[:n], dtype=float)
        l1_white = np.asarray(l1_white[:n], dtype=float)

    duration = insp[1] - insp[0]
    nperseg, noverlap = _dedicated_stft_params(fs, duration, len(h1_white))
    f1, t1, Z1 = scipy_stft(h1_white, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    f2, t2, Z2 = scipy_stft(l1_white, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    if not np.allclose(f1, f2) or not np.allclose(t1, t2):
        return ChirpMassEstimate(
            "H1+L1", False, "H1/L1 STFT grids do not match.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f1, t1, np.empty((0, 0)), insp,
        )

    p1 = spectrogram_power(Z1)
    p2 = spectrogram_power(Z2)
    tiny = np.finfo(float).tiny
    bg1 = np.median(p1, axis=1, keepdims=True)
    bg2 = np.median(p2, axis=1, keepdims=True)
    relative1 = p1 / np.maximum(bg1, tiny)
    relative2 = p2 / np.maximum(bg2, tiny)
    power = 0.5 * (relative1 + relative2)
    f_stft, t_stft = f1, t1

    tm = (t_stft >= insp[0]) & (t_stft <= insp[1])
    low = max(20.0, float(spec.fband[0]))
    high = min(float(spec.fband[1]), fs / 2 - 1.0, 600.0)
    fm = (f_stft >= low) & (f_stft <= high)
    if np.sum(tm) < 6 or np.sum(fm) < 4:
        return ChirpMassEstimate(
            "H1+L1", False, "The inspiral interval is too short for a stable network ridge estimate.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    frame_times = t_stft[tm]
    band_freq = f_stft[fm]
    band_power = power[fm][:, tm]

    tf = template.time_from_peak
    ref_f = instantaneous_frequency(template)
    good_ref = np.isfinite(ref_f) & (ref_f >= low) & (ref_f <= high)
    if np.sum(good_ref) < 4:
        return ChirpMassEstimate(
            "H1+L1", False, "Reference waveform does not provide enough inspiral-frequency support.",
            *(np.array([], dtype=float) for _ in range(6)),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    expected = np.interp(
        frame_times - event_time_s,
        tf[good_ref],
        ref_f[good_ref],
        left=np.nan,
        right=np.nan,
    )

    ridge_t: list[float] = []
    ridge_f: list[float] = []
    ridge_c: list[float] = []
    for j, (tj, f0) in enumerate(zip(frame_times, expected)):
        if not np.isfinite(f0):
            continue
        tol = max(12.0, 0.18 * float(f0))
        local = np.abs(band_freq - f0) <= tol
        if np.sum(local) < 2:
            continue
        p = band_power[local, j]
        ff = band_freq[local]
        idx = int(np.argmax(p))
        peak_power = float(p[idx])
        background = float(np.median(p)) + tiny
        confidence = peak_power / background
        excess = np.maximum(p - background, 0.0)
        if np.sum(excess) > 0:
            weights = excess**2
            f_pick = float(np.sum(ff * weights) / np.sum(weights))
        else:
            f_pick = float(ff[idx])
        ridge_t.append(float(tj))
        ridge_f.append(f_pick)
        ridge_c.append(float(confidence))

    t = np.asarray(ridge_t)
    f = np.asarray(ridge_f)
    c = np.asarray(ridge_c)
    if t.size < 6:
        return ChirpMassEstimate(
            "H1+L1", False, "Too few network ridge points survived the inspiral search corridor.",
            t, f, f.copy(), np.full_like(f, np.nan), c, np.full_like(f, np.nan),
            None, template_ref_mass, spec.chirp_mass_source, None, None,
            f_stft, t_stft, power, insp,
        )

    mass, smooth, dfdt, pointwise, r2, mono = estimate_chirp_mass_from_ridge(t, f, c)
    median_conf = float(np.nanmedian(c)) if c.size else 0.0
    credible = (
        mass is not None
        and np.isfinite(mass)
        and 2.0 < mass < 150.0
        and r2 is not None and r2 >= 0.35
        and mono is not None and mono >= 0.50
        and median_conf >= 1.10
    )
    if credible:
        message = "Network ridge fit completed using combined H1/L1 relative STFT power."
    else:
        mass = None
        message = (
            "The network spectrogram contains an inspiral-like ridge, but the "
            "leading-order fit does not pass the configured quality checks."
        )

    return ChirpMassEstimate(
        detector="H1+L1",
        success=bool(credible),
        message=message,
        ridge_time_s=t,
        ridge_frequency_hz=f,
        smooth_frequency_hz=smooth,
        dfdt_hz_per_s=dfdt,
        ridge_confidence=c,
        pointwise_mass_solar=pointwise,
        estimated_mass_solar=mass,
        template_reference_mass_solar=template_ref_mass,
        catalog_source_mass_solar=spec.chirp_mass_source,
        fit_r2=r2,
        monotonic_fraction=mono,
        stft_freq=f_stft,
        stft_time=t_stft,
        stft_power=power,
        inspiral_interval_s=insp,
    )
