"""
Run an end-to-end synthetic GravityHunter demonstration before using LIGO data.

This example intentionally uses a simple chirp injection so you can verify:
    PSD -> notch/band-pass -> whitening -> STFT -> matched filter -> SNR
"""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp

from gravityhunter.dsp.psd import welch_psd
from gravityhunter.dsp.filters import (
    apply_bandpass_zero_phase,
    apply_notch_zero_phase,
)
from gravityhunter.dsp.whitening import whiten
from gravityhunter.dsp.stft_tools import scipy_stft, spectrogram_power
from gravityhunter.dsp.matched_filter import matched_filter_snr
from gravityhunter.dsp.detector import detect_best_candidate


def build_demo(seed: int = 7):
    fs = 1024.0
    duration = 16.0
    n = int(fs * duration)
    t = np.arange(n) / fs

    rng = np.random.default_rng(seed)

    # Background: broadband noise + strong persistent 60 Hz line
    noise = rng.normal(0.0, 1.0, n)
    line60 = 2.0 * np.sin(2 * np.pi * 60.0 * t)
    background = noise + line60

    # A short increasing-frequency chirp template.
    template_duration = 1.5
    m = int(fs * template_duration)
    tt = np.arange(m) / fs

    template = chirp(
        tt,
        f0=35.0,
        f1=220.0,
        t1=template_duration,
        method="quadratic",
    )

    # Smooth edges so the finite template is not abruptly cut.
    template *= np.hanning(m)
    template /= np.max(np.abs(template))

    injection_start = int(8.0 * fs)
    data = background.copy()
    data[injection_start:injection_start + m] += 1.5 * template

    # 1) Estimate detector/background PSD from a noise-only region.
    noise_reference = background[: int(6 * fs)]
    f_psd, psd = welch_psd(
        noise_reference,
        fs,
        nperseg=2048,
        noverlap=1024,
    )

    # 2) Remove a known narrow line and restrict the useful band.
    filtered, _ = apply_notch_zero_phase(data, fs, 60.0, q=30)
    filtered, _ = apply_bandpass_zero_phase(
        filtered, fs, 25.0, 300.0, order=4
    )

    # Apply same preprocessing to the template for consistency.
    template_f, _ = apply_notch_zero_phase(template, fs, 60.0, q=30)
    template_f, _ = apply_bandpass_zero_phase(
        template_f, fs, 25.0, 300.0, order=4
    )

    # 3) Whiten for visualization/STFT.
    white, _, _, _ = whiten(
        filtered,
        fs,
        f_psd,
        psd,
        fmin=25.0,
        fmax=300.0,
        standardize=True,
    )

    # 4) STFT for time-frequency visualization.
    f_stft, t_stft, Z = scipy_stft(
        white,
        fs,
        nperseg=256,
        noverlap=192,
        window="hann",
    )
    spec_power = spectrogram_power(Z)

    # 5) PSD-weighted matched filter.
    lags, snr, raw = matched_filter_snr(
        filtered,
        template_f,
        fs,
        f_psd,
        psd,
        fmin=25.0,
        fmax=300.0,
    )

    # For this synthetic example, the template reference is its start.
    # Threshold is deliberately only an educational demo value.
    result = detect_best_candidate(
        snr,
        lags,
        fs,
        threshold=5.0,
        template_reference_index=0,
        prominence=1.0,
    )

    return {
        "fs": fs,
        "time": t,
        "data": data,
        "filtered": filtered,
        "white": white,
        "template": template,
        "injection_start_s": injection_start / fs,
        "psd_freq": f_psd,
        "psd": psd,
        "stft_freq": f_stft,
        "stft_time": t_stft,
        "spectrogram_power": spec_power,
        "lags": lags,
        "snr": snr,
        "raw_matched_filter": raw,
        "detection": result,
    }


if __name__ == "__main__":
    result = build_demo()

    print("Synthetic GravityHunter demo")
    print("----------------------------")
    print(f"True injection start: {result['injection_start_s']:.3f} s")
    print(f"Detected: {result['detection'].detected}")
    print(f"Peak SNR: {result['detection'].peak_snr:.3f}")
    print(f"Candidate time: {result['detection'].peak_time_s}")
    print("Shapes:")
    print("  data:", result["data"].shape)
    print("  whitened:", result["white"].shape)
    print("  STFT:", result["spectrogram_power"].shape)
    print("  SNR:", result["snr"].shape)
