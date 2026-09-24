from __future__ import annotations

from io import BytesIO

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


def normalize_audio(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.int16)
    x = x - np.mean(x)
    m = float(np.max(np.abs(x)))
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(x, dtype=np.int16)
    y = np.clip(x / m * peak, -1.0, 1.0)
    return np.asarray(y * 32767, dtype=np.int16)


def frequency_shift(x: np.ndarray, fs: float, shift_hz: float) -> np.ndarray:
    """Translate a real signal upward using its analytic representation.

    This is a sonification/display transform only; it is never used by the
    detector.  The analytic-signal modulation avoids the crude bin-copy
    artifacts of shifting an rFFT array directly.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or shift_hz <= 0:
        return x.copy()
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    if shift_hz >= fs / 2:
        raise ValueError("shift_hz must be below Nyquist")
    t = np.arange(x.size, dtype=float) / fs
    analytic = hilbert(x)
    shifted = analytic * np.exp(2j * np.pi * shift_hz * t)
    return np.real(shifted)


def wav_bytes(
    x: np.ndarray,
    fs: float,
    *,
    shift_hz: float = 0.0,
    playback_rate: float = 1.0,
) -> bytes:
    """Return WAV bytes suitable for Streamlit st.audio()."""
    if playback_rate <= 0:
        raise ValueError("playback_rate must be positive")
    y = frequency_shift(x, fs, shift_hz)
    rate = max(1000, int(round(fs * playback_rate)))
    pcm = normalize_audio(y)
    bio = BytesIO()
    wavfile.write(bio, rate, pcm)
    return bio.getvalue()
