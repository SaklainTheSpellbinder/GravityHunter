from __future__ import annotations

from io import BytesIO

import numpy as np
from scipy.io import wavfile


def normalize_audio(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    m = np.max(np.abs(x)) if x.size else 0.0
    if m <= 0:
        return np.zeros_like(x, dtype=np.int16)
    y = np.clip(x / m * peak, -1.0, 1.0)
    return np.asarray(y * 32767, dtype=np.int16)


def frequency_shift(x: np.ndarray, fs: float, shift_hz: float) -> np.ndarray:
    """Educational frequency translation similar to the classic GWOSC tutorial."""
    x = np.asarray(x, dtype=np.float64)
    if shift_hz <= 0 or x.size == 0:
        return x.copy()
    X = np.fft.rfft(x)
    df = fs / x.size
    bins = int(round(shift_hz / df))
    if bins <= 0:
        return x.copy()
    if bins >= X.size:
        return np.zeros_like(x)
    Y = np.zeros_like(X)
    Y[bins:] = X[:-bins]
    return np.fft.irfft(Y, n=x.size)


def wav_bytes(
    x: np.ndarray,
    fs: float,
    *,
    shift_hz: float = 0.0,
    playback_rate: float = 1.0,
) -> bytes:
    """Return WAV bytes for Streamlit st.audio()."""
    y = frequency_shift(x, fs, shift_hz)
    rate = max(1000, int(round(fs * playback_rate)))
    pcm = normalize_audio(y)
    bio = BytesIO()
    wavfile.write(bio, rate, pcm)
    return bio.getvalue()
