from __future__ import annotations

import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    freqz,
    iirnotch,
    sosfiltfilt,
    sosfreqz,
)


def design_bandpass_sos(
    fs: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 4,
) -> np.ndarray:
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError("Need 0 < low < high < Nyquist.")

    return butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def apply_bandpass_zero_phase(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Offline zero-phase band-pass using forward/backward SOS filtering.
    Returns (filtered_signal, sos).
    """
    x = np.asarray(x, dtype=np.float64)
    sos = design_bandpass_sos(fs, low_hz, high_hz, order=order)
    y = sosfiltfilt(sos, x)
    return y, sos


def design_notch(
    fs: float,
    f0_hz: float,
    *,
    q: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0 < f0_hz < fs / 2):
        raise ValueError("Notch frequency must lie below Nyquist.")
    if q <= 0:
        raise ValueError("Q must be positive.")

    b, a = iirnotch(f0_hz, q, fs=fs)
    return b, a


def apply_notch_zero_phase(
    x: np.ndarray,
    fs: float,
    f0_hz: float,
    *,
    q: float = 30.0,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Zero-phase notch using forward/backward filtering.
    """
    x = np.asarray(x, dtype=np.float64)
    b, a = design_notch(fs, f0_hz, q=q)
    y = filtfilt(b, a, x)
    return y, (b, a)


def apply_multiple_notches(
    x: np.ndarray,
    fs: float,
    notch_hz: list[float] | tuple[float, ...],
    *,
    q: float = 30.0,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    for f0 in notch_hz:
        y, _ = apply_notch_zero_phase(y, fs, f0, q=q)
    return y


def sos_frequency_response(
    sos: np.ndarray,
    fs: float,
    *,
    worN: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns frequency in Hz and complex H(f).
    """
    f, h = sosfreqz(sos, worN=worN, fs=fs)
    return f, h


def ba_frequency_response(
    b: np.ndarray,
    a: np.ndarray,
    fs: float,
    *,
    worN: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    f, h = freqz(b, a, worN=worN, fs=fs)
    return f, h
