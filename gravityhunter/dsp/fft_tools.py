from __future__ import annotations

import numpy as np


def time_axis(n: int, fs: float) -> np.ndarray:
    """Relative time axis, shape (N,)."""
    return np.arange(n, dtype=float) / fs


def dft_manual(x: np.ndarray) -> np.ndarray:
    """
    Direct O(N^2) DFT from the definition.
    Use only for tiny educational arrays.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    idx_n = np.arange(n)
    idx_k = idx_n.reshape(-1, 1)
    basis = np.exp(-2j * np.pi * idx_k * idx_n / n)
    return basis @ x


def fft_full(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Full FFT and correctly ordered np.fft.fftfreq axis.

    Returns
    -------
    f : (N,)
    X : (N,) complex
    """
    x = np.asarray(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(x.size, d=1.0 / fs)
    return f, X


def fft_centered(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered [-fs/2, ..., 0, ..., +fs/2] style representation."""
    f, X = fft_full(x, fs)
    return np.fft.fftshift(f), np.fft.fftshift(X)


def rfft_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-negative-frequency FFT for a real signal.

    Returns
    -------
    f : (N//2+1,) for even N
    X : same shape, complex
    """
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return f, X


def one_sided_amplitude(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided sinusoid-amplitude spectrum.

    DC and Nyquist (for even N) are not doubled.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    f, X = rfft_spectrum(x, fs)

    amp = np.abs(X) / n
    if n > 1:
        if n % 2 == 0:
            amp[1:-1] *= 2.0
        else:
            amp[1:] *= 2.0
    return f, amp
