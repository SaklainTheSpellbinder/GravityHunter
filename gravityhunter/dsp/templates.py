from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import hilbert


@dataclass
class TemplateRecord:
    plus: np.ndarray
    cross: np.ndarray
    fs: float
    peak_index: int
    active_start_index: int
    ringdown_end_index: int
    m1: float | None = None
    m2: float | None = None
    a1: float | None = None
    a2: float | None = None
    approximant: str | None = None

    @property
    def n(self) -> int:
        return int(self.plus.size)

    @property
    def time_from_peak(self) -> np.ndarray:
        return (np.arange(self.n) - self.peak_index) / self.fs

    @property
    def complex_template(self) -> np.ndarray:
        return self.plus + 1j * self.cross

    def compact_slice(self, *, pre_pad_s: float = 0.05, post_pad_s: float = 0.04) -> slice:
        i0 = max(0, self.active_start_index - int(round(pre_pad_s * self.fs)))
        i1 = min(self.n, self.ringdown_end_index + int(round(post_pad_s * self.fs)))
        if i1 - i0 < 16:
            i0 = max(0, self.peak_index - int(self.fs))
            i1 = min(self.n, self.peak_index + int(0.2 * self.fs))
        return slice(i0, i1)

    def phase_boundaries_relative(self) -> dict[str, tuple[float, float]]:
        """Approximate morphology regions relative to the waveform peak.

        These are visualization boundaries, not a GR parameter-estimation result.
        """
        active_start = (self.active_start_index - self.peak_index) / self.fs
        ring_end = (self.ringdown_end_index - self.peak_index) / self.fs

        # Merger is a short neighborhood around peak strain. For these BBH tutorial
        # templates, a few tens of ms makes the three stages visually useful.
        merger_start = max(active_start, -0.018)
        merger_end = min(ring_end, 0.012)
        if merger_end <= merger_start:
            merger_start, merger_end = -0.015, 0.010

        return {
            "Inspiral": (active_start, merger_start),
            "Merger": (merger_start, merger_end),
            "Ringdown": (merger_end, ring_end),
        }


def _attr(meta, name: str):
    if name not in meta.attrs:
        return None
    value = meta.attrs[name]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if np.ndim(value) == 0:
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_losc_template(path: str | Path, fs: float = 4096.0) -> TemplateRecord:
    """Load the public GWOSC tutorial plus/cross template HDF5.

    The tutorial file stores two 32-second arrays under /template. The waveform
    is zero-padded, with the coalescence close to the middle of the record.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "template" not in f:
            raise KeyError(f"{path} has no /template dataset")
        template = np.asarray(f["template"][...], dtype=np.float64)
        if template.ndim != 2 or template.shape[0] != 2:
            raise ValueError(f"Expected /template shape (2, N), got {template.shape}")
        plus, cross = template[0], template[1]
        meta = f["meta"] if "meta" in f else f
        m1 = _attr(meta, "m1")
        m2 = _attr(meta, "m2")
        a1 = _attr(meta, "a1")
        a2 = _attr(meta, "a2")
        approx = _attr(meta, "approx")

    amp = np.sqrt(plus**2 + cross**2)
    peak = int(np.argmax(amp))
    max_amp = float(amp[peak])
    if max_amp <= 0:
        raise ValueError("Template has zero amplitude")

    # Find the useful inspiral start from instantaneous frequency + amplitude.
    z = plus + 1j * cross
    phase = np.unwrap(np.angle(z))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    before = np.arange(plus.size) <= peak
    # Template polarization conventions can flip the sign of the complex phase.
    # Choose the sign from the strong pre-peak waveform rather than assuming one.
    strong_pre = before & (amp > max_amp * 1e-3) & np.isfinite(inst_freq)
    if np.any(strong_pre):
        med = float(np.nanmedian(inst_freq[strong_pre]))
        if med < 0:
            inst_freq = -inst_freq
    useful = before & (amp > max_amp * 1e-4) & (inst_freq > 20.0) & (inst_freq < fs / 2)
    idx = np.flatnonzero(useful)
    active_start = int(idx[0]) if idx.size else max(0, peak - int(1.5 * fs))

    # Ringdown end: first sustained decay below a small fraction of the peak,
    # with a conservative cap so the plot is not dominated by zero padding.
    cap = min(plus.size - 1, peak + int(0.25 * fs))
    post = amp[peak:cap + 1]
    below = np.flatnonzero(post < max_amp * 0.025)
    if below.size:
        ring_end = peak + int(below[0])
        ring_end = max(ring_end, peak + int(0.025 * fs))
    else:
        ring_end = cap

    return TemplateRecord(
        plus=plus,
        cross=cross,
        fs=float(fs),
        peak_index=peak,
        active_start_index=active_start,
        ringdown_end_index=int(ring_end),
        m1=None if m1 is None else float(m1),
        m2=None if m2 is None else float(m2),
        a1=None if a1 is None else float(a1),
        a2=None if a2 is None else float(a2),
        approximant=None if approx is None else str(approx),
    )


def instantaneous_frequency(template: TemplateRecord) -> np.ndarray:
    z = template.complex_template
    phase = np.unwrap(np.angle(z))
    f = np.gradient(phase) * template.fs / (2 * np.pi)
    amp = np.abs(z)
    strong = amp > np.max(amp) * 1e-3
    if np.any(strong):
        med = float(np.nanmedian(f[strong]))
        if med < 0:
            f = -f
    # Hide meaningless values where the zero-padded waveform amplitude vanishes.
    mask = amp > np.max(amp) * 1e-4
    out = np.full_like(f, np.nan, dtype=float)
    out[mask] = f[mask]
    return out
