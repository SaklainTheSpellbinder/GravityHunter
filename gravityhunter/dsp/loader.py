from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class StrainRecord:
    strain: np.ndarray
    fs: float
    start_time: float | None = None
    detector: str | None = None

    @property
    def n(self) -> int:
        return int(self.strain.size)

    @property
    def duration(self) -> float:
        return self.n / self.fs

    @property
    def time(self) -> np.ndarray:
        """Relative time axis, shape (N,)."""
        return np.arange(self.n, dtype=float) / self.fs

    @property
    def absolute_time(self) -> np.ndarray | None:
        if self.start_time is None:
            return None
        return self.start_time + self.time


def inspect_hdf5(path: str | Path) -> list[str]:
    """Return all group/dataset paths in an HDF5 file."""
    found: list[str] = []
    with h5py.File(path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            shape = getattr(obj, "shape", "")
            found.append(f"{kind:7s} /{name} {shape}")
        f.visititems(visitor)
    return found


def _scalar_value(group: h5py.Group, name: str) -> Any | None:
    """Read a scalar from either a child dataset or an attribute."""
    if name in group:
        value = group[name][()]
    elif name in group.attrs:
        value = group.attrs[name]
    else:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def _decode_text(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _validate_record(strain: np.ndarray, fs: float) -> None:
    if strain.ndim != 1:
        raise ValueError(f"Expected a 1-D strain array; got {strain.shape}")
    if strain.size < 2:
        raise ValueError("Strain array is empty or too short")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid sample rate: {fs}")
    if not np.all(np.isfinite(strain)):
        raise ValueError("Strain contains NaN/Inf samples")


def load_hdf5_dataset(
    path: str | Path,
    dataset_path: str,
    fs: float,
    *,
    start_time: float | None = None,
    detector: str | None = None,
) -> StrainRecord:
    """Generic loader when the exact dataset path/sample rate are known."""
    with h5py.File(path, "r") as f:
        strain = np.asarray(f[dataset_path][:], dtype=np.float64)
    _validate_record(strain, float(fs))
    return StrainRecord(strain, float(fs), start_time=start_time, detector=detector)


def load_common_gwosc_hdf5(path: str | Path) -> StrainRecord:
    """Load the standard GWOSC/LOSC event HDF5 layout.

    Supported metadata variants include:
      - /strain/Strain with Xspacing and optionally Xstart attributes
      - /meta/GPSstart, /meta/Duration, /meta/Detector scalar datasets
      - older /meta attributes with the same names
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "strain/Strain" not in f:
            raise KeyError(
                "Could not find /strain/Strain. "
                "Use inspect_hdf5(path) and load_hdf5_dataset() for a custom file."
            )

        ds = f["strain/Strain"]
        strain = np.asarray(ds[:], dtype=np.float64)

        if "Xspacing" in ds.attrs:
            dt = float(ds.attrs["Xspacing"])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Invalid Xspacing={dt} in {path.name}")
            fs = 1.0 / dt
        else:
            duration = None
            if "meta" in f:
                raw_duration = _scalar_value(f["meta"], "Duration")
                if raw_duration is not None:
                    duration = float(raw_duration)
            if duration is None or duration <= 0:
                raise KeyError("Could not infer sample rate from Xspacing or meta/Duration")
            fs = strain.size / duration

        start_time = None
        detector = None

        if "Xstart" in ds.attrs:
            raw_start = float(ds.attrs["Xstart"])
            if np.isfinite(raw_start):
                start_time = raw_start

        if "meta" in f:
            meta = f["meta"]
            if start_time is None:
                raw_start = _scalar_value(meta, "GPSstart")
                if raw_start is not None:
                    start_time = float(raw_start)
            detector = _decode_text(_scalar_value(meta, "Detector"))

    _validate_record(strain, float(fs))
    return StrainRecord(
        strain=strain,
        fs=float(fs),
        start_time=start_time,
        detector=detector,
    )


def slice_by_time(record: StrainRecord, start_s: float, end_s: float) -> StrainRecord:
    """Slice a record using seconds relative to its first sample."""
    if not (0 <= start_s < end_s <= record.duration):
        raise ValueError(f"Need 0 <= start < end <= {record.duration:.6f} s.")

    i0 = int(round(start_s * record.fs))
    i1 = int(round(end_s * record.fs))
    i0 = min(max(i0, 0), record.n)
    i1 = min(max(i1, i0 + 1), record.n)
    new_start = None if record.start_time is None else record.start_time + i0 / record.fs

    return StrainRecord(
        strain=record.strain[i0:i1].copy(),
        fs=record.fs,
        start_time=new_start,
        detector=record.detector,
    )
