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
    """
    Return all group/dataset paths in an HDF5 file.

    Use this first if the exact dataset hierarchy is unknown.
    """
    found: list[str] = []

    with h5py.File(path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            shape = getattr(obj, "shape", "")
            found.append(f"{kind:7s} /{name} {shape}")

        f.visititems(visitor)

    return found


def load_hdf5_dataset(
    path: str | Path,
    dataset_path: str,
    fs: float,
    *,
    start_time: float | None = None,
    detector: str | None = None,
) -> StrainRecord:
    """
    Generic loader when you know the exact dataset path and sample rate.
    """
    with h5py.File(path, "r") as f:
        strain = np.asarray(f[dataset_path][:], dtype=np.float64)

    if strain.ndim != 1:
        raise ValueError(f"Expected a 1-D strain dataset; got {strain.shape}")

    return StrainRecord(
        strain=strain,
        fs=float(fs),
        start_time=start_time,
        detector=detector,
    )


def load_common_gwosc_hdf5(path: str | Path) -> StrainRecord:
    """
    Convenience loader for the common GWOSC HDF5 layout in which strain is
    stored at /strain/Strain and Xspacing stores the sampling interval.

    If your downloaded file has a different hierarchy, call inspect_hdf5()
    and then use load_hdf5_dataset() instead.
    """
    with h5py.File(path, "r") as f:
        if "strain/Strain" not in f:
            raise KeyError(
                "Could not find /strain/Strain. "
                "Call inspect_hdf5(path) and use load_hdf5_dataset()."
            )

        ds = f["strain/Strain"]
        strain = np.asarray(ds[:], dtype=np.float64)

        if "Xspacing" in ds.attrs:
            dt = float(ds.attrs["Xspacing"])
            fs = 1.0 / dt
        else:
            # Fallback when duration metadata is available.
            duration = None
            if "meta" in f and "Duration" in f["meta"].attrs:
                duration = float(f["meta"].attrs["Duration"])
            if duration is None:
                raise KeyError(
                    "Could not infer sample rate from Xspacing or meta/Duration."
                )
            fs = strain.size / duration

        start_time = None
        detector = None
        if "meta" in f:
            meta = f["meta"].attrs
            if "GPSstart" in meta:
                start_time = float(meta["GPSstart"])
            if "Detector" in meta:
                raw = meta["Detector"]
                detector = raw.decode() if isinstance(raw, bytes) else str(raw)

    return StrainRecord(
        strain=strain,
        fs=float(fs),
        start_time=start_time,
        detector=detector,
    )


def slice_by_time(
    record: StrainRecord,
    start_s: float,
    end_s: float,
) -> StrainRecord:
    """
    Slice using relative seconds from the start of the record.
    """
    if not (0 <= start_s < end_s <= record.duration):
        raise ValueError(
            f"Need 0 <= start < end <= {record.duration:.6f} s."
        )

    i0 = int(round(start_s * record.fs))
    i1 = int(round(end_s * record.fs))
    new_start = (
        None if record.start_time is None
        else record.start_time + i0 / record.fs
    )

    return StrainRecord(
        strain=record.strain[i0:i1].copy(),
        fs=record.fs,
        start_time=new_start,
        detector=record.detector,
    )
