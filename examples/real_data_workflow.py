"""
Skeleton for running the core pipeline on one real local HDF5 detector file.

You MUST first inspect your actual HDF5 hierarchy and confirm the dataset path
and metadata before relying on a loader.
"""

from __future__ import annotations

from gravityhunter.dsp.loader import (
    inspect_hdf5,
    load_common_gwosc_hdf5,
)
from gravityhunter.dsp.psd import welch_psd
from gravityhunter.dsp.filters import (
    apply_bandpass_zero_phase,
    apply_multiple_notches,
)
from gravityhunter.dsp.whitening import whiten
from gravityhunter.dsp.stft_tools import scipy_stft


def inspect_file(path: str) -> None:
    for line in inspect_hdf5(path):
        print(line)


def preprocess_detector(
    path: str,
    *,
    noise_slice: slice,
    notches: list[float],
    band: tuple[float, float],
):
    rec = load_common_gwosc_hdf5(path)
    x = rec.strain
    fs = rec.fs

    noise_ref = x[noise_slice]
    nperseg = min(4096, noise_ref.size)
    if nperseg < 16:
        raise ValueError("Noise reference region is too short.")

    f_psd, psd = welch_psd(
        noise_ref,
        fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    y = apply_multiple_notches(x, fs, notches, q=30.0)
    y, _ = apply_bandpass_zero_phase(
        y,
        fs,
        band[0],
        band[1],
        order=4,
    )

    yw, _, _, _ = whiten(
        y,
        fs,
        f_psd,
        psd,
        fmin=band[0],
        fmax=band[1],
        standardize=True,
    )

    f_stft, t_stft, Z = scipy_stft(
        yw,
        fs,
        nperseg=256,
        noverlap=192,
    )

    return {
        "record": rec,
        "psd_freq": f_psd,
        "psd": psd,
        "filtered": y,
        "whitened": yw,
        "stft_freq": f_stft,
        "stft_time": t_stft,
        "stft": Z,
    }


# Example usage after you have a local file:
#
# path = "data/GW150914_H1.hdf5"
# inspect_file(path)
#
# result = preprocess_detector(
#     path,
#     noise_slice=slice(0, 6 * 4096),   # replace after confirming fs/time region
#     notches=[60.0],                    # choose from your measured PSD
#     band=(25.0, 300.0),               # tune from your data / event
# )
#
# IMPORTANT:
# Do not copy these numerical preprocessing choices blindly into the final
# project. First inspect the actual PSD and event/data interval.
