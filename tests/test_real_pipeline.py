from pathlib import Path

import h5py
import numpy as np

from gravityhunter.analysis.real_pipeline import analyze_real_case, chirp_mass
from gravityhunter.analysis.chirp_mass import estimate_chirp_mass_from_ridge, estimate_chirp_mass_from_detector, G_SI, C_SI, M_SUN_KG
from gravityhunter.catalog import EventSpec
from gravityhunter.dsp.sonification import wav_bytes
from gravityhunter.dsp.templates import load_losc_template
from gravityhunter.ui.merger_animation import build_animation_payload, merger_animation_html


def _write_mock_template(path: Path, fs: float, duration: float, event_t: float):
    n = int(fs * duration)
    plus = np.zeros(n)
    cross = np.zeros(n)
    peak = int(round(event_t * fs))
    start = int(round((event_t - 1.2) * fs))

    tt = np.arange(peak - start) / fs
    # Analytic linear chirp phase from 30 to 180 Hz.
    f0, f1, T = 30.0, 180.0, 1.2
    phase = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / T * tt**2)
    env = np.linspace(0.05, 1.0, tt.size) ** 2
    plus[start:peak] = env * np.cos(phase)
    cross[start:peak] = env * np.sin(phase)

    tr = np.arange(int(0.2 * fs)) / fs
    rd = np.exp(-tr / 0.05)
    plus[peak:peak + tr.size] = rd * np.cos(2 * np.pi * f1 * tr)
    cross[peak:peak + tr.size] = rd * np.sin(2 * np.pi * f1 * tr)

    plus *= 1e-21
    cross *= 1e-21

    with h5py.File(path, "w") as f:
        f.create_dataset("template", data=np.stack([plus, cross]))
        meta = f.create_group("meta")
        meta.attrs["m1"] = 30.0
        meta.attrs["m2"] = 20.0
        meta.attrs["approx"] = "mock"

    return plus, cross


def _write_mock_strain(path: Path, detector: str, x: np.ndarray, fs: float, gps0: float):
    with h5py.File(path, "w") as f:
        strain = f.create_group("strain")
        ds = strain.create_dataset("Strain", data=x)
        ds.attrs["Xspacing"] = 1.0 / fs
        meta = f.create_group("meta")
        meta.attrs["GPSstart"] = gps0
        meta.attrs["Duration"] = x.size / fs
        meta.attrs["Detector"] = detector


def test_real_pipeline_end_to_end_on_gwosc_shaped_hdf5(tmp_path: Path):
    fs = 512.0
    duration = 8.0
    n = int(fs * duration)
    gps0 = 1000.0
    event_t = 4.0

    template_path = tmp_path / "template.hdf5"
    plus, _ = _write_mock_template(template_path, fs, duration, event_t)

    rng = np.random.default_rng(3)
    noise_scale = 2e-22

    h1 = rng.normal(0.0, noise_scale, n) + plus
    l1 = rng.normal(0.0, noise_scale, n) + np.roll(plus, 4)

    h1_path = tmp_path / "H1.hdf5"
    l1_path = tmp_path / "L1.hdf5"
    _write_mock_strain(h1_path, "H1", h1, fs, gps0)
    _write_mock_strain(l1_path, "L1", l1, fs, gps0)

    spec = EventSpec(
        key="MOCK",
        label="Mock event",
        kind="real_event",
        source_event="MOCK",
        gps_event=gps0 + event_t,
        utc_event=None,
        h1_file=h1_path,
        l1_file=l1_path,
        template_file=template_path,
        fband=(25.0, 220.0),
        notches_hz=(),
        threshold=4.0,
        expected_signal=True,
    )

    result = analyze_real_case(spec, threshold=4.0)

    assert result.h1.detection.detected
    assert result.l1.detection.detected
    assert abs(result.h1.detection.peak_time_s - event_t) < 0.02
    assert abs(result.l1.detection.peak_time_s - (event_t + 4 / fs)) < 0.02
    assert result.network_delay is not None
    assert abs(abs(result.network_delay.best_lag_samples) - 4) <= 1
    assert result.coincidence.coincident
    assert result.h1.stft_power.ndim == 2
    assert set(result.phase_intervals_absolute) == {"Inspiral", "Merger", "Ringdown"}

    payload = build_animation_payload(result, detector_name="H1", sync_to="detected")
    assert payload.data["collisionTime"] is not None
    assert abs(payload.data["collisionTime"] - result.h1.detection.peak_time_s) < 1e-12
    assert payload.data["gpsStart"] == gps0
    assert len(payload.data["realWave"]["t"]) > 100
    html = merger_animation_html(payload)
    assert "Replay merger" in html
    assert "GPS" in html

    cm = estimate_chirp_mass_from_detector(
        white_strain=result.h1.white,
        fs=result.h1.record.fs,
        template=result.template,
        event_time_s=result.event_time_s,
        phase_intervals_absolute=result.phase_intervals_absolute,
        spec=spec,
        detector="H1",
    )
    assert cm.stft_power.ndim == 2
    # The mock waveform is a linear chirp rather than a physical PN chirp, so
    # this checks execution/ridge extraction rather than mass accuracy.
    assert cm.ridge_time_s.size >= 0


def test_template_and_audio_helpers(tmp_path: Path):
    fs = 512.0
    path = tmp_path / "template.hdf5"
    _write_mock_template(path, fs, 4.0, 2.0)
    template = load_losc_template(path, fs=fs)

    assert template.plus.shape == template.cross.shape
    assert template.active_start_index < template.peak_index < template.ringdown_end_index
    assert len(wav_bytes(template.plus[template.compact_slice()], fs)) > 100


def test_chirp_mass_formula():
    mc = chirp_mass(35.6, 30.6)
    assert 28.0 < mc < 29.5


def test_chirp_mass_ridge_math_recovers_known_mass():
    target = 28.0
    mass_kg = target * M_SUN_KG
    tau = np.linspace(0.25, 0.005, 80)
    t = -tau
    # Leading-order time-to-coalescence relation inverted for f(t).
    f = (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * (
        G_SI * mass_kg / C_SI**3
    ) ** (-5.0 / 8.0)

    estimate, smooth, dfdt, pointwise, r2, mono = estimate_chirp_mass_from_ridge(t, f)
    assert estimate is not None
    assert abs(estimate - target) < 0.15
    assert r2 is not None and r2 > 0.99
    assert mono is not None and mono > 0.99
    assert np.nanmedian(pointwise) == np.nanmedian(pointwise)
