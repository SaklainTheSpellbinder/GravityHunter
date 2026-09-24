"""Regression checks against the real GWOSC files bundled with the project.

These are intentionally broad scientific acceptance ranges rather than attempts
at bit-for-bit reproduction of the historical GWOSC tutorial.  They protect the
important invariants: correct event timing, plausible SNR ordering/magnitude,
physically compatible H1/L1 delays, and rejection of the bundled off-source
controls.
"""
from pathlib import Path

import numpy as np
import pytest

from gravityhunter.analysis.real_pipeline import analyze_real_case
from gravityhunter.catalog import get_event_spec
from gravityhunter.dsp.psd import welch_psd


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _have_bundled_data() -> bool:
    required = [
        DATA_ROOT / "GW150914" / "H1.hdf5",
        DATA_ROOT / "GW150914" / "L1.hdf5",
        DATA_ROOT / "GW151226" / "H1.hdf5",
        DATA_ROOT / "GW151226" / "L1.hdf5",
        DATA_ROOT / "GW170104" / "H1.hdf5",
        DATA_ROOT / "GW170104" / "L1.hdf5",
        DATA_ROOT / "templates" / "GW150914_4_template.hdf5",
        DATA_ROOT / "templates" / "GW151226_4_template.hdf5",
        DATA_ROOT / "templates" / "GW170104_4_template.hdf5",
    ]
    return all(p.exists() for p in required)


pytestmark = pytest.mark.skipif(not _have_bundled_data(), reason="bundled GWOSC data not present")


@pytest.mark.parametrize(
    "key,h1_snr_range,l1_snr_range,h1_ref_offset_ms,l1_ref_offset_ms,delay_range_ms",
    [
        ("GW150914", (15.0, 22.0), (10.0, 16.0), (-2.0, 2.0), (-10.0, -4.0), (5.0, 9.0)),
        ("GW151226", (6.0, 10.0), (5.5, 9.0), (-6.0, 3.0), (-7.0, 3.0), (0.2, 2.0)),
        ("GW170104", (7.0, 11.0), (7.5, 12.0), (4.0, 13.0), (6.0, 15.0), (-5.0, -1.0)),
    ],
)
def test_real_event_recovery(key, h1_snr_range, l1_snr_range, h1_ref_offset_ms, l1_ref_offset_ms, delay_range_ms):
    result = analyze_real_case(get_event_spec(key))
    assert result.event_time_s is not None
    assert result.h1.detection.detected
    assert result.l1.detection.detected
    assert result.coincidence.coincident
    assert result.network_delay is not None

    for det, snr_range, offset_range in [
        (result.h1, h1_snr_range, h1_ref_offset_ms),
        (result.l1, l1_snr_range, l1_ref_offset_ms),
    ]:
        assert snr_range[0] <= det.detection.peak_snr <= snr_range[1]
        assert det.detection.peak_time_s is not None
        offset_ms = 1000.0 * (det.detection.peak_time_s - result.event_time_s)
        assert offset_range[0] <= offset_ms <= offset_range[1]
        assert np.all(np.isfinite(det.raw))
        assert np.all(np.isfinite(det.filtered))
        assert np.all(np.isfinite(det.white))
        assert np.isclose(np.std(det.white), 1.0, atol=1e-10)

    delay_ms = 1000.0 * result.network_delay.delay_seconds
    assert delay_range_ms[0] <= delay_ms <= delay_range_ms[1]


def test_bundled_quiet_controls_stay_below_default_threshold():
    for key in ("QUIET150914", "QUIET170104"):
        spec = get_event_spec(key)
        result = analyze_real_case(spec)
        assert not result.h1.detection.detected
        assert not result.l1.detection.detected
        assert not result.coincidence.coincident
        assert result.h1.detection.peak_snr < spec.threshold
        assert result.l1.detection.peak_snr < spec.threshold


def test_whitening_flattens_real_detector_noise_without_nan():
    # Check representative H1/L1 outputs for all three events.  We do not demand
    # a perfectly white PSD; real detector data and finite Welch estimates retain
    # some structure.  An 80%-central spread below 8 dB is a conservative guard.
    for key in ("GW150914", "GW151226", "GW170104"):
        result = analyze_real_case(get_event_spec(key))
        for det in (result.h1, result.l1):
            f, p = welch_psd(
                det.white,
                det.record.fs,
                nperseg=4096,
                noverlap=2048,
                average="median",
            )
            lo, hi = result.spec.fband
            mask = (f >= lo) & (f <= min(float(hi), 600.0))
            db = 10.0 * np.log10(np.maximum(p[mask], np.finfo(float).tiny))
            q10, q90 = np.percentile(db, [10, 90])
            assert q90 - q10 < 8.0
