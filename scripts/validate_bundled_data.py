"""Run GravityHunter's acceptance checks on the bundled real GWOSC records.

No network access is used.  This script is intended for final verification and
presentation setup, not for downloading data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from gravityhunter.analysis.chirp_mass import estimate_chirp_mass_from_case
from gravityhunter.analysis.real_pipeline import analyze_real_case
from gravityhunter.catalog import get_event_spec
from gravityhunter.dsp.psd import welch_psd

EVENTS = ("GW150914", "GW151226", "GW170104")
QUIET = ("QUIET150914", "QUIET170104")


def whiten_spread_db(result, detector: str) -> float:
    det = result.h1 if detector == "H1" else result.l1
    f, p = welch_psd(
        det.white,
        det.record.fs,
        nperseg=min(4096, det.white.size),
        noverlap=min(2048, max(0, min(4096, det.white.size) // 2)),
        average="median",
    )
    lo, hi = result.spec.fband
    mask = (f >= lo) & (f <= min(float(hi), 600.0))
    db = 10.0 * np.log10(np.maximum(p[mask], np.finfo(float).tiny))
    q10, q90 = np.percentile(db, [10, 90])
    return float(q90 - q10)


def main() -> None:
    report: dict[str, object] = {"events": {}, "quiet_controls": {}}

    for key in EVENTS:
        r = analyze_real_case(get_event_spec(key))
        event = {
            "reference_record_time_s": r.event_time_s,
            "H1": {
                "detected": r.h1.detection.detected,
                "peak_snr": r.h1.detection.peak_snr,
                "peak_time_s": r.h1.detection.peak_time_s,
                "peak_gps": None if r.h1.detection.peak_time_s is None else r.h1.record.start_time + r.h1.detection.peak_time_s,
                "whiten_psd_80pct_spread_db": whiten_spread_db(r, "H1"),
            },
            "L1": {
                "detected": r.l1.detection.detected,
                "peak_snr": r.l1.detection.peak_snr,
                "peak_time_s": r.l1.detection.peak_time_s,
                "peak_gps": None if r.l1.detection.peak_time_s is None else r.l1.record.start_time + r.l1.detection.peak_time_s,
                "whiten_psd_80pct_spread_db": whiten_spread_db(r, "L1"),
            },
            "coincident": r.coincidence.coincident,
            "candidate_time_difference_ms": None if r.coincidence.absolute_time_difference_s is None else 1000.0 * r.coincidence.absolute_time_difference_s,
            "cross_correlation_delay_ms": None if r.network_delay is None else 1000.0 * r.network_delay.delay_seconds,
        }
        cm = estimate_chirp_mass_from_case(r)
        event["experimental_chirp_mass"] = {
            "success": cm.success,
            "estimate_solar_mass": cm.estimated_mass_solar,
            "fit_r2": cm.fit_r2,
            "ridge_points": int(cm.ridge_time_s.size),
            "message": cm.message,
        }
        report["events"][key] = event

    for key in QUIET:
        r = analyze_real_case(get_event_spec(key))
        report["quiet_controls"][key] = {
            "H1_peak_snr": r.h1.detection.peak_snr,
            "L1_peak_snr": r.l1.detection.peak_snr,
            "H1_detected": r.h1.detection.detected,
            "L1_detected": r.l1.detection.detected,
            "coincident": r.coincidence.coincident,
        }

    out = Path("validation_results.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
