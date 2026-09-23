from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class EventSpec:
    key: str
    label: str
    kind: str  # real_event | quiet | synthetic
    source_event: str
    gps_event: float | None
    utc_event: str | None
    h1_file: Path | None
    l1_file: Path | None
    template_file: Path | None
    fband: tuple[float, float]
    notches_hz: tuple[float, ...]
    threshold: float
    expected_signal: bool
    quiet_window_s: tuple[float, float] | None = None
    display_half_width_s: float = 0.75
    sonify_half_width_s: float = 2.0
    # Published/reference event parameters. These are metadata, not inferred by GravityHunter.
    m1_source: float | None = None
    m2_source: float | None = None
    chirp_mass_source: float | None = None
    final_mass_source: float | None = None
    radiated_energy: float | None = None
    luminosity_distance_mpc: float | None = None
    sky_area_deg2: float | None = None
    network_snr: float | None = None
    release: str | None = None
    source_url: str | None = None


def _event_path(name: str, detector: str) -> Path:
    return DATA_ROOT / name / f"{detector}.hdf5"


def _template_path(name: str) -> Path:
    return DATA_ROOT / "templates" / f"{name}_4_template.hdf5"


EVENTS: dict[str, EventSpec] = {
    "GW150914": EventSpec(
        key="GW150914",
        label="GW150914 — first direct BBH detection",
        kind="real_event",
        source_event="GW150914",
        gps_event=1126259462.4,
        utc_event="2015-09-14 09:50:44 UTC",
        h1_file=_event_path("GW150914", "H1"),
        l1_file=_event_path("GW150914", "L1"),
        template_file=_template_path("GW150914"),
        fband=(43.0, 300.0),
        notches_hz=(60.0, 120.0, 180.0),
        threshold=5.5,
        expected_signal=True,
        display_half_width_s=0.35,
        sonify_half_width_s=2.0,
        m1_source=35.6,
        m2_source=30.6,
        chirp_mass_source=28.6,
        final_mass_source=63.1,
        radiated_energy=3.1,
        luminosity_distance_mpc=440.0,
        sky_area_deg2=182.0,
        network_snr=23.6,
        release="GWTC-1-confident / v3",
        source_url="https://gwosc.org/eventapi/html/event/GW150914/v3",
    ),
    "GW151226": EventSpec(
        key="GW151226",
        label="GW151226 — lower-mass, longer inspiral",
        kind="real_event",
        source_event="GW151226",
        gps_event=1135136350.6,
        utc_event="2015-12-26 03:38:52 UTC",
        h1_file=_event_path("GW151226", "H1"),
        l1_file=_event_path("GW151226", "L1"),
        template_file=_template_path("GW151226"),
        fband=(43.0, 800.0),
        notches_hz=(60.0, 120.0, 180.0),
        threshold=5.5,
        expected_signal=True,
        display_half_width_s=0.8,
        sonify_half_width_s=2.0,
        m1_source=13.7,
        m2_source=7.7,
        chirp_mass_source=8.9,
        final_mass_source=20.5,
        radiated_energy=1.0,
        luminosity_distance_mpc=450.0,
        sky_area_deg2=1033.0,
        network_snr=13.1,
        release="GWTC-1-confident / v2",
        source_url="https://gwosc.org/eventapi/html/GWTC-1-confident/GW151226/v2/",
    ),
    "GW170104": EventSpec(
        key="GW170104",
        label="GW170104 — O2 binary black-hole merger",
        kind="real_event",
        source_event="GW170104",
        gps_event=1167559936.6,
        utc_event="2017-01-04 10:11:58 UTC",
        h1_file=_event_path("GW170104", "H1"),
        l1_file=_event_path("GW170104", "L1"),
        template_file=_template_path("GW170104"),
        fband=(43.0, 800.0),
        notches_hz=(60.0, 120.0, 180.0),
        threshold=5.5,
        expected_signal=True,
        display_half_width_s=0.45,
        sonify_half_width_s=2.0,
        m1_source=30.8,
        m2_source=20.0,
        chirp_mass_source=21.4,
        final_mass_source=48.9,
        radiated_energy=2.2,
        luminosity_distance_mpc=990.0,
        sky_area_deg2=921.0,
        network_snr=13.0,
        release="GWTC-1-confident / v2",
        source_url="https://gwosc.org/eventapi/html/GWTC-1-confident/GW170104/v2/",
    ),
    # Negative examples reuse genuine detector data but select an off-source interval.
    "QUIET150914": EventSpec(
        key="QUIET150914",
        label="Quiet real data — before GW150914 (no merger in window)",
        kind="quiet",
        source_event="GW150914",
        gps_event=None,
        utc_event=None,
        h1_file=_event_path("GW150914", "H1"),
        l1_file=_event_path("GW150914", "L1"),
        template_file=_template_path("GW150914"),
        fband=(43.0, 300.0),
        notches_hz=(60.0, 120.0, 180.0),
        threshold=5.5,
        expected_signal=False,
        quiet_window_s=(2.0, 10.0),
        display_half_width_s=2.5,
        sonify_half_width_s=2.0,
        release="Off-source window from GW150914 local strain",
    ),
    "QUIET170104": EventSpec(
        key="QUIET170104",
        label="Quiet real data — after GW170104 (no merger in window)",
        kind="quiet",
        source_event="GW170104",
        gps_event=None,
        utc_event=None,
        h1_file=_event_path("GW170104", "H1"),
        l1_file=_event_path("GW170104", "L1"),
        template_file=_template_path("GW170104"),
        fband=(43.0, 800.0),
        notches_hz=(60.0, 120.0, 180.0),
        threshold=5.5,
        expected_signal=False,
        quiet_window_s=(22.0, 30.0),
        display_half_width_s=2.5,
        sonify_half_width_s=2.0,
        release="Off-source window from GW170104 local strain",
    ),
    "SYNTHETIC": EventSpec(
        key="SYNTHETIC",
        label="Synthetic validation chirp",
        kind="synthetic",
        source_event="Synthetic",
        gps_event=None,
        utc_event=None,
        h1_file=None,
        l1_file=None,
        template_file=None,
        fband=(25.0, 300.0),
        notches_hz=(60.0,),
        threshold=5.0,
        expected_signal=True,
    ),
}


def get_event_spec(key: str) -> EventSpec:
    try:
        return EVENTS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown GravityHunter case: {key}") from exc


def selectable_cases() -> list[EventSpec]:
    order = ["GW150914", "GW151226", "GW170104", "QUIET150914", "QUIET170104", "SYNTHETIC"]
    return [EVENTS[k] for k in order]


def missing_files(spec: EventSpec) -> list[Path]:
    if spec.kind == "synthetic":
        return []
    files = [spec.h1_file, spec.l1_file, spec.template_file]
    return [p for p in files if p is not None and not p.exists()]
