from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from examples.synthetic_pipeline import build_demo
from gravityhunter.analysis.real_pipeline import analyze_real_case, chirp_mass
from gravityhunter.analysis.chirp_mass import estimate_chirp_mass_from_case
from gravityhunter.catalog import EventSpec, get_event_spec, missing_files, selectable_cases
from gravityhunter.dsp.detector import detect_best_candidate
from gravityhunter.dsp.fft_tools import rfft_spectrum
from gravityhunter.dsp.psd import asd_from_psd, welch_psd
from gravityhunter.dsp.sonification import wav_bytes
from gravityhunter.dsp.stft_tools import power_to_db
from gravityhunter.dsp.templates import instantaneous_frequency
from gravityhunter.ui.merger_animation import build_animation_payload, merger_animation_html


st.set_page_config(
    page_title="GravityHunter",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CREAM = "#F1E6C5"
CREAM_DIM = "#B8B09C"
NAVY = "#06111E"
NAVY_2 = "#0C1C2D"
PANEL = "#0D1D30"
TEXT = "#DDE5EC"
MUTED = "#95A8BA"
BLUE = "#6F9FC6"
GOLD = "#D3B76B"
PURPLE = "#9A8FC1"
GREEN = "#86C8AA"
RED = "#D98F8F"
GRID = "rgba(210,220,230,0.09)"
TRACE_COLORS = [BLUE, CREAM, GREEN, PURPLE, GOLD]
PHASE_COLORS = {
    "Inspiral": "rgba(111,159,198,0.12)",
    "Merger": "rgba(211,183,107,0.17)",
    "Ringdown": "rgba(154,143,193,0.13)",
}

# Display settings are assigned from the sidebar before any page is rendered.
DISPLAY_MODE = "Auto focus"
ROBUST_AMPLITUDE = True
ADAPTIVE_CONTRAST = True

GLOBAL_CSS = f"""
<style>
:root {{ --cream:{CREAM}; --navy:{NAVY}; --panel:{PANEL}; --text:{TEXT}; --muted:{MUTED}; }}
html, body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.stApp {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--text);
  background-color: {NAVY};
  background-image:
    radial-gradient(circle at 24px 28px, rgba(241,230,197,.34) 0 .85px, transparent 1.25px),
    radial-gradient(circle at 96px 92px, rgba(241,230,197,.18) 0 .65px, transparent 1.05px),
    linear-gradient(158deg, #06111E 0%, #081827 55%, #040B13 100%);
  background-size: 190px 190px, 260px 260px, auto;
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: rgba(6,17,30,.82); }}
[data-testid="stSidebar"] {{
  background: rgba(8,24,40,.98);
  border-right: 1px solid rgba(221,229,236,.09);
}}
[data-testid="stSidebar"] * {{ color: var(--text); }}
/* Streamlit uses Material Symbols ligatures for sidebar/expander icons.
   Never let the application font override that icon font, otherwise names
   such as keyboard_double_arrow_left are rendered as literal text. */
span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] {{
  font-family: "Material Symbols Rounded" !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-size: inherit;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  -webkit-font-feature-settings: "liga" !important;
  font-feature-settings: "liga" !important;
  -webkit-font-smoothing: antialiased;
}}
.block-container {{ max-width: 1480px; padding-top: 1.35rem; padding-bottom: 3.5rem; }}
h1,h2,h3,h4 {{ color:var(--cream)!important; letter-spacing:-.018em; }}
p,li,label {{ color:var(--text); }}
.gh-hero {{
  background: linear-gradient(132deg, rgba(13,38,62,.91), rgba(8,24,40,.90));
  border: 1px solid rgba(221,229,236,.10);
  box-shadow: 0 12px 34px rgba(0,0,0,.20);
  border-radius: 18px; padding: 22px 25px 20px; margin-bottom:16px;
}}
.gh-kicker {{ color:{CREAM_DIM}; text-transform:uppercase; letter-spacing:.14em; font-size:.70rem; font-weight:700; }}
.gh-title {{ color:{CREAM}; font-size:2.02rem; line-height:1.08; font-weight:730; margin:.32rem 0 .48rem; }}
.gh-sub {{ color:#C3D0DB; max-width:1080px; font-size:.96rem; line-height:1.55; }}
.gh-card {{
  background:rgba(13,29,48,.84); border:1px solid rgba(221,229,236,.10);
  border-radius:14px; padding:14px 16px; min-height:104px;
  box-shadow:0 7px 18px rgba(0,0,0,.13);
}}
.gh-card .value {{ color:{CREAM}; font-size:1.36rem; font-weight:720; font-variant-numeric:tabular-nums; }}
.gh-card .label {{ color:#91A6B9; font-size:.72rem; text-transform:uppercase; letter-spacing:.075em; }}
.gh-card .note {{ color:#C0CCD6; font-size:.80rem; margin-top:.32rem; line-height:1.35; }}
.gh-status {{ border-radius:16px; padding:17px 20px; border:1px solid rgba(221,229,236,.10); margin:.3rem 0 .95rem; }}
.gh-status.good {{ background:linear-gradient(120deg, rgba(38,91,74,.27),rgba(13,29,48,.88)); }}
.gh-status.quiet {{ background:linear-gradient(120deg,rgba(42,61,82,.40),rgba(13,29,48,.88)); }}
.gh-status.bad {{ background:linear-gradient(120deg,rgba(100,48,48,.28),rgba(13,29,48,.88)); }}
.gh-status .big {{ font-size:1.30rem; font-weight:760; color:{CREAM}; }}
.gh-status .small {{ color:#C2CFDA; font-size:.86rem; line-height:1.45; }}
.gh-phase {{ border-radius:14px; padding:12px 14px; background:rgba(13,29,48,.76); border:1px solid rgba(221,229,236,.09); min-height:108px; }}
.gh-phase .phase-name {{ font-size:1rem; font-weight:720; color:{CREAM}; }}
.gh-phase .phase-note {{ color:#BBC9D4; font-size:.82rem; margin-top:.28rem; }}
.gh-pill {{ display:inline-block; padding:5px 9px; margin:3px 4px 3px 0; border-radius:999px; border:1px solid rgba(221,229,236,.10); background:rgba(111,159,198,.07); color:#D8E1E9; font-size:.75rem; }}
.gh-small {{ color:#98ABBC; font-size:.80rem; }}
.gh-display-note {{ color:#8FA3B5; font-size:.76rem; margin-top:-.25rem; margin-bottom:.55rem; }}
[data-testid="stMetric"] {{ background:rgba(13,29,48,.80); border:1px solid rgba(221,229,236,.09); padding:10px 12px; border-radius:13px; }}
[data-testid="stMetricLabel"] {{ color:#90A5B8; }}
[data-testid="stMetricValue"] {{ color:{CREAM}; font-variant-numeric:tabular-nums; }}
div[data-testid="stExpander"] {{ background:rgba(12,28,46,.72); border-color:rgba(221,229,236,.09); border-radius:12px; }}
hr {{ border-color:rgba(221,229,236,.09); }}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def hero(section: str, title: str, subtitle: str):
    st.markdown(
        f"""<div class='gh-hero'><div class='gh-kicker'>✦ {section}</div><div class='gh-title'>{title}</div><div class='gh-sub'>{subtitle}</div></div>""",
        unsafe_allow_html=True,
    )


def cards(items):
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(
                f"""<div class='gh-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>""",
                unsafe_allow_html=True,
            )


def status_banner(title: str, note: str, kind: str = "good"):
    st.markdown(
        f"<div class='gh-status {kind}'><div class='big'>{title}</div><div class='small'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, *, height: int = 390, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,17,30,.72)",
        font=dict(
            color=TEXT,
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            size=12,
        ),
        title_font=dict(color=CREAM, size=16),
        margin=dict(l=42, r=22, t=56 if title else 28, b=42),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=NAVY_2, font_color=CREAM),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=CREAM_DIM))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=CREAM_DIM))
    return fig


def _robust_linear_range(series, percentile: float = 99.5) -> tuple[float, float] | None:
    arrays = []
    for y in series:
        a = np.asarray(y, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            arrays.append(a)
    if not arrays:
        return None
    v = np.concatenate(arrays)
    if v.size < 4:
        return None
    lo = float(np.nanpercentile(v, 100.0 - percentile))
    hi = float(np.nanpercentile(v, percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    # Strain-like signals are naturally centered around zero; symmetric limits
    # keep positive/negative excursions visually comparable.
    if lo < 0 < hi:
        lim = max(abs(lo), abs(hi)) * 1.08
        return -lim, lim
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def line_figure(x, ys, labels, *, x_label="Time (s)", y_label="Amplitude", title=None, height=390):
    x_arr = np.asarray(x)
    y_arrs = [np.asarray(y) for y in ys]
    if x_arr.size > 30000:
        idx = np.linspace(0, x_arr.size - 1, 30000).astype(int)
        plot_x = x_arr[idx]
        plot_ys = [y[idx] for y in y_arrs]
    else:
        plot_x = x_arr
        plot_ys = y_arrs

    fig = go.Figure()
    for i, (y, label) in enumerate(zip(plot_ys, labels)):
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=y,
                mode="lines",
                name=label,
                line=dict(width=1.25, color=TRACE_COLORS[i % len(TRACE_COLORS)]),
            )
        )
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    if ROBUST_AMPLITUDE:
        yrange = _robust_linear_range(y_arrs)
        if yrange is not None:
            fig.update_yaxes(range=list(yrange))
    return style_figure(fig, height=height, title=title)


def display_mask(t: np.ndarray, center: float | None, half_width: float) -> np.ndarray:
    """Return a display-only time mask. DSP calculations always use full data."""
    t = np.asarray(t, dtype=float)
    if DISPLAY_MODE == "Full record" or center is None:
        return np.ones(t.shape, dtype=bool)
    lo = max(float(t[0]), float(center) - float(half_width))
    hi = min(float(t[-1]), float(center) + float(half_width))
    return (t >= lo) & (t <= hi)


def spectrogram_limits(db: np.ndarray, mask: np.ndarray | None = None, floor_db: float = -75.0) -> tuple[float, float]:
    """Display-only contrast limits; does not alter STFT power or inference."""
    if not ADAPTIVE_CONTRAST:
        return floor_db, 0.0
    vals = np.asarray(db if mask is None else db[mask], dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 32:
        return floor_db, 0.0
    lo = max(floor_db, float(np.nanpercentile(vals, 5.0)))
    hi = min(0.0, float(np.nanpercentile(vals, 99.7)))
    if hi - lo < 20.0:
        lo = max(floor_db, hi - 20.0)
    return lo, hi


def display_note() -> None:
    st.markdown(
        "<div class='gh-display-note'>Display framing and contrast affect visualization only; all DSP calculations use the unchanged full-resolution arrays.</div>",
        unsafe_allow_html=True,
    )


def add_phase_regions(fig: go.Figure, phases: dict[str, tuple[float, float]], *, relative_to: float = 0.0):
    for name, (a, b) in phases.items():
        fig.add_vrect(x0=a - relative_to, x1=b - relative_to, fillcolor=PHASE_COLORS[name], opacity=1, line_width=0, annotation_text=name, annotation_position="top left")


def mass_sankey(spec: EventSpec) -> go.Figure:
    m1 = float(spec.m1_source or 0)
    m2 = float(spec.m2_source or 0)
    mf = float(spec.final_mass_source or 0)
    er = float(spec.radiated_energy or max(m1 + m2 - mf, 0))
    total = max(m1 + m2, 1e-9)
    # Split both inputs proportionally into remnant and radiated energy.
    frac_f = min(max(mf / total, 0), 1)
    links_source = [0, 0, 1, 1]
    links_target = [2, 3, 2, 3]
    values = [m1 * frac_f, m1 * (1-frac_f), m2 * frac_f, m2 * (1-frac_f)]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[f"BH 1\n{m1:.1f} M☉", f"BH 2\n{m2:.1f} M☉", f"Remnant\n{mf:.1f} M☉", f"GW energy\n≈{er:.1f} M☉c²"], pad=20, thickness=20),
        link=dict(source=links_source, target=links_target, value=values),
    ))
    return style_figure(fig, height=360, title="Mass-energy story of the merger")


@st.cache_data(show_spinner=False)
def synthetic_data(seed: int):
    return build_demo(seed=seed)


@st.cache_data(show_spinner="Running full real-data DSP pipeline…")
def real_data(case_key: str, threshold: float):
    return analyze_real_case(get_event_spec(case_key), threshold=threshold)


@st.cache_data(show_spinner="Estimating chirp mass from the H1/L1 inspiral ridge…")
def chirp_mass_data(case_key: str, threshold: float):
    result = real_data(case_key, threshold)
    if result.event_time_s is None or not result.spec.expected_signal:
        return None
    return estimate_chirp_mass_from_case(result)


def detection_recovered(detection, event_time_s: float | None, tol_s: float = 0.25) -> bool:
    """Known-event validation only: did the record-wide candidate land near catalog time?"""
    return bool(
        detection.detected
        and detection.peak_time_s is not None
        and event_time_s is not None
        and abs(detection.peak_time_s - event_time_s) <= tol_s
    )


# ───────────────────────────── Sidebar ─────────────────────────────
st.sidebar.markdown("## ✦ GravityHunter")
st.sidebar.caption("GWOSC Open-Data Analysis Console")

cases = selectable_cases()
labels = [c.label for c in cases]
label_to_key = {c.label: c.key for c in cases}
selected_label = st.sidebar.selectbox("Data set", labels, index=0)
spec = get_event_spec(label_to_key[selected_label])

page = st.sidebar.radio(
    "Analysis view",
    [
        "Overview",
        "Waveform Morphology",
        "Event Timeline",
        "Detector Strain",
        "Spectrum & Noise",
        "Signal Conditioning",
        "Time–Frequency Analysis",
        "Event Detection",
        "Detector Coincidence",
        "Source Parameters",
        "Inspiral Diagnostics",
        "Audio Reconstruction",
        "Validation",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
detector_name = st.sidebar.radio("Detector channel", ["H1", "L1"], horizontal=True)
with st.sidebar.expander("View settings", expanded=False):
    DISPLAY_MODE = st.radio(
        "Time-domain framing",
        ["Auto focus", "Full record"],
        horizontal=True,
        help="Auto focus frames the selected event/candidate for readability. It never changes the underlying signal-processing calculation.",
    )
    ROBUST_AMPLITUDE = st.checkbox(
        "Robust amplitude scale",
        value=True,
        help="Uses percentile-based y-axis limits to prevent isolated glitches from flattening the visible waveform. Data are not clipped or modified.",
    )
    ADAPTIVE_CONTRAST = st.checkbox(
        "Adaptive spectrogram contrast",
        value=True,
        help="Uses robust dB display limits so weak time-frequency structure remains visible. STFT values are unchanged.",
    )
    st.caption("Visualization controls only — DSP outputs are unchanged.")

with st.sidebar.expander("Detection settings", expanded=False):
    threshold = st.slider(
        "Candidate SNR threshold",
        2.0,
        15.0,
        float(spec.threshold),
        0.25,
        key=f"threshold_{spec.key}",
        help="Local GravityHunter candidate threshold. It is not a published astrophysical significance threshold.",
    )
seed = 7
if spec.kind == "synthetic":
    seed = st.sidebar.number_input("Synthetic seed", min_value=0, max_value=9999, value=7, step=1)

with st.sidebar.expander("Data provenance", expanded=False):
    st.write("Real cases are loaded from the bundled local GWOSC HDF5 files under `./data/...`. No network access is required.")

if spec.kind != "synthetic":
    missing = missing_files(spec)
    if missing:
        st.sidebar.error("Local files are missing")
        for p in missing:
            st.sidebar.caption(str(p.relative_to(p.parents[2])))
        st.sidebar.caption("Restore the corresponding H1/L1/template files under ./data from the project bundle.")

# Load selected source.
base = None
real = None
if spec.kind == "synthetic":
    base = synthetic_data(int(seed))
else:
    missing = missing_files(spec)
    if not missing:
        try:
            real = real_data(spec.key, float(threshold))
        except Exception as exc:
            st.error(f"Could not run the real-data pipeline: {exc}")
            st.exception(exc)
            st.stop()

if spec.kind != "synthetic" and real is None:
    hero("DATA AVAILABILITY", spec.label, "Required bundled H1/L1 strain or reference-waveform files are missing from ./data. Restore the project data folder before running this case.")
    st.stop()


# Common real helpers.
if real is not None:
    det = real.h1 if detector_name == "H1" else real.l1
    fs = det.record.fs
    time = det.record.time
    event_t = real.event_time_s
else:
    fs = float(base["fs"])
    time = base["time"]
    event_t = float(base["injection_start_s"])


# ───────────────────────────── Mission Control ─────────────────────────────
if page == "Overview":
    if real is not None:
        hero("OBSERVATION SUMMARY", spec.label, "Local GWOSC H1/L1 strain processed through the same calibrated analysis chain used across all views. Off-source selections are processed identically as negative controls.")
        h1d, l1d = real.h1.detection, real.l1.detection
        if spec.expected_signal:
            recovered_h1 = detection_recovered(h1d, event_t)
            recovered_l1 = detection_recovered(l1d, event_t)
            if recovered_h1 and recovered_l1:
                status_banner("✦ REAL MERGER RECOVERED IN H1 + L1", "The strongest local template matches in both detectors are consistent with the catalog coalescence time within the configured validation tolerance.", "good")
            else:
                status_banner(
                    "CATALOG EVENT PRESENT — GRAVITYHUNTER HAS NOT RECOVERED BOTH DETECTORS",
                    "The file really contains this merger, but the record-wide matched-filter candidate in H1 and/or L1 is not within ±250 ms of the catalog coalescence time at the current threshold. This is a detector-result warning, not a statement that the astrophysical event did not happen.",
                    "bad",
                )
        else:
            if not h1d.detected and not l1d.detected:
                status_banner("✓ QUIET WINDOW: NO EVENT CANDIDATE", "This is genuine detector strain away from the merger. The same pipeline runs, but the selected threshold produces no candidate in either detector.", "quiet")
            else:
                status_banner("QUIET WINDOW PRODUCED A CANDIDATE", "The configured threshold produced a candidate in off-source data. This is classified as a false alarm and should be treated as detector noise, not an astrophysical event.", "bad")

        cards([
            ("Source", "REAL GWOSC", spec.release or "local"),
            ("Sample rate", f"{fs:.0f} Hz", f"{det.record.n:,} selected samples"),
            ("H1 peak SNR", f"{h1d.peak_snr:.2f}", f"t={h1d.peak_time_s:.3f}s" if h1d.peak_time_s is not None else "no candidate"),
            ("L1 peak SNR", f"{l1d.peak_snr:.2f}", f"t={l1d.peak_time_s:.3f}s" if l1d.peak_time_s is not None else "no candidate"),
        ])
        st.markdown("### Pipeline")
        st.markdown("<span class='gh-pill'>local HDF5</span><span class='gh-pill'>FFT</span><span class='gh-pill'>Welch PSD</span><span class='gh-pill'>notch + band-pass</span><span class='gh-pill'>whitening</span><span class='gh-pill'>STFT</span><span class='gh-pill'>two-quadrature matched filter</span><span class='gh-pill'>SNR</span><span class='gh-pill'>H1/L1 coincidence</span><span class='gh-pill'>inspiral diagnostics</span><span class='gh-pill'>synchronized merger timeline</span>", unsafe_allow_html=True)

        mask = display_mask(
            time,
            event_t if spec.expected_signal else None,
            max(spec.display_half_width_s, 0.35),
        )
        fig = line_figure(time[mask], [det.raw[mask], det.filtered[mask]], ["Raw strain", "Filtered"], title=f"{detector_name} selected region", y_label="Strain / filtered strain")
        if event_t is not None:
            fig.add_vline(x=event_t, line_dash="dash", line_color=GOLD, annotation_text="GWOSC reference")
        st.plotly_chart(fig, use_container_width=True)
        display_note()
    else:
        hero("CONTROLLED VALIDATION", "Synthetic signal recovery", "A deterministic injection case for regression testing of filtering, whitening, matched filtering and event-time recovery.")
        d = detect_best_candidate(base["snr"], base["lags"], fs, threshold=threshold, prominence=1.0)
        cards([
            ("Source", "SYNTHETIC", "controlled chirp injection"),
            ("Peak SNR", f"{d.peak_snr:.2f}", "matched-filter score"),
            ("Candidate", f"{d.peak_time_s:.3f} s" if d.peak_time_s is not None else "—", "true injection 8.000 s"),
        ])


# ───────────────────────────── Waveform Morphology ─────────────────────────────
elif page == "Waveform Morphology":
    if real is None:
        hero("WAVEFORM MORPHOLOGY", "Synthetic chirp morphology", "The synthetic signal validates the chirp-processing path but is not assigned compact-binary inspiral, merger and ringdown phases.")
        mask = (time >= 6.5) & (time <= 10.0)
        fig = line_figure(time[mask], [base["white"][mask]], ["Whitened synthetic data"], title="Synthetic event neighborhood")
        fig.add_vline(x=8.0, line_color=GOLD, line_dash="dash", annotation_text="injection")
        st.plotly_chart(fig, use_container_width=True)
    elif not spec.expected_signal:
        hero("OFF-SOURCE CONTROL", "Quiet detector interval", "Genuine H1/L1 strain from an interval without a catalog merger. No compact-binary morphology is imposed on the data.")
        status_banner("NO MERGER EXPECTED IN THIS WINDOW", "The identical conditioning and detection pipeline is applied to this off-source interval.", "quiet")
        d = real.h1 if detector_name == "H1" else real.l1
        fig = line_figure(d.record.time, [d.white], [f"{detector_name} whitened"], title="Whitened quiet strain", y_label="Standardized amplitude", height=420)
        st.plotly_chart(fig, use_container_width=True)
        fm = d.stft_freq <= min(spec.fband[1], 900)
        quiet_db = power_to_db(d.stft_power, floor_db=-70)
        zlo, zhi = spectrogram_limits(quiet_db[fm], floor_db=-70)
        heat = go.Figure(go.Heatmap(x=d.stft_time, y=d.stft_freq[fm], z=quiet_db[fm], colorscale="Cividis", zmin=zlo, zmax=zhi, colorbar=dict(title="dB rel.")))
        heat.update_xaxes(title="Time (s)"); heat.update_yaxes(title="Frequency (Hz)")
        st.plotly_chart(style_figure(heat, height=480, title="Quiet spectrogram — no rising merger track expected"), use_container_width=True)
    else:
        hero("WAVEFORM MORPHOLOGY", f"{spec.key}: inspiral → merger → ringdown", "Real conditioned detector strain is shown together with the configured public reference waveform. Phase bands are approximate morphology regions around the reference coalescence peak.")
        h1d, l1d = real.h1.detection, real.l1.detection
        recovered = detection_recovered(h1d, event_t) and detection_recovered(l1d, event_t)
        status_banner(
            "✦ GRAVITYHUNTER + REFERENCE AGREE" if recovered else "GWOSC REFERENCE VISUALIZATION — NOT A GRAVITYHUNTER DETECTION",
            f"The three-stage bands come from the public reference waveform around the GWOSC reference coalescence t ≈ {event_t:.3f} s. H1 local peak={h1d.peak_snr:.2f}, L1 local peak={l1d.peak_snr:.2f}.",
            "good" if recovered else "quiet",
        )

        phase_cols = st.columns(3)
        phase_text = {
            "Inspiral": "The two black holes orbit faster; frequency and amplitude rise — the chirp.",
            "Merger": "Peak-amplitude coalescence: the two horizons form one remnant black hole.",
            "Ringdown": "The remnant settles by emitting a rapidly decaying oscillation.",
        }
        for col, name in zip(phase_cols, ["Inspiral", "Merger", "Ringdown"]):
            a, b = real.phase_intervals_absolute[name]
            with col:
                st.markdown(f"<div class='gh-phase'><div class='phase-name'>{name}</div><div class='phase-note'>{phase_text[name]}</div><div class='gh-small'>{(a-event_t)*1000:.1f} to {(b-event_t)*1000:.1f} ms relative to peak</div></div>", unsafe_allow_html=True)

        # Reference morphology around the event.
        tr = real.template
        tt = tr.time_from_reference
        stage_min = min(v[0] - event_t for v in real.phase_intervals_absolute.values()) - 0.03
        stage_max = max(v[1] - event_t for v in real.phase_intervals_absolute.values()) + 0.03
        tm = (tt >= stage_min) & (tt <= stage_max)
        ref = tr.plus / max(np.max(np.abs(tr.plus)), 1e-30)
        fig_ref = go.Figure(go.Scatter(x=tt[tm] * 1000, y=ref[tm], mode="lines", name="reference waveform", line=dict(width=1.8)))
        rel_phases = {k: (a-event_t, b-event_t) for k,(a,b) in real.phase_intervals_absolute.items()}
        for name,(a,b) in rel_phases.items():
            fig_ref.add_vrect(x0=a*1000, x1=b*1000, fillcolor=PHASE_COLORS[name], opacity=1, line_width=0, annotation_text=name, annotation_position="top left")
        fig_ref.add_vline(x=0, line_dash="dash", line_color=GOLD, annotation_text="peak / merger")
        fig_ref.update_xaxes(title="Time relative to merger (ms)"); fig_ref.update_yaxes(title="Normalized template strain")
        st.plotly_chart(style_figure(fig_ref, height=390, title="Reference waveform: the three stages clearly visible"), use_container_width=True)

        # Real detector view around the same event.
        d = real.h1 if detector_name == "H1" else real.l1
        mask = display_mask(d.record.time, event_t, max(spec.display_half_width_s, 0.35))
        fig_real = go.Figure(go.Scatter(x=d.record.time[mask]-event_t, y=d.white[mask], mode="lines", name=f"{detector_name} whitened", line=dict(width=1.0, color=BLUE)))
        rel_abs = {k:(a-event_t,b-event_t) for k,(a,b) in real.phase_intervals_absolute.items()}
        add_phase_regions(fig_real, rel_abs)
        fig_real.add_vline(x=0, line_dash="dash", line_color=GOLD)
        fig_real.update_xaxes(title="Time relative to catalog merger (s)"); fig_real.update_yaxes(title="Whitened strain")
        st.plotly_chart(style_figure(fig_real, height=410, title=f"Real {detector_name} data in the collision neighborhood"), use_container_width=True)

        # Spectrogram + reference template ridge.
        fmask = d.stft_freq <= min(spec.fband[1], 900)
        tmask = display_mask(d.stft_time, event_t, max(spec.display_half_width_s, 0.35))
        db = power_to_db(d.stft_power, floor_db=-75)
        shown_db = db[fmask][:, tmask]
        zlo, zhi = spectrogram_limits(shown_db, floor_db=-75)
        heat = go.Figure(go.Heatmap(x=d.stft_time[tmask]-event_t, y=d.stft_freq[fmask], z=shown_db, colorscale="Cividis", zmin=zlo, zmax=zhi, colorbar=dict(title="dB rel.")))
        fi = instantaneous_frequency(tr)
        ridge_mask = np.isfinite(fi) & (tt >= stage_min) & (tt <= stage_max) & (fi >= spec.fband[0]) & (fi <= min(spec.fband[1], 900))
        heat.add_trace(go.Scatter(x=tt[ridge_mask], y=fi[ridge_mask], mode="lines", name="reference chirp ridge", line=dict(color=CREAM, width=2.2)))
        for name,(a,b) in rel_abs.items():
            heat.add_vrect(x0=a, x1=b, fillcolor=PHASE_COLORS[name], opacity=1, line_width=0)
        heat.add_vline(x=0, line_dash="dash", line_color=GOLD)
        heat.update_xaxes(title="Time relative to merger (s)"); heat.update_yaxes(title="Frequency (Hz)")
        st.plotly_chart(style_figure(heat, height=540, title="Real spectrogram + expected chirp track"), use_container_width=True)
        st.caption("The cream ridge comes from the public reference template and is overlaid to make the expected inspiral trajectory explicit. The heatmap itself comes from the real detector data.")
        display_note()


# ───────────────────────────── Event Timeline ─────────────────────────────
elif page == "Event Timeline":
    hero(
        "EVENT SYNCHRONIZATION",
        "32-second merger timeline",
        "A data-synchronized schematic of the selected detector record. Record/GPS timing, conditioned strain and matched-filter score are analysis outputs; orbital geometry is driven by reference-waveform phase and frequency and is not a numerical GR simulation.",
    )
    if real is None:
        st.info("Select a real event or quiet real-data case for the synchronized timeline. The synthetic validation chirp does not have a physically meaningful black-hole mass/orbit model.")
    else:
        if spec.expected_signal:
            d = real.h1 if detector_name == "H1" else real.l1
            recovered_here = detection_recovered(d.detection, event_t)
            if recovered_here:
                sync_choice = st.radio(
                    "Collision synchronization",
                    ["GravityHunter detected time", "GWOSC reference time"],
                    horizontal=True,
                    help="Detected time uses the locally recovered matched-filter candidate; GWOSC reference time is the external validation timestamp.",
                )
                sync_to = "detected" if sync_choice.startswith("GravityHunter") else "catalog"
            else:
                sync_to = "catalog"
                st.warning(
                    f"{detector_name} has not produced a validated GravityHunter candidate near the GWOSC reference event time, so detected-time synchronization is disabled. The animation below is explicitly synchronized to the GWOSC reference."
                )

            detected_gps = None if d.record.start_time is None or d.detection.peak_time_s is None else d.record.start_time + d.detection.peak_time_s
            chosen_position = d.detection.peak_time_s if sync_to == "detected" and d.detection.peak_time_s is not None else event_t
            cards([
                ("Record GPS start", f"{d.record.start_time:.1f}" if d.record.start_time is not None else "—", "first sample of local detector record"),
                ("GravityHunter peak GPS", f"{detected_gps:.4f}" if detected_gps is not None else "—", f"{detector_name} matched-filter candidate"),
                ("Reference GPS", f"{spec.gps_event:.4f}" if spec.gps_event is not None else "—", spec.utc_event or "external reference"),
                ("Animation position", f"{chosen_position:.4f} s" if chosen_position is not None else "—", "selected synchronization inside record"),
            ])
            st.caption("Reference GPS is the GWOSC event timestamp used only for validation/synchronization. GravityHunter peak GPS is recovered from the local strain by the matched filter.")
        else:
            sync_to = "catalog"
            status_banner(
                "NEGATIVE CONTROL — NO COLLISION TRIGGER",
                "The same 32-second-style timeline/strain playback is available, but quiet real data deliberately does not spawn a black-hole merger animation.",
                "quiet",
            )

        payload = build_animation_payload(real, detector_name=detector_name, sync_to=sync_to)
        components.html(merger_animation_html(payload), height=950, scrolling=False)
        if spec.expected_signal:
            st.info("Slow-motion replay stretches only the visual playback rate; the record and GPS clocks continue to display physical event time.")
        st.caption("Scientific boundary: the curvature fabric and spheres are schematic. Event timing, waveform phase/frequency, detector strain and matched-filter SNR are data-driven; the page does not claim to solve Einstein's field equations in real time.")


# ───────────────────────────── Detector Strain ─────────────────────────────
elif page == "Detector Strain":
    hero("DETECTOR DATA", "Strain and Fourier spectrum", "Calibrated detector strain in the time domain with its one-sided Fourier magnitude spectrum.")
    if real is not None:
        d = real.h1 if detector_name == "H1" else real.l1
        tmask = display_mask(d.record.time, event_t if spec.expected_signal else None, max(spec.display_half_width_s, 0.5))
        shown_t = d.record.time[tmask]
        shown_y = d.raw[tmask]
        stride = max(1, shown_t.size // 30000)
        fig = line_figure(shown_t[::stride], [shown_y[::stride]], [f"{detector_name} raw strain"], title="Real calibrated strain", y_label="Strain")
        if event_t is not None:
            fig.add_vline(x=event_t, line_dash="dash", line_color=GOLD, annotation_text="GWOSC reference")
        st.plotly_chart(fig, use_container_width=True)
        f, X = rfft_spectrum(d.raw, fs)
        f2, X2 = rfft_spectrum(d.filtered, fs)
        specfig = go.Figure()
        specfig.add_trace(go.Scatter(x=f, y=np.abs(X), mode="lines", name="Raw", line=dict(width=1)))
        specfig.add_trace(go.Scatter(x=f2, y=np.abs(X2), mode="lines", name="Filtered", line=dict(width=1)))
        specfig.update_xaxes(title="Frequency (Hz)", range=[0, min(1000, fs/2)])
        specfig.update_yaxes(title="|X(f)|", type="log")
        st.plotly_chart(style_figure(specfig, height=430, title="One-sided FFT magnitude"), use_container_width=True)
        st.caption(f"N={d.record.n:,} • fs={fs:.0f} Hz • Δf={fs/d.record.n:.4f} Hz • Nyquist={fs/2:.0f} Hz")
        display_note()
    else:
        f, X = rfft_spectrum(base["data"], fs)
        st.plotly_chart(line_figure(time, [base["data"]], ["Synthetic strain"], title="Synthetic signal"), use_container_width=True)
        ff = go.Figure(go.Scatter(x=f, y=np.abs(X), mode="lines")); ff.update_xaxes(title="Frequency (Hz)", range=[0,350]); ff.update_yaxes(type="log", title="|X|")
        st.plotly_chart(style_figure(ff, title="FFT"), use_container_width=True)


# ───────────────────────────── Spectrum & Noise ─────────────────────────────
elif page == "Spectrum & Noise":
    hero("NOISE CHARACTERIZATION", "Power and amplitude spectral density", "Welch spectral estimates characterize frequency-dependent detector noise used by whitening and matched filtering.")
    mode = st.radio("View", ["PSD", "ASD"], horizontal=True)
    if real is not None:
        d = real.h1 if detector_name == "H1" else real.l1
        y = d.psd if mode == "PSD" else asd_from_psd(d.psd)
        fig = go.Figure(go.Scatter(x=d.psd_freq, y=y, mode="lines", name=mode, line=dict(width=1.2)))
        fig.update_xaxes(title="Frequency (Hz)", type="log", range=[np.log10(10), np.log10(min(1000, fs/2))])
        fig.update_yaxes(title=mode, type="log")
        for f0 in spec.notches_hz:
            if f0 < fs/2:
                fig.add_vline(x=f0, line_dash="dot", line_color=CREAM_DIM)
        st.plotly_chart(style_figure(fig, height=500, title=f"{detector_name} Welch {mode}"), use_container_width=True)
    else:
        y = base["psd"] if mode == "PSD" else asd_from_psd(base["psd"])
        fig = go.Figure(go.Scatter(x=base["psd_freq"], y=y, mode="lines")); fig.update_xaxes(type="log", title="Frequency (Hz)"); fig.update_yaxes(type="log", title=mode)
        st.plotly_chart(style_figure(fig, height=500, title=f"Synthetic Welch {mode}"), use_container_width=True)


# ───────────────────────────── Signal Conditioning ─────────────────────────────
elif page == "Signal Conditioning":
    hero("SIGNAL CONDITIONING", "Filtering and whitening", "Zero-phase filtering defines the analysis band for visualization; PSD-based whitening equalizes frequency-dependent detector noise.")
    if real is not None:
        d = real.h1 if detector_name == "H1" else real.l1
        m = display_mask(
            d.record.time,
            event_t if spec.expected_signal else None,
            max(1.0, spec.display_half_width_s),
        )
        st.plotly_chart(line_figure(d.record.time[m], [d.raw[m], d.filtered[m], d.white[m]], ["Raw", "Filtered", "Whitened"], title="Before / after in time domain", y_label="Amplitude", height=430), use_container_width=True)
        nper = min(2048, d.raw.size)
        fr, pr = welch_psd(d.raw, fs, nperseg=nper, noverlap=nper//2)
        ff, pf = welch_psd(d.filtered, fs, nperseg=nper, noverlap=nper//2)
        fw, pw = welch_psd(d.white, fs, nperseg=nper, noverlap=nper//2)
        fig = go.Figure()
        for f,p,name in [(fr,pr,"Raw"),(ff,pf,"Filtered"),(fw,pw,"Whitened")]:
            fig.add_trace(go.Scatter(x=f,y=p,mode="lines",name=name,line=dict(width=1.1)))
        fig.update_xaxes(title="Frequency (Hz)", type="log", range=[np.log10(10),np.log10(min(1000,fs/2))]); fig.update_yaxes(title="PSD",type="log")
        st.plotly_chart(style_figure(fig, height=470, title="PSD before / after"), use_container_width=True)
        st.caption(f"Case settings: band-pass {spec.fband[0]:.0f}–{spec.fband[1]:.0f} Hz; notches {', '.join(str(int(x)) for x in spec.notches_hz)} Hz when inside Nyquist.")
        display_note()
    else:
        st.plotly_chart(line_figure(time, [base["data"], base["filtered"], base["white"]], ["Raw","Filtered","Whitened"], title="Synthetic preprocessing"), use_container_width=True)


# ───────────────────────────── Time–Frequency Analysis ─────────────────────────────
elif page == "Time–Frequency Analysis":
    hero("TIME–FREQUENCY ANALYSIS", "Short-time Fourier spectrogram", "The STFT resolves transient frequency evolution across time. Compact-binary inspiral power is expected to sweep upward toward coalescence.")
    if real is not None:
        d = real.h1 if detector_name == "H1" else real.l1
        db = power_to_db(d.stft_power, floor_db=-75)
        fm = d.stft_freq <= min(spec.fband[1], 900)
        if DISPLAY_MODE == "Auto focus" and event_t is not None:
            tm = (d.stft_time >= event_t - 2.0) & (d.stft_time <= event_t + 1.0)
        else:
            tm = np.ones_like(d.stft_time, dtype=bool)
        shown_db = db[fm][:, tm]
        zlo, zhi = spectrogram_limits(shown_db, floor_db=-75)
        heat = go.Figure(go.Heatmap(x=d.stft_time[tm], y=d.stft_freq[fm], z=shown_db, colorscale="Cividis", zmin=zlo,zmax=zhi,colorbar=dict(title="dB rel.")))
        if event_t is not None:
            heat.add_vline(x=event_t, line_dash="dash", line_color=GOLD, annotation_text="merger")
        heat.update_xaxes(title="Time (s)"); heat.update_yaxes(title="Frequency (Hz)")
        st.plotly_chart(style_figure(heat, height=560, title=f"{detector_name} whitened spectrogram"), use_container_width=True)
        if event_t is not None:
            st.caption("The GWOSC event time is shown as an external reference marker; the heatmap itself is computed from the selected detector strain.")
        display_note()
    else:
        db = power_to_db(base["spectrogram_power"], floor_db=-70); fm=base["stft_freq"]<=320
        heat=go.Figure(go.Heatmap(x=base["stft_time"],y=base["stft_freq"][fm],z=db[fm],colorscale="Cividis",zmin=-70,zmax=0)); heat.update_xaxes(title="Time (s)");heat.update_yaxes(title="Frequency (Hz)")
        st.plotly_chart(style_figure(heat,height=550,title="Synthetic chirp spectrogram"),use_container_width=True)


# ───────────────────────────── Event Detection ─────────────────────────────
elif page == "Event Detection":
    hero("EVENT DETECTION", "PSD-weighted matched-filter statistic", "The local candidate statistic compares detector strain with the configured compact-binary reference waveform while weighting each frequency by the measured noise PSD.")
    if real is not None:
        d = real.h1 if detector_name == "H1" else real.l1
        candidate_t = (d.lags + real.template_reference_index) / fs
        template_n = real.template_plus_compact.size
        full_overlap = (d.lags >= 0) & (d.lags <= max(0, d.record.n - template_n))
        valid = full_overlap & (candidate_t >= 0) & (candidate_t < d.record.duration)
        dd = d.detection
        if spec.expected_signal:
            timing_err = None if dd.peak_time_s is None else dd.peak_time_s - event_t
            recovered_here = detection_recovered(dd, event_t)
            detector_status = "RECOVERED" if recovered_here else ("CANDIDATE ELSEWHERE" if dd.detected else "NO CANDIDATE")
            cards([
                ("GravityHunter result", detector_status, f"candidate threshold {threshold:.2f}"),
                ("Peak SNR", f"{dd.peak_snr:.2f}", f"{detector_name} two-quadrature match"),
                ("Candidate time", f"{dd.peak_time_s:.4f} s" if dd.peak_time_s is not None else "—", f"GWOSC reference {event_t:.4f} s"),
                ("Candidate − reference", f"{timing_err*1000:+.1f} ms" if timing_err is not None else "—", "validation only; the reference is not used to choose the record-wide peak"),
            ])
            if not recovered_here:
                st.warning("The selected record contains a catalog event, but the local candidate does not satisfy the configured timing/threshold recovery criterion for this detector.")
        else:
            cards([
                ("Expected", "NO MERGER", "off-source negative control"),
                ("Candidate", "YES" if dd.detected else "NO", f"threshold {threshold:.2f}"),
                ("Peak SNR", f"{dd.peak_snr:.2f}", "a crossing here is a false alarm"),
            ])
        plot_t = candidate_t[valid]
        plot_snr = d.snr[valid]
        if plot_t.size > 30000:
            idx = np.linspace(0, plot_t.size - 1, 30000).astype(int)
            plot_t, plot_snr = plot_t[idx], plot_snr[idx]
        fig = go.Figure(go.Scatter(x=plot_t, y=plot_snr, mode="lines", name="SNR", line=dict(width=1.25, color=GREEN)))
        fig.add_hline(y=threshold, line_dash="dash", line_color=CREAM_DIM, annotation_text="threshold")
        if event_t is not None:
            fig.add_vline(x=event_t,line_dash="dot",line_color=GOLD,annotation_text="GWOSC reference")
        if dd.peak_time_s is not None:
            fig.add_vline(x=dd.peak_time_s,line_dash="dash",line_color=CREAM,annotation_text="candidate")
        focus_center = event_t if event_t is not None else dd.peak_time_s
        if DISPLAY_MODE == "Auto focus" and focus_center is not None:
            fig.update_xaxes(title="Candidate time (s)", range=[max(0, focus_center-3), min(d.record.duration, focus_center+3)])
        else:
            fig.update_xaxes(title="Candidate time (s)")
        fig.update_yaxes(title="Matched-filter SNR")
        st.plotly_chart(style_figure(fig,height=500,title=f"{detector_name} template-match score"),use_container_width=True)
        display_note()
        with st.expander("Matched-filter definition"):
            st.latex(r"z(t)\propto\mathcal{F}^{-1}\left\{\frac{X(f)S^*(f)}{S_n(f)}\right\}")
            st.write("The plus/cross template basis is combined using its measured noise-weighted overlap, so the two response channels are not assumed to be exactly orthogonal.")
    else:
        dd=detect_best_candidate(base["snr"],base["lags"],fs,threshold=threshold,prominence=1.0); ct=base["lags"]/fs;valid=(ct>=0)&(ct<=time[-1]);
        fig=go.Figure(go.Scatter(x=ct[valid],y=base["snr"][valid],mode="lines"));fig.add_hline(y=threshold);fig.add_vline(x=8,line_color=GOLD);fig.update_xaxes(range=[5,11],title="Time (s)");fig.update_yaxes(title="SNR");st.plotly_chart(style_figure(fig,height=480,title="Synthetic matched filter"),use_container_width=True)


# ───────────────────────────── Detector Coincidence ─────────────────────────────
elif page == "Detector Coincidence":
    hero("MULTI-DETECTOR ANALYSIS", "H1/L1 coincidence and arrival delay", "Independent detector candidates are compared in absolute GPS time. Cross-correlation of a common event window provides a second estimate of relative arrival delay.")
    if real is not None:
        h1d,l1d=real.h1.detection,real.l1.detection
        signed_candidate_ms = None
        if h1d.peak_time_s is not None and l1d.peak_time_s is not None:
            h1gps = real.h1.record.start_time + h1d.peak_time_s if real.h1.record.start_time is not None else h1d.peak_time_s
            l1gps = real.l1.record.start_time + l1d.peak_time_s if real.l1.record.start_time is not None else l1d.peak_time_s
            signed_candidate_ms = 1000.0 * (h1gps - l1gps)
        cards([
            ("H1", "DETECTED" if h1d.detected else "NO", f"SNR {h1d.peak_snr:.2f}"),
            ("L1", "DETECTED" if l1d.detected else "NO", f"SNR {l1d.peak_snr:.2f}"),
            ("Coincidence", "YES" if real.coincidence.coincident else "NO", "|H1 − L1| ≤ 10 ms"),
            ("H1 − L1 delay", f"{real.network_delay.delay_seconds*1000:+.2f} ms" if real.network_delay else "—", "positive = H1 later than L1"),
        ])
        if signed_candidate_ms is not None:
            st.caption(f"Matched-filter trigger difference H1 − L1: {signed_candidate_ms:+.3f} ms. Cross-correlation uses the same sign convention.")
        if spec.expected_signal and event_t is not None:
            mh = display_mask(real.h1.record.time, event_t, 0.22)
            ml = display_mask(real.l1.record.time, event_t, 0.22)
            fig=line_figure(real.h1.record.time[mh]-event_t,[real.h1.white[mh]], ["H1"],x_label="Time from merger (s)",y_label="Whitened strain",title="H1 detector strain",height=300)
            st.plotly_chart(fig,use_container_width=True)
            fig2=line_figure(real.l1.record.time[ml]-event_t,[real.l1.white[ml]], ["L1"],x_label="Time from merger (s)",y_label="Whitened strain",title="L1 detector strain",height=300)
            st.plotly_chart(fig2,use_container_width=True)
            if real.network_delay:
                lag_ms=real.network_delay.lags/fs*1000; mm=np.abs(lag_ms)<=10
                cc=go.Figure(go.Scatter(x=lag_ms[mm],y=real.network_delay.correlation[mm],mode="lines",name="cross-correlation"));cc.add_vline(x=real.network_delay.delay_seconds*1000,line_dash="dash",line_color=GOLD);cc.update_xaxes(title="Lag (ms)");cc.update_yaxes(title="Correlation")
                st.plotly_chart(style_figure(cc,height=390,title="H1/L1 cross-correlation"),use_container_width=True)
            display_note()
        else:
            status_banner("NEGATIVE CONTROL", "For a quiet window, there is no known astrophysical coincidence to recover. Candidate coincidence, if it occurs, should be treated as a false alarm.", "quiet")
    else:
        st.info("Multi-detector coincidence is evaluated only for local real H1/L1 records.")


# ───────────────────────────── Source Parameters ─────────────────────────────
elif page == "Source Parameters":
    if real is None or not spec.expected_signal:
        hero("CATALOG PARAMETERS", "No source-parameter record for this selection", "Source masses, remnant properties, distance and localization are displayed only for catalog merger events.")
    else:
        hero("CATALOG PARAMETERS", f"{spec.key}: published source properties", "Published GWOSC/GWTC parameter estimates are shown separately from quantities inferred by GravityHunter.")
        calc_mc=chirp_mass(spec.m1_source,spec.m2_source)
        cards([
            ("Primary", f"{spec.m1_source:.1f} M☉", "source-frame catalog median"),
            ("Secondary", f"{spec.m2_source:.1f} M☉", "source-frame catalog median"),
            ("Chirp mass", f"{spec.chirp_mass_source:.1f} M☉", f"formula from displayed medians ≈ {calc_mc:.1f} M☉"),
            ("Remnant", f"{spec.final_mass_source:.1f} M☉", f"≈ {spec.radiated_energy:.1f} M☉c² radiated"),
        ])
        st.plotly_chart(mass_sankey(spec),use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Luminosity distance",f"{spec.luminosity_distance_mpc:.0f} Mpc")
        c2.metric("Sky localization area",f"{spec.sky_area_deg2:.0f} deg²")
        c3.metric("Published network SNR",f"{spec.network_snr:.1f}")
        st.caption("Sky position is reported as a localization probability area; no sky-map reconstruction is performed in this build.")
        if spec.source_url:
            st.markdown(f"Reference metadata: `{spec.source_url}`")


# ───────────────────────────── Inspiral Diagnostics ─────────────────────────────
elif page == "Inspiral Diagnostics":
    hero(
        "INSPIRAL DIAGNOSTICS",
        "Frequency evolution and experimental chirp-mass fit",
        "A combined H1/L1 time-frequency ridge is fit to the leading-order inspiral relation t = tc − A f^(-8/3). This ridge-based mass estimate is an experimental diagnostic; event detection does not depend on it.",
    )
    if real is None:
        st.info("The synthetic validation chirp was not generated from a physical compact-binary inspiral law, so GravityHunter intentionally does not assign it a black-hole chirp mass. Select a real merger case.")
    elif not spec.expected_signal:
        status_banner(
            "NO CHIRP MASS FOR QUIET DATA",
            "A mass estimate is only attempted after a real merger candidate supplies a coherent inspiral region. GravityHunter will not turn arbitrary noise ridges into a black-hole mass.",
            "quiet",
        )
    else:
        est = chirp_mass_data(spec.key, float(threshold))
        if est is None:
            st.warning("No chirp-mass estimate is available for this selection.")
        else:
            ref = est.template_reference_mass_solar
            err = None
            if est.success and ref is not None and ref > 0:
                err = abs(est.estimated_mass_solar - ref) / ref * 100.0
            cards([
                ("Experimental ridge fit", f"{est.estimated_mass_solar:.2f} M☉" if est.estimated_mass_solar is not None else "unavailable", "H1+L1 • leading-order detector-frame diagnostic"),
                ("Template reference", f"{ref:.2f} M☉" if ref is not None else "—", "computed from template m₁,m₂ metadata"),
                ("Catalog source-frame", f"{spec.chirp_mass_source:.2f} M☉" if spec.chirp_mass_source is not None else "—", "published source-frame median"),
                ("Fit agreement", f"R² {est.fit_r2:.3f}" if est.fit_r2 is not None else "—", f"template-ref error {err:.1f}%" if err is not None else "quality diagnostic"),
            ])
            if est.success:
                ref_disagreement = None if ref is None or ref <= 0 else abs(est.estimated_mass_solar - ref) / ref
                if ref_disagreement is not None and ref_disagreement > 0.20:
                    status_banner(
                        "EXPERIMENTAL FIT — MODEL DISAGREEMENT",
                        "The ridge fit converged mathematically, but it differs by more than 20% from the chirp mass associated with the reference waveform. Treat this as a time-frequency diagnostic, not a validated source-parameter measurement.",
                        "bad",
                    )
                else:
                    status_banner("EXPERIMENTAL RIDGE FIT COMPLETED", est.message, "good")
            else:
                status_banner("NO STABLE RIDGE-MASS FIT", est.message, "bad")

            if est.inspiral_interval_s is not None and est.stft_power.size:
                a, b = est.inspiral_interval_s
                tm = (est.stft_time >= a) & (est.stft_time <= b)
                fm = (est.stft_freq >= 20.0) & (est.stft_freq <= min(spec.fband[1], 600.0))
                db = power_to_db(est.stft_power, floor_db=-75)
                shown_db = db[fm][:, tm]
                zlo, zhi = spectrogram_limits(shown_db, floor_db=-75)
                heat = go.Figure(go.Heatmap(
                    x=est.stft_time[tm] - event_t,
                    y=est.stft_freq[fm],
                    z=shown_db,
                    colorscale="Cividis", zmin=zlo, zmax=zhi, colorbar=dict(title="dB rel."),
                ))
                if est.ridge_time_s.size:
                    heat.add_trace(go.Scatter(
                        x=est.ridge_time_s-event_t, y=est.ridge_frequency_hz, mode="markers",
                        name="real-STFT ridge points", marker=dict(size=6, symbol="circle-open"),
                    ))
                    heat.add_trace(go.Scatter(
                        x=est.ridge_time_s-event_t, y=est.smooth_frequency_hz, mode="lines",
                        name="smoothed f(t)", line=dict(color=CREAM, width=2.2),
                    ))
                heat.update_xaxes(title="Time relative to merger (s)")
                heat.update_yaxes(title="Frequency (Hz)")
                st.plotly_chart(style_figure(heat, height=540, title="H1+L1 normalized inspiral power + extracted ridge"), use_container_width=True)

            if est.ridge_time_s.size:
                c1, c2 = st.columns(2)
                with c1:
                    ffig = go.Figure()
                    ffig.add_trace(go.Scatter(x=est.ridge_time_s-event_t, y=est.ridge_frequency_hz, mode="markers", name="ridge"))
                    ffig.add_trace(go.Scatter(x=est.ridge_time_s-event_t, y=est.smooth_frequency_hz, mode="lines", name="smooth f(t)", line=dict(color=CREAM, width=2)))
                    ffig.update_xaxes(title="Time from merger (s)"); ffig.update_yaxes(title="Frequency (Hz)")
                    st.plotly_chart(style_figure(ffig, height=360, title="Frequency evolution f(t)"), use_container_width=True)
                with c2:
                    dfig = go.Figure(go.Scatter(x=est.ridge_time_s-event_t, y=est.dfdt_hz_per_s, mode="lines+markers", name="df/dt"))
                    dfig.update_xaxes(title="Time from merger (s)"); dfig.update_yaxes(title="df/dt (Hz/s)")
                    st.plotly_chart(style_figure(dfig, height=360, title="Inspiral sweep rate df/dt"), use_container_width=True)

                valid_mass = np.isfinite(est.pointwise_mass_solar) & (est.pointwise_mass_solar > 0) & (est.pointwise_mass_solar < 300)
                if np.any(valid_mass):
                    mfig = go.Figure(go.Scatter(x=(est.ridge_time_s-event_t)[valid_mass], y=est.pointwise_mass_solar[valid_mass], mode="markers", name="pointwise Mchirp"))
                    if est.estimated_mass_solar is not None:
                        mfig.add_hline(y=est.estimated_mass_solar, line_color=CREAM, line_dash="dash", annotation_text="robust fit")
                    if ref is not None:
                        mfig.add_hline(y=ref, line_color=GOLD, line_dash="dot", annotation_text="template reference")
                    mfig.update_xaxes(title="Time from merger (s)"); mfig.update_yaxes(title="Chirp mass (M☉)")
                    st.plotly_chart(style_figure(mfig, height=370, title="Pointwise mass estimates and robust fitted result"), use_container_width=True)

            with st.expander("Inference model"):
                st.latex(r"t_c-t=\frac{5}{256}\left(\frac{G\mathcal{M}}{c^3}\right)^{-5/3}(\pi f)^{-8/3}")
                st.write("The fitted form is **t = tc − A f^(-8/3)**. This avoids using noisy numerical differentiation as the primary mass estimator; df/dt is retained as a secondary diagnostic.")
                st.write("The reference template defines only a local frequency-search corridor. Ridge frequencies are then extracted from the combined normalized H1/L1 STFT power inside that corridor.")
                st.warning("Observed frequency evolution measures a detector-frame/redshifted mass combination. The catalog card is source-frame, so do not interpret their difference as pure algorithm error. The template reference is the cleaner same-workflow comparison when template mass metadata are available.")


# ───────────────────────────── Audio Reconstruction ─────────────────────────────
elif page == "Audio Reconstruction":
    hero("SONIFICATION", "Audio reconstruction of conditioned strain", "The selected conditioned strain is normalized to audio. An optional +400 Hz frequency translation moves low-frequency merger structure into a more audible range.")
    if real is not None:
        d=real.h1 if detector_name=="H1" else real.l1
        if event_t is not None:
            center=event_t
        else:
            center=d.record.duration/2
        half=min(spec.sonify_half_width_s,d.record.duration/2)
        mask=(d.record.time>=center-half)&(d.record.time<center+half)
        audio=d.white[mask]
        st.markdown(f"#### {detector_name} • {2*half:.1f} second clip")
        c1,c2=st.columns(2)
        with c1:
            st.caption("Direct processed signal")
            st.audio(wav_bytes(audio,fs),format="audio/wav")
        with c2:
            st.caption("Enhanced: spectrum shifted upward by 400 Hz")
            st.audio(wav_bytes(audio,fs,shift_hz=400.0),format="audio/wav")
        if spec.expected_signal:
            tr=real.template; sl=tr.compact_slice(); temp=tr.plus[sl]
            st.markdown("#### Reference template")
            st.audio(wav_bytes(temp,fs,shift_hz=400.0),format="audio/wav")
        st.caption("Audio reconstruction is derived from conditioned strain and is not used by the detection statistic.")
    else:
        mask=(time>=6)&(time<=10); st.audio(wav_bytes(base["white"][mask],fs,shift_hz=400),format="audio/wav")


# ───────────────────────────── Synthetic Validation ─────────────────────────────
elif page == "Validation":
    hero("VALIDATION", "Controlled signal injection", "A known synthetic chirp is injected at adjustable amplitude to verify recovery behavior independently of the catalog-event records.")
    alpha=st.slider("Synthetic injection strength α",0.0,2.0,1.0,0.05)
    syn=synthetic_data(int(seed))
    # Single-injection validation view.
    start=int(round(syn["injection_start_s"]*syn["fs"]));template=syn["template"];m=template.size;background=syn["data"].copy();background[start:start+m]-=1.5*template;injected=background.copy();injected[start:start+m]+=alpha*template
    st.plotly_chart(line_figure(syn["time"], [injected], [f"α={alpha:.2f}"], title="Controlled injected signal"), use_container_width=True)
    st.caption("Synthetic injections are kept separate from catalog-event analysis.")


st.markdown("---")
st.markdown("<div class='gh-small'>GravityHunter • Gravitational-Wave Signal Analysis • GWOSC Open Data</div>", unsafe_allow_html=True)
