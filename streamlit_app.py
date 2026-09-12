from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from examples.synthetic_pipeline import build_demo
from gravityhunter.dsp.detector import detect_best_candidate
from gravityhunter.dsp.fft_tools import rfft_spectrum
from gravityhunter.dsp.filters import (
    apply_bandpass_zero_phase,
    apply_notch_zero_phase,
)
from gravityhunter.dsp.loader import inspect_hdf5, load_common_gwosc_hdf5
from gravityhunter.dsp.matched_filter import matched_filter_snr
from gravityhunter.dsp.network import coincidence_check, estimate_delay
from gravityhunter.dsp.psd import asd_from_psd, welch_psd
from gravityhunter.dsp.stft_tools import power_to_db
from gravityhunter.dsp.whitening import whiten


st.set_page_config(
    page_title="GravityHunter",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


CREAM = "#F3E8C8"
CREAM_DIM = "#D6C8A7"
NAVY = "#061427"
NAVY_2 = "#0B203B"
BLUE = "#7FA9D6"
GRID = "rgba(243,232,200,0.10)"


GLOBAL_CSS = f"""
<style>
:root {{
  --gh-cream: {CREAM};
  --gh-cream-dim: {CREAM_DIM};
  --gh-navy: {NAVY};
  --gh-navy2: {NAVY_2};
}}

.stApp {{
  color: var(--gh-cream);
  background-color: {NAVY};
  background-image:
    radial-gradient(circle at 18px 24px, rgba(243,232,200,.78) 0 1.1px, transparent 1.6px),
    radial-gradient(circle at 72px 88px, rgba(243,232,200,.46) 0 1px, transparent 1.5px),
    radial-gradient(circle at 35px 106px, rgba(243,232,200,.30) 0 .8px, transparent 1.3px),
    radial-gradient(circle at 115px 38px, rgba(243,232,200,.24) 0 .7px, transparent 1.2px),
    linear-gradient(160deg, #061427 0%, #081a31 52%, #04101f 100%);
  background-size: 140px 140px, 180px 180px, 120px 120px, 210px 210px, auto;
  background-attachment: fixed;
}}

[data-testid="stHeader"] {{ background: rgba(6,20,39,.75); }}
[data-testid="stSidebar"] {{
  background: rgba(7, 24, 45, .96);
  border-right: 1px solid rgba(243,232,200,.12);
}}
[data-testid="stSidebar"] * {{ color: var(--gh-cream); }}

.block-container {{
  max-width: 1450px;
  padding-top: 2rem;
  padding-bottom: 4rem;
}}

h1, h2, h3, h4 {{ color: var(--gh-cream) !important; letter-spacing: -0.02em; }}
p, li, label {{ color: #E9E5DA; }}

.gh-hero {{
  background: linear-gradient(135deg, rgba(15,48,82,.84), rgba(6,20,39,.76));
  border: 1px solid rgba(243,232,200,.18);
  box-shadow: 0 18px 45px rgba(0,0,0,.22);
  border-radius: 24px;
  padding: 28px 30px 24px 30px;
  margin-bottom: 20px;
  backdrop-filter: blur(10px);
}}
.gh-kicker {{
  color: {CREAM_DIM};
  text-transform: uppercase;
  letter-spacing: .16em;
  font-size: .76rem;
  font-weight: 700;
}}
.gh-title {{
  color: {CREAM};
  font-size: 2.3rem;
  line-height: 1.05;
  font-weight: 760;
  margin: .35rem 0 .55rem 0;
}}
.gh-sub {{
  color: #CCD9E6;
  max-width: 900px;
  font-size: 1rem;
}}
.gh-card {{
  background: rgba(9, 31, 58, .78);
  border: 1px solid rgba(243,232,200,.14);
  border-radius: 18px;
  padding: 16px 18px;
  min-height: 112px;
  box-shadow: 0 10px 26px rgba(0,0,0,.15);
}}
.gh-card .value {{ color: {CREAM}; font-size: 1.55rem; font-weight: 740; }}
.gh-card .label {{ color: #9FB5CB; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
.gh-card .note {{ color: #C8D5E1; font-size: .84rem; margin-top: .35rem; }}
.gh-pill {{
  display: inline-block;
  padding: 6px 10px;
  margin: 3px 4px 3px 0;
  border-radius: 999px;
  border: 1px solid rgba(243,232,200,.16);
  background: rgba(127,169,214,.08);
  color: {CREAM};
  font-size: .78rem;
}}
.gh-ok {{ color: #B9E3CB; }}
.gh-warn {{ color: #F1D4A4; }}
.gh-small {{ color:#AFC0D0; font-size:.82rem; }}

[data-testid="stMetric"] {{
  background: rgba(9,31,58,.72);
  border: 1px solid rgba(243,232,200,.12);
  padding: 12px 14px;
  border-radius: 16px;
}}
[data-testid="stMetricLabel"] {{ color: #9FB5CB; }}
[data-testid="stMetricValue"] {{ color: {CREAM}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
  background: rgba(10,34,63,.72);
  border-radius: 12px 12px 0 0;
  color: #CAD7E3;
  border: 1px solid rgba(243,232,200,.09);
}}
.stTabs [aria-selected="true"] {{
  color: {CREAM} !important;
  background: rgba(30,64,99,.88) !important;
}}

div[data-testid="stExpander"] {{
  background: rgba(8,28,53,.68);
  border-color: rgba(243,232,200,.12);
  border-radius: 14px;
}}

hr {{ border-color: rgba(243,232,200,.12); }}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def style_figure(fig: go.Figure, *, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,20,39,.78)",
        font=dict(color=CREAM),
        margin=dict(l=30, r=20, t=55 if title else 25, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=NAVY_2, font_color=CREAM),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=CREAM_DIM))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=CREAM_DIM))
    return fig


def line_figure(x, ys, labels, *, x_label="Time (s)", y_label="Amplitude", title=None, height=380):
    fig = go.Figure()
    for y, label in zip(ys, labels):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(width=1.4)))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return style_figure(fig, height=height, title=title)


def hero(section: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="gh-hero">
          <div class="gh-kicker">✦ {section}</div>
          <div class="gh-title">{title}</div>
          <div class="gh-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cards(items):
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="gh-card">
                  <div class="label">{label}</div>
                  <div class="value">{value}</div>
                  <div class="note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def demo_data(seed: int):
    return build_demo(seed=seed)


@st.cache_data(show_spinner=False)
def network_demo(seed: int, delay_samples: int):
    base = build_demo(seed=seed)
    fs = base["fs"]
    template = base["template"]
    n = int(2.8 * fs)
    start = int(0.55 * fs)
    rng = np.random.default_rng(seed + 1000)

    h1 = rng.normal(0, 0.24, n)
    l1 = rng.normal(0, 0.24, n)
    h1[start:start + template.size] += template
    l1_start = start + delay_samples
    l1[l1_start:l1_start + template.size] += 0.88 * template

    est = estimate_delay(
        h1,
        l1,
        fs,
        max_abs_delay_s=0.03,
        use_absolute_peak=True,
    )
    return {
        "fs": fs,
        "time": np.arange(n) / fs,
        "h1": h1,
        "l1": l1,
        "true_delay_s": delay_samples / fs,
        "h1_time": start / fs,
        "l1_time": l1_start / fs,
        "estimate": est,
    }


@st.cache_data(show_spinner=False)
def injection_run(seed: int, alpha: float, threshold: float):
    base = build_demo(seed=seed)
    fs = base["fs"]
    data = base["data"].copy()
    template = base["template"]
    start = int(round(base["injection_start_s"] * fs))
    m = template.size

    # Recover the synthetic background used by build_demo(), then inject a
    # user-controlled amplitude.
    background = data.copy()
    background[start:start + m] -= 1.5 * template
    injected = background.copy()
    injected[start:start + m] += alpha * template

    filtered, _ = apply_notch_zero_phase(injected, fs, 60.0, q=30)
    filtered, _ = apply_bandpass_zero_phase(filtered, fs, 25.0, 300.0, order=4)

    template_f, _ = apply_notch_zero_phase(template, fs, 60.0, q=30)
    template_f, _ = apply_bandpass_zero_phase(template_f, fs, 25.0, 300.0, order=4)

    lags, snr, _ = matched_filter_snr(
        filtered,
        template_f,
        fs,
        base["psd_freq"],
        base["psd"],
        fmin=25.0,
        fmax=300.0,
    )
    det = detect_best_candidate(
        snr,
        lags,
        fs,
        threshold=threshold,
        template_reference_index=0,
        prominence=1.0,
    )
    return injected, lags, snr, det, start / fs


# Sidebar
st.sidebar.markdown("## ✦ GravityHunter")
st.sidebar.caption("Gravitational-wave DSP workbench")
page = st.sidebar.radio(
    "Navigate",
    [
        "Mission Control",
        "Signal & FFT",
        "Noise / PSD",
        "Cleaning & Whitening",
        "Time–Frequency",
        "Matched Filter",
        "Detector Network",
        "Injection Lab",
        "Real HDF5 Preview",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
seed = st.sidebar.number_input("Synthetic seed", min_value=0, max_value=9999, value=7, step=1)
threshold = st.sidebar.slider("Demo SNR threshold", 1.0, 15.0, 5.0, 0.25)
st.sidebar.caption("The synthetic threshold is a demo parameter, not a universal LIGO detection threshold.")

with st.sidebar.expander("Core pipeline", expanded=False):
    st.markdown(
        """
        `strain → FFT → PSD → notch/band-pass → whitening → STFT → matched filter → SNR → coincidence`
        """
    )

base = demo_data(int(seed))
fs = float(base["fs"])
time = base["time"]


if page == "Mission Control":
    hero(
        "Local demo",
        "GravityHunter Mission Control",
        "A working localhost dashboard over the DSP modules implemented so far. The default source is a controlled synthetic chirp so every stage can be demonstrated before real GWOSC integration is finalized.",
    )

    cards([
        ("Sample rate", f"{fs:.0f} Hz", "Synthetic interferometer stream"),
        ("Duration", f"{time[-1]:.1f} s", "One analysis segment"),
        ("Peak SNR", f"{base['detection'].peak_snr:.2f}", "PSD-weighted matched filter"),
        ("Candidate", f"{base['detection'].peak_time_s:.3f} s", "True injection starts at 8.000 s"),
    ])

    st.markdown("### Pipeline status")
    st.markdown(
        """
        <div>
          <span class="gh-pill">✓ FFT</span>
          <span class="gh-pill">✓ Welch PSD / ASD</span>
          <span class="gh-pill">✓ notch + band-pass</span>
          <span class="gh-pill">✓ whitening</span>
          <span class="gh-pill">✓ STFT</span>
          <span class="gh-pill">✓ matched filter</span>
          <span class="gh-pill">✓ peak detection</span>
          <span class="gh-pill">✓ H1/L1 delay</span>
          <span class="gh-pill">✓ injections</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    event_lo, event_hi = 6.8, 10.0
    mask = (time >= event_lo) & (time <= event_hi)
    fig = line_figure(
        time[mask],
        [base["data"][mask], base["filtered"][mask]],
        ["Raw synthetic strain", "Filtered"],
        title="Event neighborhood",
        y_label="Relative strain amplitude",
    )
    fig.add_vline(x=base["injection_start_s"], line_dash="dash", line_color=CREAM_DIM)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Use the left navigation to demonstrate each DSP stage separately. The Real HDF5 Preview page already accepts a GWOSC-style HDF5 upload for waveform/FFT/PSD inspection.")


elif page == "Signal & FFT":
    hero(
        "Module 2–4",
        "Signal, sampling, and spectrum",
        "Inspect the sampled waveform and its one-sided FFT. The 60 Hz synthetic line is deliberately present so the spectral cleaning stage has something obvious to remove.",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        view = st.slider("Time window (seconds)", 0.0, float(time[-1]), (6.0, 11.0), 0.1)
        mask = (time >= view[0]) & (time <= view[1])
        st.plotly_chart(
            line_figure(time[mask], [base["data"][mask]], ["Raw strain"], title="Time-domain strain"),
            use_container_width=True,
        )

    f_raw, X_raw = rfft_spectrum(base["data"], fs)
    f_filt, X_filt = rfft_spectrum(base["filtered"], fs)
    with c2:
        spec = go.Figure()
        spec.add_trace(go.Scatter(x=f_raw, y=np.abs(X_raw), mode="lines", name="Raw", line=dict(width=1.2)))
        spec.add_trace(go.Scatter(x=f_filt, y=np.abs(X_filt), mode="lines", name="Filtered", line=dict(width=1.2)))
        spec.update_xaxes(title="Frequency (Hz)", range=[0, 350])
        spec.update_yaxes(title="|X(f)|", type="log")
        st.plotly_chart(style_figure(spec, title="One-sided FFT magnitude"), use_container_width=True)

    st.caption(f"N = {base['data'].size:,} samples • Δf = {fs / base['data'].size:.4f} Hz • Nyquist = {fs/2:.1f} Hz")


elif page == "Noise / PSD":
    hero(
        "Module 5",
        "Noise spectrum, PSD, and ASD",
        "Welch averaging turns fluctuating FFT power into a more stable estimate of frequency-dependent noise. Toggle between PSD and ASD to show the distinction clearly.",
    )

    mode = st.radio("View", ["PSD", "ASD"], horizontal=True)
    if mode == "PSD":
        y = base["psd"]
        ylabel = "PSD (relative units²/Hz)"
    else:
        y = asd_from_psd(base["psd"])
        ylabel = "ASD (relative units/√Hz)"

    fig = go.Figure(go.Scatter(x=base["psd_freq"], y=y, mode="lines", name=mode, line=dict(width=1.4)))
    fig.update_xaxes(title="Frequency (Hz)", range=[1, 400], type="log")
    fig.update_yaxes(title=ylabel, type="log")
    fig.add_vline(x=60.0, line_dash="dash", line_color=CREAM_DIM, annotation_text="60 Hz line")
    st.plotly_chart(style_figure(fig, height=470, title=f"Welch {mode}"), use_container_width=True)

    cards([
        ("Welch segment", "2048 samples", f"{2048/fs:.2f} s per segment"),
        ("Overlap", "50%", "1024 samples"),
        ("PSD spacing", f"{fs/2048:.2f} Hz", "Set by segment length"),
    ])


elif page == "Cleaning & Whitening":
    hero(
        "Module 6–7",
        "Filtering and whitening",
        "The synthetic pipeline first suppresses a narrow 60 Hz line, then uses a 25–300 Hz Butterworth band-pass. Whitening divides spectral amplitudes by √PSD so the retained noise becomes much more uniform across frequency.",
    )

    event = (time >= 6.8) & (time <= 10.2)
    fig = line_figure(
        time[event],
        [base["data"][event], base["filtered"][event], base["white"][event]],
        ["Raw", "Filtered", "Whitened (standardized)"],
        title="Waveform comparison",
        y_label="Amplitude",
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    nperseg = 2048
    f0, p0 = welch_psd(base["data"], fs, nperseg=nperseg, noverlap=nperseg//2)
    f1, p1 = welch_psd(base["filtered"], fs, nperseg=nperseg, noverlap=nperseg//2)
    f2, p2 = welch_psd(base["white"], fs, nperseg=nperseg, noverlap=nperseg//2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=f0, y=p0, mode="lines", name="Raw"))
    fig2.add_trace(go.Scatter(x=f1, y=p1, mode="lines", name="Filtered"))
    fig2.add_trace(go.Scatter(x=f2, y=p2, mode="lines", name="Whitened"))
    fig2.update_xaxes(title="Frequency (Hz)", range=[10, 350], type="log")
    fig2.update_yaxes(title="PSD", type="log")
    st.plotly_chart(style_figure(fig2, height=450, title="PSD before and after cleaning"), use_container_width=True)

    st.markdown("<span class='gh-small'>Demo settings: 60 Hz notch (Q=30), 25–300 Hz band-pass, order 4. These are presentation defaults for the synthetic stream, not final real-data choices.</span>", unsafe_allow_html=True)


elif page == "Time–Frequency":
    hero(
        "Module 8",
        "STFT spectrogram",
        "The FFT says which frequencies exist; the STFT keeps the time dimension. The rising ridge is the synthetic chirp moving from low to high frequency.",
    )

    power_db = power_to_db(base["spectrogram_power"], floor_db=-70)
    fmask = base["stft_freq"] <= 320
    heat = go.Figure(
        data=go.Heatmap(
            x=base["stft_time"],
            y=base["stft_freq"][fmask],
            z=power_db[fmask, :],
            colorscale="Cividis",
            colorbar=dict(title="dB rel."),
            zmin=-70,
            zmax=0,
        )
    )
    heat.update_xaxes(title="Time (s)")
    heat.update_yaxes(title="Frequency (Hz)")
    heat.add_vline(x=base["injection_start_s"], line_dash="dash", line_color=CREAM_DIM)
    st.plotly_chart(style_figure(heat, height=550, title="Whitened spectrogram"), use_container_width=True)

    cards([
        ("Window", "256 samples", f"{256/fs*1000:.0f} ms"),
        ("Overlap", "192 samples", "75% overlap"),
        ("Hop", "64 samples", f"{64/fs*1000:.1f} ms/frame"),
        ("STFT shape", str(base["spectrogram_power"].shape), "(frequency, time)"),
    ])


elif page == "Matched Filter":
    hero(
        "Module 9–10",
        "Matched filter and event candidate",
        "The detector correlates the expected chirp with the data while weighting every frequency by the inverse noise PSD. The result is an SNR-like score versus lag/time.",
    )

    # Re-evaluate only the threshold so the sidebar is interactive.
    det = detect_best_candidate(
        base["snr"],
        base["lags"],
        fs,
        threshold=threshold,
        template_reference_index=0,
        prominence=1.0,
    )
    lag_time = base["lags"] / fs
    valid = (lag_time >= 0) & (lag_time <= time[-1])

    cards([
        ("Status", "DETECTED" if det.detected else "NO CANDIDATE", f"Threshold {threshold:.2f}"),
        ("Peak SNR", f"{det.peak_snr:.2f}", "Noise-weighted template match"),
        ("Candidate time", f"{det.peak_time_s:.3f} s" if det.peak_time_s is not None else "—", "True injection: 8.000 s"),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lag_time[valid], y=base["snr"][valid], mode="lines", name="SNR", line=dict(width=1.35)))
    fig.add_hline(y=threshold, line_dash="dash", line_color=CREAM_DIM, annotation_text="threshold")
    fig.add_vline(x=base["injection_start_s"], line_dash="dot", line_color=BLUE, annotation_text="true injection")
    if det.peak_time_s is not None:
        fig.add_vline(x=det.peak_time_s, line_dash="dash", line_color=CREAM, annotation_text="candidate")
    fig.update_xaxes(title="Template start / lag time (s)", range=[5, 11])
    fig.update_yaxes(title="Matched-filter SNR")
    st.plotly_chart(style_figure(fig, height=480, title="SNR versus time"), use_container_width=True)

    with st.expander("What the detector is computing"):
        st.latex(r"z(t) \propto \mathcal{F}^{-1}\left\{\frac{X(f)S^*(f)}{S_n(f)}\right\}")
        st.write("The numerator is correlation in frequency space; dividing by the PSD downweights frequencies where the detector is naturally noisy.")


elif page == "Detector Network":
    hero(
        "Module 11",
        "H1 / L1 timing and coincidence",
        "This synthetic two-detector panel injects the same chirp into independent noise streams with a small known delay. Cross-correlation estimates the relative lag, then a simplified coincidence check tests whether both candidate times are compatible.",
    )

    delay_samples = st.slider("Synthetic L1 delay (samples)", 1, 15, 7, 1)
    net = network_demo(int(seed), int(delay_samples))
    est = net["estimate"]
    coincidence = coincidence_check(
        h1_detected=True,
        l1_detected=True,
        h1_time_s=net["h1_time"],
        l1_time_s=net["l1_time"],
        max_time_difference_s=0.02,
    )

    cards([
        ("Known delay", f"{net['true_delay_s']*1000:.2f} ms", f"{delay_samples} samples"),
        ("Estimated |delay|", f"{abs(est.delay_seconds)*1000:.2f} ms", f"lag = {est.best_lag_samples} samples"),
        ("Coincidence", "YES" if coincidence.coincident else "NO", "20 ms demo compatibility window"),
    ])

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = line_figure(net["time"], [net["h1"], net["l1"]], ["H1", "L1"], title="Synthetic detector streams", y_label="Amplitude", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        lag_s = est.lags / net["fs"]
        mask = np.abs(lag_s) <= 0.03
        fig2 = go.Figure(go.Scatter(x=lag_s[mask] * 1000, y=est.correlation[mask], mode="lines", name="cross-correlation"))
        fig2.add_vline(x=est.delay_seconds * 1000, line_dash="dash", line_color=CREAM_DIM)
        fig2.update_xaxes(title="Lag (ms)")
        fig2.update_yaxes(title="Correlation")
        st.plotly_chart(style_figure(fig2, height=420, title="H1/L1 cross-correlation"), use_container_width=True)

    st.caption("Lag sign follows the implemented correlation convention. For the final report, validate which sign means H1-first/L1-first using a known synthetic shift, then keep that convention fixed.")


elif page == "Injection Lab":
    hero(
        "Module 12",
        "Signal injection and detector stress test",
        "Recover the synthetic background, inject the same chirp at a controllable amplitude α, and rerun the matched filter. This is the foundation for later detection-probability and ROC experiments.",
    )

    alpha = st.slider("Injection strength α", 0.0, 2.0, 1.0, 0.05)
    injected, lags, snr, det, true_t = injection_run(int(seed), float(alpha), float(threshold))
    lag_t = lags / fs
    valid = (lag_t >= 0) & (lag_t <= time[-1])

    cards([
        ("α", f"{alpha:.2f}", "Signal amplitude scale"),
        ("Peak SNR", f"{det.peak_snr:.2f}", "Changes with injection strength"),
        ("Detected", "YES" if det.detected else "NO", f"Threshold {threshold:.2f}"),
    ])

    fig = go.Figure(go.Scatter(x=lag_t[valid], y=snr[valid], mode="lines", name="SNR", line=dict(width=1.3)))
    fig.add_hline(y=threshold, line_dash="dash", line_color=CREAM_DIM)
    fig.add_vline(x=true_t, line_dash="dot", line_color=BLUE, annotation_text="true injection")
    if det.peak_time_s is not None:
        fig.add_vline(x=det.peak_time_s, line_dash="dash", line_color=CREAM, annotation_text="candidate")
    fig.update_xaxes(title="Time (s)", range=[5, 11])
    fig.update_yaxes(title="SNR")
    st.plotly_chart(style_figure(fig, height=480, title="Detection strength under controlled injection"), use_container_width=True)

    st.markdown("#### Quick strength sweep")
    sweep = np.linspace(0.0, 2.0, 9)
    peaks = []
    for a in sweep:
        _, _, _, d, _ = injection_run(int(seed), float(a), float(threshold))
        peaks.append(d.peak_snr)
    fig2 = go.Figure(go.Scatter(x=sweep, y=peaks, mode="lines+markers", name="Peak SNR"))
    fig2.add_hline(y=threshold, line_dash="dash", line_color=CREAM_DIM, annotation_text="threshold")
    fig2.update_xaxes(title="Injection strength α")
    fig2.update_yaxes(title="Peak SNR")
    st.plotly_chart(style_figure(fig2, height=360), use_container_width=True)


elif page == "Real HDF5 Preview":
    hero(
        "Real-data bridge",
        "GWOSC HDF5 preview",
        "Upload one local detector file. This basic frontend will load the common GWOSC strain layout and immediately show metadata, waveform, FFT, and a first Welch PSD. Real matched filtering still needs an event-compatible template and deliberately chosen preprocessing parameters.",
    )

    upload = st.file_uploader("Upload an H1 or L1 HDF5 file", type=["hdf5", "h5"])
    if upload is None:
        st.markdown(
            """
            <div class="gh-card">
              <div class="label">Waiting for data</div>
              <div class="value">Drop an HDF5 file here</div>
              <div class="note">The synthetic pages remain fully functional without any external file.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        suffix = Path(upload.name).suffix or ".hdf5"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = tmp.name

        try:
            rec = load_common_gwosc_hdf5(tmp_path)
            cards([
                ("Detector", rec.detector or "unknown", upload.name),
                ("Samples", f"{rec.n:,}", f"duration {rec.duration:.2f} s"),
                ("Sample rate", f"{rec.fs:.0f} Hz", f"Nyquist {rec.fs/2:.0f} Hz"),
            ])

            # Downsample only for drawing the raw waveform; DSP stays on original samples.
            stride = max(1, rec.n // 25000)
            fig = line_figure(rec.time[::stride], [rec.strain[::stride]], ["strain"], title="Uploaded strain waveform", y_label="Calibrated strain", height=410)
            st.plotly_chart(fig, use_container_width=True)

            f, X = rfft_spectrum(rec.strain, rec.fs)
            fft_fig = go.Figure(go.Scatter(x=f, y=np.abs(X), mode="lines", name="FFT", line=dict(width=1)))
            fft_fig.update_xaxes(title="Frequency (Hz)", range=[0, min(500, rec.fs/2)])
            fft_fig.update_yaxes(title="|X|", type="log")

            noise_seconds = min(8.0, rec.duration)
            noise_n = max(16, int(noise_seconds * rec.fs))
            noise = rec.strain[:noise_n]
            nperseg = int(min(4096, noise.size))
            if nperseg % 2 == 1:
                nperseg -= 1
            f_p, p = welch_psd(noise, rec.fs, nperseg=nperseg, noverlap=nperseg//2)
            psd_fig = go.Figure(go.Scatter(x=f_p, y=p, mode="lines", name="PSD", line=dict(width=1.1)))
            psd_fig.update_xaxes(title="Frequency (Hz)", range=[1, min(500, rec.fs/2)], type="log")
            psd_fig.update_yaxes(title="PSD", type="log")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(style_figure(fft_fig, title="FFT magnitude"), use_container_width=True)
            with c2:
                st.plotly_chart(style_figure(psd_fig, title="First Welch PSD"), use_container_width=True)

            st.success("Real strain is loaded. Next, we should choose the event/noise intervals from this file and wire the same filter → whitening → STFT → matched-filter chain using a compatible template.")

        except Exception as exc:
            st.error(f"The convenience GWOSC loader could not read this file: {exc}")
            st.caption("HDF5 hierarchy found in the upload:")
            try:
                for item in inspect_hdf5(tmp_path):
                    st.code(item)
            except Exception as inspect_exc:
                st.error(f"Could not inspect HDF5 structure: {inspect_exc}")


st.markdown("---")
st.markdown(
    "<div class='gh-small'>GravityHunter • educational DSP workbench • current localhost prototype</div>",
    unsafe_allow_html=True,
)
