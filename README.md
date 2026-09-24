# GravityHunter

GravityHunter is a Streamlit application for gravitational-wave signal analysis using public GWOSC strain data. It implements the main DSP stages directly with NumPy/SciPy and exposes the intermediate results instead of treating the detector as a black box.

The bundled build contains three real binary-black-hole events, two off-source controls, and one synthetic validation case.

## What the application does

```text
H1 / L1 strain
    ↓
FFT + Welch PSD / ASD
    ↓
notch + band-pass conditioning
    ↓
PSD-based whitening
    ↓
STFT / spectrogram
    ↓
PSD-weighted two-basis matched filter
    ↓
SNR candidate timing
    ↓
H1/L1 coincidence + cross-correlation delay
```

Additional views provide waveform morphology, a synchronized merger timeline, catalog source parameters, sonification, and an experimental inspiral-ridge chirp-mass diagnostic.

## Bundled real-data cases

| Case | Role |
|---|---|
| GW150914 | Real H1/L1 merger record |
| GW151226 | Real H1/L1 merger record |
| GW170104 | Real H1/L1 merger record |
| Quiet GW150914 interval | Off-source negative control |
| Quiet GW170104 interval | Off-source negative control |
| Synthetic injection | Controlled algorithm validation |

The final backend regression suite verifies recovery of all three real events and rejection of the two quiet controls.

## Analysis views

- **Overview** — selected record, detector status, SNR summary, and pipeline state.
- **Waveform Morphology** — inspiral, merger, and ringdown reference morphology beside measured strain and spectrogram data.
- **Event Timeline** — synchronized record/GPS clock, waveform, SNR, and schematic binary-merger visualization.
- **Detector Strain** — calibrated strain and one-sided Fourier magnitude.
- **Spectrum & Noise** — Welch PSD/ASD and detector noise structure.
- **Signal Conditioning** — filtering and whitening before/after views.
- **Time–Frequency Analysis** — STFT spectrogram of the selected detector channel.
- **Event Detection** — GravityHunter matched-filter SNR, threshold, and candidate time.
- **Detector Coincidence** — independent H1/L1 candidates and signed arrival-delay estimate.
- **Source Parameters** — published event properties, kept separate from GravityHunter inference.
- **Inspiral Diagnostics** — experimental H1/L1 ridge extraction and leading-order chirp-mass fit.
- **Audio Reconstruction** — sonification of conditioned detector strain.
- **Validation** — controlled synthetic injection.

The **Auto focus / Full record** display control changes only chart framing. Robust amplitude scaling and adaptive spectrogram contrast are also display-only; all DSP calculations use unchanged full-resolution arrays.

## Run locally

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/validate_bundled_data.py
python -m streamlit run streamlit_app.py
```

Windows users can also run `run_local.bat` after installing dependencies.

## Data layout

```text
data/
├── GW150914/{H1,L1}.hdf5
├── GW151226/{H1,L1}.hdf5
├── GW170104/{H1,L1}.hdf5
├── templates/
│   ├── GW150914_4_template.hdf5
│   ├── GW151226_4_template.hdf5
│   └── GW170104_4_template.hdf5
└── SHA256SUMS.txt
```

The bundled HDF5 files are public GWOSC data/reference-waveform files used by the verified build. No runtime download is required.


## Scope and interpretation

GravityHunter is an independent course-project implementation, not the production LIGO/Virgo/KAGRA search pipeline.

- Matched-filter SNR is the primary single-detector event statistic.
- H1/L1 coincidence and cross-correlation provide network consistency checks.
- Catalog timestamps and source parameters are external reference information.
- Spectrogram morphology is visualization, not the primary detection statistic.
- The synchronized spacetime-fabric animation is schematic; its timing and waveform dynamics are data-driven, but it is not a numerical solution of Einstein's field equations.
- Inspiral ridge/chirp-mass fitting is an experimental diagnostic and does not affect event detection.

Public data source: https://gwosc.org/
