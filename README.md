# GravityHunter — Gravitational-Wave Signal Analysis

GravityHunter is an offline gravitational-wave signal-analysis application built around bundled public GWOSC H1/L1 strain records. The application runs the same DSP backend on known compact-binary merger records and off-source controls, with a synthetic injection retained for algorithm validation.

## Bundled cases

- **GW150914** — real H1 + L1 strain
- **GW151226** — real H1 + L1 strain
- **GW170104** — real H1 + L1 strain
- **Quiet interval from the GW150914 record** — negative control
- **Quiet interval from the GW170104 record** — negative control
- **Synthetic validation chirp** — controlled signal-injection test

No upload or network download is required for the bundled cases.

## Core analysis path

```text
local H1/L1 HDF5
        ↓
strain + GPS metadata
        ↓
off-source Welch PSD
        ↓
signal conditioning + whitening
        ↓
STFT / spectrogram
        ↓
public plus/cross waveform reference
        ↓
PSD-weighted two-basis matched filter
        ↓
SNR vs time
        ↓
H1/L1 candidate timestamps
        ↓
coincidence + cross-correlation delay
```

Detection is performed on tapered raw strain and template waveforms with inverse-PSD frequency weighting. The separately conditioned/whitened time series are used for visualization, STFT, cross-correlation and audio reconstruction.

## Analysis views

### Overview
Selected case, detector status, candidate SNRs and network result.

### Waveform Morphology
Reference inspiral/merger/ringdown morphology shown beside the processed real strain and spectrogram. The phase regions are visualization regions derived from the public reference waveform; they are not independent GR parameter inference.

### Event Timeline
Synchronized 32-second event visualization with record time, absolute GPS time, processed strain, matched-filter SNR and schematic binary motion. The collision can use a validated GravityHunter trigger time or the external GWOSC reference timestamp. Quiet controls do not trigger a merger animation.

### Detector Strain
Raw calibrated strain and Fourier magnitude spectrum.

### Spectrum & Noise
Welch PSD / ASD used to characterize the detector's colored noise.

### Signal Conditioning
Band-pass/notch conditioning and PSD-based whitening with before/after comparisons.

### Time–Frequency Analysis
STFT spectrogram calculated from the selected real detector channel.

### Event Detection
GravityHunter's quantitative matched-filter candidate: SNR-vs-time, fixed/default candidate threshold and recovered timestamp. This is the main page for the project's own event-detection claim.

### Detector Coincidence
Independent H1/L1 candidate status, absolute-time coincidence and waveform cross-correlation delay.

### Source Parameters
Published/reference event properties kept explicitly separate from quantities calculated by GravityHunter.

### Inspiral Diagnostics
Experimental H1+L1 spectrogram-ridge analysis. It displays frequency evolution, sweep rate and a leading-order chirp-mass diagnostic where the fit is stable. This page is **not** part of the core event-detection decision and is not presented as full LVK parameter estimation.

### Audio Reconstruction
Audio generated from the same processed detector strain, including an optional +400 Hz frequency translation.

### Synthetic Validation
Controlled injection experiment for checking detector response as signal strength changes.

## Local data layout

The final project bundle already contains:

```text
data/
├── GW150914/
│   ├── H1.hdf5
│   └── L1.hdf5
├── GW151226/
│   ├── H1.hdf5
│   └── L1.hdf5
├── GW170104/
│   ├── H1.hdf5
│   └── L1.hdf5
├── templates/
│   ├── GW150914_4_template.hdf5
│   ├── GW151226_4_template.hdf5
│   └── GW170104_4_template.hdf5
└── SHA256SUMS.txt
```

These are the exact files used for the final backend acceptance test. There is no downloader dependency.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run streamlit_app.py
```

On Windows you can also run:

```text
run_local.bat
```

Streamlit normally serves the app at:

```text
http://localhost:8501
```

## Verification

Run the complete automated test suite:

```bash
pytest -q
```

Run the bundled real-data acceptance summary:

```bash
python scripts/validate_bundled_data.py
```

The validation script uses no network access and writes:

```text
validation_results.json
```

See [`FINAL_BACKEND_VALIDATION.md`](FINAL_BACKEND_VALIDATION.md) for the measured final results and limitations.

## Interpretation boundaries

- **Matched-filter SNR** is GravityHunter's primary single-detector event statistic.
- **H1/L1 coincidence and delay** provide network consistency checks.
- **Spectrogram morphology** is a time-frequency visualization, not the primary detector.
- **Reference waveform overlays** are external model/reference information and are labelled separately from measured strain.
- **Source Parameters** are published metadata, not inferred by this application.
- **Inspiral Diagnostics** is experimental and does not affect detection.
- The fixed project SNR threshold is validated only for the bundled demonstration cases and controls; it is not a published astrophysical-significance threshold.

GravityHunter is an independent course-project signal-analysis implementation, not the production LIGO/Virgo/KAGRA search or Bayesian parameter-estimation pipeline.
