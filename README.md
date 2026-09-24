# GravityHunter — Gravitational-Wave Signal Analysis

GravityHunter is a local gravitational-wave signal-analysis application built around public GWOSC strain data. It runs the same DSP pipeline for selected compact-binary merger records and off-source controls, with a synthetic injection case retained for regression validation.

## What the current version does

The sidebar can select:

- **GW150914** — real H1 + L1 strain
- **GW151226** — real H1 + L1 strain
- **GW170104** — real H1 + L1 strain
- **Quiet real data before GW150914** — negative/no-merger control
- **Quiet real data after GW170104** — negative/no-merger control
- **Synthetic validation chirp** — controlled algorithm test

For a real event, the same backend executes:

```text
local H1/L1 HDF5
        ↓
strain + metadata loading
        ↓
off-source Welch PSD
        ↓
notch filtering
        ↓
band-pass filtering
        ↓
PSD whitening
        ↓
STFT / spectrogram
        ↓
public plus/cross waveform template
        ↓
PSD-weighted two-quadrature matched filter
        ↓
SNR vs time
        ↓
H1/L1 candidate times
        ↓
cross-correlation delay
        ↓
coincidence result
```

The quiet cases go through the **same** processing path. They are not hard-coded blank examples.

## Analysis views

### Overview
Overall status, H1/L1 SNR summary, and selected real-data region.

### Waveform Morphology
Waveform-level morphology view:
- public reference waveform with approximate **Inspiral → Merger → Ringdown** regions;
- real whitened detector data in the same event neighborhood;
- real spectrogram with the reference instantaneous-frequency ridge overlaid.

The stage boundaries are visualization aids, not a full GR parameter-estimation result.

### Event Timeline
A browser-side synchronized animation for real cases:
- plays the complete selected detector record on a master clock;
- shows live **record time + absolute GPS time**;
- can synchronize the collision to GravityHunter's detected time or the catalog reference time;
- uses reference GW phase/frequency to drive the schematic orbital motion;
- uses a Newtonian inspiral separation estimate only before merger;
- transitions two compact bodies into one remnant and a damped ringdown visualization;
- shows the real processed detector strain and matched-filter SNR with moving cursors;
- supports pause, scrub, jump-to-merger, full-record playback, and 5×/20×/50× slow-motion merger replay;
- quiet/no-event cases deliberately do **not** trigger a collision.

The fabric/orbit is a schematic explanatory visualization, not a numerical GR/spacetime simulation.

### Detector Strain
Raw strain and frequency-domain content.

### Spectrum & Noise
Welch PSD / ASD used to characterize colored detector noise.

### Signal Conditioning
Before/after filtering and whitening comparisons.

### Time–Frequency Analysis
Real STFT spectrogram.

### Event Detection
Quantitative event-candidate view: PSD-weighted template match, SNR threshold, candidate time, and timing error relative to the catalog reference.

### Detector Coincidence
H1/L1 detection status, candidate-time coincidence, and cross-correlation arrival-delay estimate.

### Source Parameters
Published/reference event metadata:
- primary black-hole mass;
- secondary black-hole mass;
- chirp mass;
- final/remnant mass;
- approximate radiated mass-energy;
- luminosity distance;
- sky-localization area;
- published network SNR.

GravityHunter does not infer the full astrophysical posterior. **Chirp-Mass Inference** provides one leading-order signal-derived source-parameter estimate from the inspiral frequency evolution.

### Chirp-Mass Inference
Advanced DSP inference from the selected real detector:

```text
H1/L1 whitened strain
→ dedicated inspiral STFTs
→ per-detector background normalization
→ combined H1/L1 relative-power map
→ template-guided local search corridor
→ ridge frequencies selected from measured network STFT power
→ smooth f(t)
→ robust fit of t = tc − A f^(-8/3)
→ leading-order chirp-mass estimate
```

The page shows the combined H1/L1 time-frequency power map, extracted ridge, `f(t)`, `df/dt` diagnostic, fitted chirp mass, template-reference chirp mass, and catalog source-frame chirp mass. This is a leading-order inference method, not a replacement for full Bayesian parameter estimation.

### Audio Reconstruction
Audio generated from the same processed detector time series. Includes direct playback and an optional +400 Hz frequency-shifted version for easier listening.

### Synthetic Validation
Synthetic controlled experiment retained for testing the algorithm while varying signal strength.

## One-time setup

Create and activate a Python environment if desired, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the configured local data once while online:

```bash
python scripts/download_demo_data.py
```

On Windows you can instead double-click:

```text
setup_data.bat
```

This creates:

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
└── templates/
    ├── GW150914_4_template.hdf5
    ├── GW151226_4_template.hdf5
    └── GW170104_4_template.hdf5
```

Once those files exist, the application runs offline.

## Run

```bash
python -m streamlit run streamlit_app.py
```

or on Windows:

```text
run_local.bat
```

Then open the localhost address printed by Streamlit, normally:

```text
http://localhost:8501
```

## Tests

```bash
pytest -q
```

The tests include:
- manual DFT vs NumPy FFT;
- direct vs FFT correlation;
- confusion metrics;
- an end-to-end GWOSC-shaped mock HDF5 real-data path;
- H1/L1 injected delay recovery;
- template loading;
- sonification helper;
- chirp-mass formula;
- recovery of a known physical chirp mass from an analytic inspiral ridge;
- animation payload/GPS synchronization generation;
- browser animation JavaScript syntax checking during development;
- chirp-mass real-pipeline execution on GWOSC-shaped mock HDF5.

## Important interpretation

- **Spectrogram**: visually shows time-frequency evolution; a rising ridge is the chirp/inspiral signature.
- **Matched-filter SNR**: the local quantitative candidate statistic used by GravityHunter.
- **Cross-correlation**: estimates relative H1/L1 lag; it is not the primary event-detection statistic.
- **Waveform Morphology**: combines measured conditioned strain with a clearly labelled public reference template to separate observed data from reference inspiral/merger/ringdown morphology.
- **Quiet controls**: show what happens when the same detector code is applied where no catalog merger is expected.

GravityHunter is an independent signal-analysis prototype and is not the production LIGO/Virgo/KAGRA search or parameter-estimation pipeline.


