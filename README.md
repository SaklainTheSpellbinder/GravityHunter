# GravityHunter — Core DSP Code (Modules 1–12)

This bundle covers the concepts learned so far:

1. sampled time series / HDF5 loading
2. DFT / FFT / frequency axes
3. windowing and leakage
4. Welch PSD / ASD
5. band-pass and notch filtering
6. PSD-based whitening
7. STFT / spectrogram
8. direct and FFT-based correlation
9. PSD-weighted matched filtering
10. SNR / peak detection
11. H1/L1 delay and coincidence utilities
12. waveform injections and evaluation metrics

## Install

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
```

## Run tests

From the project root:

```bash
pytest -q
```

## Run the synthetic end-to-end demo

```bash
python -m examples.synthetic_pipeline
```

Start with this before real LIGO data. It verifies that the pipeline can find
a known chirp injected into noisy data.

## Real-data workflow

Use:

```bash
python
```

and inspect your HDF5 file first:

```python
from gravityhunter.dsp.loader import inspect_hdf5

for line in inspect_hdf5("data/your_file.hdf5"):
    print(line)
```

Then load it using the matching dataset hierarchy.

The convenience `load_common_gwosc_hdf5()` assumes the common layout:

```text
/strain/Strain
```

with sample spacing metadata. If your file differs, use the generic loader.

## Recommended workflow

```text
local HDF5
    ↓
load calibrated strain x[n]
    ↓
confirm fs, N, duration, time interval
    ↓
raw waveform
    ↓
FFT / frequency axis
    ↓
Welch PSD / ASD from a representative noise region
    ↓
inspect noise lines
    ↓
notch selected narrow lines
    ↓
band-pass useful analysis band
    ↓
whiten using the noise PSD
    ↓
STFT / spectrogram
    ↓
load or generate a compatible template
    ↓
PSD-weighted matched filter
    ↓
SNR versus lag/time
    ↓
peak + threshold
    ↓
single-detector candidate
    ↓
repeat independently for H1 and L1
    ↓
cross-correlation / candidate-time comparison
    ↓
relative delay
    ↓
coincidence
    ↓
multi-detector candidate
```

## Important scientific cautions

- A filter is not a detector.
- Whitening does not remove noise; it flattens frequency-dependent noise.
- STFT is primarily a time-frequency visualization/analysis tool.
- Matched filtering is the main single-detector detection statistic here.
- A high single-detector SNR is not proof of an astrophysical event.
- Thresholds should be justified empirically on event and no-event data.
- The code is an educational detector, not the LIGO production pipeline.

## Local Streamlit frontend

A basic working frontend is included in `streamlit_app.py`.

Run from the project root:

```bash
streamlit run streamlit_app.py
```

Then open the localhost address printed by Streamlit (normally `http://localhost:8501`).

The UI uses a dark navy night-sky theme with cream star details and includes:

- mission-control overview
- waveform + FFT
- PSD / ASD
- filtering + whitening
- STFT spectrogram
- matched-filter SNR + threshold
- synthetic H1/L1 delay + coincidence
- signal-injection stress lab
- real GWOSC-style HDF5 waveform/FFT/PSD preview

The synthetic defaults are demonstration settings. Real detector filtering bands, line notches, thresholding, and templates should be chosen from the actual data rather than copied blindly from the synthetic example.
