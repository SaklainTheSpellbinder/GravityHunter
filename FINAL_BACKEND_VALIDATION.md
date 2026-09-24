# GravityHunter — Final Real-Data Backend Validation

Validation date: 2026-09-24

This report records the final acceptance pass performed against the **bundled real GWOSC H1/L1 strain files and public reference waveform templates**. No network download is required to reproduce these checks.

## Scope

The validation covers the core signal-processing path:

```text
HDF5 strain + GPS metadata
→ off-source Welch PSD
→ conditioning / whitening / STFT
→ PSD-weighted plus/cross matched filtering
→ SNR candidate time
→ H1/L1 coincidence
→ cross-correlation delay
→ quiet-data negative controls
```

The separate ridge-based chirp-mass estimator is treated as an **experimental diagnostic**, not a core detection criterion.

## Important timing correction made during validation

The public GWOSC tutorial waveform files use the **template midpoint / end-time convention** for matched-filter trigger timing. The final code therefore uses:

```python
template.reference_index == template.n // 2
```

for matched-filter candidate timing. The exact strain-amplitude peak is retained only for morphology/phase visualization.

Using the amplitude peak as the matched-filter time reference caused millisecond-to-tens-of-milliseconds offsets and was incorrect for these public tutorial templates.

## Bundled file integrity / structure

All six real strain files were opened with `h5py`, verified to contain `/strain/Strain`, and contain:

- 131,072 samples
- 4,096 Hz sample rate
- 32 s duration
- finite calibrated strain samples
- detector and GPS-start metadata

The final bundle also contains `data/SHA256SUMS.txt` so the exact locally validated binary files can be checked after copying the project.

## Real-event acceptance results

### GW150914

Reference synchronization GPS: **1126259462.44**

| Result | GravityHunter |
|---|---:|
| H1 peak GPS | 1126259462.439453 |
| H1 SNR | 17.923 |
| L1 peak GPS | 1126259462.432617 |
| L1 SNR | 12.743 |
| H1 − L1 candidate-time difference | +6.836 ms |
| waveform cross-correlation delay | +7.324 ms |
| network coincidence | PASS |

The recovered trigger times and SNRs closely reproduce the public GWOSC tutorial result for this single-template analysis (approximately H1 18.6 at 1126259462.4395 and L1 13.2 at 1126259462.4324).

### GW151226

Reference synchronization GPS: **1135136350.65**

| Result | GravityHunter |
|---|---:|
| H1 peak GPS | 1135136350.647461 |
| H1 SNR | 7.478 |
| L1 peak GPS | 1135136350.646729 |
| L1 SNR | 5.971 |
| H1 − L1 candidate-time difference | +0.732 ms |
| waveform cross-correlation delay | +0.977 ms |
| network coincidence | PASS |

The relative H1/L1 timing is consistent with the published approximately millisecond-scale inter-site arrival difference for this event.

### GW170104

Reference synchronization GPS: **1167559936.60**

| Result | GravityHunter |
|---|---:|
| H1 peak GPS | 1167559936.608398 |
| H1 SNR | 8.466 |
| L1 peak GPS | 1167559936.611084 |
| L1 SNR | 9.134 |
| H1 − L1 candidate-time difference | −2.686 ms |
| waveform cross-correlation delay | −3.174 ms |
| network coincidence | PASS |

The recovered trigger times closely reproduce the public GWOSC tutorial single-template values (approximately H1 7.8 at 1167559936.6084 and L1 9.5 at 1167559936.6113).

## Negative controls

At the fixed default local candidate threshold of **5.5**:

| Off-source control | H1 max SNR | L1 max SNR | Coincident candidate |
|---|---:|---:|---|
| quiet interval from GW150914 file | 4.146 | 3.931 | No |
| quiet interval from GW170104 file | 4.602 | 4.495 | No |

The threshold therefore separates all three selected real events from both bundled off-source controls in this demonstration dataset. It is a **project operating threshold**, not a LIGO astrophysical-significance threshold.

## Off-source behavior inside the event records

The largest valid matched-filter responses more than 2 s away from the known events remain below 5.5 in all six event-detector channels. This guards against the previous 0 s / 32 s edge-artifact failure mode.

## Whitening checks

The whitened time series are finite, zero-centered/unit-standardized for display, and their central 80% PSD spread inside the analysis band is:

| Event | H1 spread | L1 spread |
|---|---:|---:|
| GW150914 | 5.31 dB | 4.98 dB |
| GW151226 | 2.30 dB | 2.80 dB |
| GW170104 | 2.12 dB | 2.17 dB |

A perfectly flat finite-sample PSD is neither expected nor required; these values confirm that the strong original color is substantially flattened without NaN/Inf failures.

## Matched-filter logic verified

The final implementation:

1. estimates a one-sided PSD density from off-source data;
2. tapers raw data and public plus/cross templates;
3. applies inverse-PSD frequency weighting directly rather than double filtering the detector statistic;
4. normalizes each template using the same noise-weighted inner product;
5. accounts for the actual plus/cross template overlap using a 2×2 Gram-matrix statistic rather than assuming perfect orthogonality;
6. excludes partial-overlap correlation lags;
7. uses the public template midpoint/end-time convention for trigger timestamps;
8. searches the complete valid record rather than selecting a peak using the reference event time.

The reference event time is used for **validation, PSD event exclusion, and display synchronization**, not for selecting the matched-filter maximum.

## Detector-network logic verified

- Candidate coincidence uses absolute GPS timestamps where available.
- The H1/L1 coincidence window is 10 ms.
- Cross-correlation is performed around the independently recovered candidates, not around the catalog/reference time.
- The cross-correlation search is restricted to ±10 ms.
- Absolute correlation magnitude is used because H1/L1 antenna response can invert waveform sign.

## Experimental chirp-mass diagnostic

The current STFT-ridge estimator is **not promoted as a validated source-parameter measurement**.

With the default event analysis bands:

- GW150914: too few reliable ridge points;
- GW151226: mathematical fit converges to about 11.97 M☉, but this disagrees materially with the template-associated mass (~9.72 M☉), so the UI labels it experimental/model-discrepant;
- GW170104: too few reliable ridge points.

This limitation does **not** affect event detection, SNR timing, H1/L1 coincidence, or delay estimation. A proper small template-bank parameter search is the recommended next step if a robust measured chirp-mass estimate is required.

## Automated tests

The final bundle contains both synthetic/unit tests and real-data regression tests. Run:

```bash
pytest -q
```

The real-data tests verify:

- all three events are recovered in H1 and L1;
- recovered SNRs stay within stable expected ranges;
- trigger times stay near the reference event neighborhood;
- H1/L1 delays have the correct sign and millisecond scale;
- both bundled quiet controls remain below the default threshold;
- whitening output remains finite and reasonably flat.

A separate offline validation command is available:

```bash
python scripts/validate_bundled_data.py
```

It writes `validation_results.json` and uses only bundled local files.

## Final backend assessment

**Core detector: validated for the bundled project cases.**

The real-data event-detection backend now behaves consistently with the public GWOSC tutorial-level analysis for the selected data/template set. It remains an independent course-project signal-analysis implementation, not the production LVK search or parameter-estimation pipeline.
