# Bundled GravityHunter data

This directory contains the exact local GWOSC strain/template files used in the final backend validation. The application requires no upload and no network access for the configured cases.

```text
data/
├── GW150914/H1.hdf5
├── GW150914/L1.hdf5
├── GW151226/H1.hdf5
├── GW151226/L1.hdf5
├── GW170104/H1.hdf5
├── GW170104/L1.hdf5
├── templates/GW150914_4_template.hdf5
├── templates/GW151226_4_template.hdf5
├── templates/GW170104_4_template.hdf5
└── SHA256SUMS.txt
```

The two quiet controls exposed by the app are off-source slices of these same real detector records; they are not generated noise.

`SHA256SUMS.txt` records the exact files accepted in the final validation pass.
