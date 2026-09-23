# GravityHunter local data folder

The Streamlit app is now folder-driven. During the presentation it does **not** require file uploads and does not need live GWOSC access.

Run once while online:

```bash
python scripts/download_demo_data.py
```

or on Windows double-click:

```text
setup_data.bat
```

Expected layout after downloading:

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

The event files are public 32-second, 4-kHz H1/L1 GWOSC strain releases. The template files are the plus/cross waveform files used in the public LOSC/GWOSC BBH tutorial repository.

The dropdown also exposes two **quiet/no-merger controls**. These are not fake arrays: GravityHunter takes off-source time windows from the real GW150914/GW170104 HDF5 files and runs the same DSP/detection code on them.
