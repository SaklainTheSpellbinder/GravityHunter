"""Download the local datasets used by the GravityHunter presentation.

Run once while online:
    python scripts/download_demo_data.py

After that, Streamlit reads only from ./data and no upload is required.
Use --force to replace already-downloaded files.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FILES = {
    # Real strain: official GWOSC event releases, 32 s at 4096 Hz.
    "GW150914/H1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "GW150914/L1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
    "GW151226/H1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/H-H1_GWOSC_4KHZ_R1-1135136335-32.hdf5",
    "GW151226/L1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/L-L1_GWOSC_4KHZ_R1-1135136335-32.hdf5",
    "GW170104/H1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW170104/v2/H-H1_GWOSC_4KHZ_R1-1167559921-32.hdf5",
    "GW170104/L1.hdf5": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW170104/v2/L-L1_GWOSC_4KHZ_R1-1167559921-32.hdf5",
    # Educational SEOBNRv2-style plus/cross template files used by the public
    # GWOSC/LOSC BBH tutorial repository.
    "templates/GW150914_4_template.hdf5": "https://raw.githubusercontent.com/gwosc-tutorial/LOSC_Event_tutorial/master/GW150914_4_template.hdf5",
    "templates/GW151226_4_template.hdf5": "https://raw.githubusercontent.com/gwosc-tutorial/LOSC_Event_tutorial/master/GW151226_4_template.hdf5",
    "templates/GW170104_4_template.hdf5": "https://raw.githubusercontent.com/gwosc-tutorial/LOSC_Event_tutorial/master/GW170104_4_template.hdf5",
}


def _validate_hdf5(path: Path) -> None:
    """Fail early if an HTML error page or interrupted file was saved as .hdf5."""
    with h5py.File(path, "r") as f:
        if not ("strain" in f or "template" in f):
            raise ValueError(f"{path.name} does not look like a supported GravityHunter HDF5 file")


def download(relpath: str, url: str, force: bool = False) -> None:
    target = DATA / relpath
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        try:
            _validate_hdf5(target)
            print(f"✓ {relpath} already exists and is readable")
            return
        except Exception:
            print(f"! {relpath} exists but is invalid; downloading again")

    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()

    print(f"↓ {relpath}\n  {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 GravityHunter-course-project/1.0",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            done = 0
            block_size = 1024 * 256
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(
                        f"\r    {pct:6.2f}%  {done/1e6:7.2f}/{total/1e6:7.2f} MB"
                    )
                    sys.stdout.flush()
            if total:
                print()

        _validate_hdf5(part)
        part.replace(target)
        print(f"  saved → {target.relative_to(ROOT)}")
    except Exception:
        if part.exists():
            part.unlink()
        raise


def main() -> int:
    force = "--force" in sys.argv
    DATA.mkdir(exist_ok=True)
    failures: list[tuple[str, str, Exception]] = []

    for relpath, url in FILES.items():
        try:
            download(relpath, url, force=force)
        except Exception as exc:
            failures.append((relpath, url, exc))
            print(f"ERROR downloading {relpath}: {exc}")
            print("You may download the URL manually and save it at the shown path.\n")

    if failures:
        print("\nSome files failed:\n")
        for relpath, url, exc in failures:
            print(f"  {relpath}\n    {url}\n    {exc}\n")
        return 1

    print("\nAll GravityHunter demo data is ready.")
    print("Start the app with: python -m streamlit run streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
