"""
NEURO//FLUX — Model Weight Setup  (setup_synthseg.py)
======================================================
Downloads the SynthSeg 2.0 pre-trained model weights into the models/ folder.
SynthSeg is now bundled inside NeuroFlux — no separate venv or git clone needed.

Usage
  neuroflux-setup
  python -m neuroflux.setup_synthseg [--skip-models]

  --skip-models  Dry-run: check paths but do not download anything.

Model weights are NOT shipped with the package (they are ~600 MB in total).
They are downloaded from GitHub Releases on first run.
"""

import argparse
import os
import pathlib
import sys
import urllib.request

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = pathlib.Path(__file__).parent
_MODELS_DIR = _HERE.parent.parent / "models"   # repo_root/models/

_RELEASE_BASE = (
    "https://github.com/D3njo/neuroflux/releases/download/v1.0-models"
)

MODEL_FILES = [
    "synthseg_1.0.h5",
    "synthseg_2.0.h5",
    "synthseg_parc_2.0.h5",
    "synthseg_qc_2.0.h5",
    "synthseg_robust_2.0.h5",
]


# ── Download helper ───────────────────────────────────────────────────────────

def _download_models(models_dir: pathlib.Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    any_failed = False

    for fname in MODEL_FILES:
        dest = models_dir / fname

        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"  [skip] {fname} already present ({dest.stat().st_size / 1e6:.0f} MB).")
            continue

        url = f"{_RELEASE_BASE}/{fname}"
        tmp = pathlib.Path(str(dest) + ".tmp")
        print(f"  Downloading {fname} …")
        try:
            def _progress(block, block_size, total):
                if total > 0:
                    pct = min(100, block * block_size * 100 // total)
                    print(f"\r    {fname}: {pct}%", end="", flush=True)

            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
            print()

            if tmp.stat().st_size < 1_000_000:
                tmp.unlink()
                raise RuntimeError("Downloaded file too small — is the release public?")

            tmp.replace(dest)
            print(f"  ✓ {fname} ({dest.stat().st_size / 1e6:.0f} MB) downloaded.")
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            print(f"  ✗ {fname}: {e}")
            any_failed = True

    if any_failed:
        print()
        print("=" * 60)
        print("ACTION REQUIRED — some model weights could not be downloaded.")
        print("=" * 60)
        print(f"Place the .h5 files manually in:\n  {models_dir}")
        print("=" * 60)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="NEUROFLUX — Download SynthSeg 2.0 model weights"
    )
    p.add_argument(
        "--skip-models", action="store_true",
        help="Dry-run: check paths but skip download",
    )
    args = p.parse_args()

    models_dir = _MODELS_DIR.resolve()
    print(f"Model directory: {models_dir}")

    if args.skip_models:
        print("[skip-models] No downloads performed.")
    else:
        _download_models(models_dir)

    print("\n" + "=" * 60)
    print("Setup complete.  SynthSeg is bundled inside NeuroFlux.")
    print(f"  models: {models_dir}")
    print("\nYou can now run:")
    print("  neuroflux-segment <input.nii.gz> [output_dir] [--robust]")
    print("=" * 60)


if __name__ == "__main__":
    main()
