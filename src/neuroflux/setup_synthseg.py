"""
NEURO//FLUX — SynthSeg 2.0 Setup  (setup_synthseg.py)
======================================================
Run once before first use.  Creates an isolated Python venv containing
SynthSeg and its exact TensorFlow/Keras dependencies -- separate from the
main NEUROFLUX environment to avoid TF version conflicts.

What this script does
  1. Creates synthseg_env/       -- isolated venv (Python 3.8 recommended)
  2. Clones https://github.com/BBillot/SynthSeg into synthseg_repo/
  3. pip-installs SynthSeg + its dependencies inside the venv
  4. Downloads the pre-trained model weights via Docker (primary) or curl
     (fallback) into synthseg_repo/models/

Model weight acquisition strategy
  1. Docker  — extracts from freesurfer/freesurfer:7.4.1 (cross-platform,
               recommended, requires Docker Desktop)
  2. curl    — tries UCL Dropbox / MGH FTP / HuggingFace (links may be stale)
  3. Manual  — prints instructions if both methods fail

Usage
  python setup_synthseg.py [--python /path/to/python3.8] [--skip-env] [--skip-models]

  --python     Path to Python 3.8 interpreter (default: current interpreter)
  --skip-env   Skip venv creation + dependency install (jump straight to models)
  --skip-models  Skip model weight download entirely

Python 3.8 is required for SynthSeg 2.0's tensorflow==2.2.0 dependency.

After setup:
  python segment.py <input.nii.gz>
"""

import argparse
import os
import platform
import subprocess
import sys
import shutil
import urllib.request

SERVER_DIR  = os.path.dirname(os.path.abspath(__file__))
ENV_DIR     = os.path.join(SERVER_DIR, "synthseg_env")
REPO_DIR    = os.path.join(SERVER_DIR, "synthseg_repo")
REPO_URL    = "https://github.com/BBillot/SynthSeg.git"
MODELS_DIR  = os.path.join(REPO_DIR, "models")

# Model weights -- hosted on UCL Dropbox (official SynthSeg distribution)
# See: https://github.com/BBillot/SynthSeg#2-segmenting-your-own-data
MODEL_FILES = [
    # (filename, url)
    (
        "synthseg_2.0.h5",
        "https://www.dropbox.com/s/hrx4y4eeewfd4ld/synthseg_2.0.h5?dl=1",
    ),
    (
        "synthseg_2.0_robust.h5",
        "https://www.dropbox.com/s/2bqf0z2dxnkuucl/synthseg_2.0_robust.h5?dl=1",
    ),
]

# pip requirements for SynthSeg inside its isolated venv
# Exact versions required -- SynthSeg is pinned to TF 2.2 / Keras 2.3
VENV_REQUIREMENTS = [
    "tensorflow==2.2.0",
    "keras==2.3.1",
    "nibabel>=3.2",
    "numpy>=1.19,<1.24",    # TF 2.2 needs numpy < 1.24
    "protobuf==3.20.3",     # TF 2.2 requires protobuf < 4
    "matplotlib",
    "scipy",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    """Run a shell command, streaming output. Raises on non-zero exit."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode})")


def _venv_python():
    if platform.system() == "Windows":
        return os.path.join(ENV_DIR, "Scripts", "python.exe")
    return os.path.join(ENV_DIR, "bin", "python")


def _venv_pip():
    if platform.system() == "Windows":
        return os.path.join(ENV_DIR, "Scripts", "pip.exe")
    return os.path.join(ENV_DIR, "bin", "pip")


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_create_venv(base_python):
    if os.path.isdir(ENV_DIR):
        print(f"[skip] synthseg_env already exists: {ENV_DIR}")
        return
    print(f"\n[1/4] Creating isolated venv in {ENV_DIR} …")
    _run([base_python, "-m", "venv", ENV_DIR])
    print("      venv created.")


def step_clone_repo():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print(f"[skip] synthseg_repo already cloned: {REPO_DIR}")
        return
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    print(f"\n[2/4] Cloning SynthSeg repo …")
    _run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    print("      Clone complete.")


def step_install_deps():
    print(f"\n[3/4] Installing SynthSeg dependencies inside venv …")
    pip = _venv_pip()
    # On Windows, pip cannot upgrade itself directly — use the python -m pip form instead
    _run([_venv_python(), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([_venv_python(), "-m", "pip", "install"] + VENV_REQUIREMENTS)
    # Install SynthSeg itself (no TF — already installed above)
    # Use --config-settings editable_mode=compat to silence legacy setup.py warning
    _run([_venv_python(), "-m", "pip", "install", "-e", REPO_DIR,
          "--no-deps", "--use-pep517",
          "--config-settings", "editable_mode=compat"])
    print("      Dependencies installed.")


def step_download_models():
    """Download SynthSeg model weights.

    Strategy (in order):
      1. Docker  — extracts weights from freesurfer/freesurfer:7.4.1 image.
                   Works on Windows / macOS / Linux without any FreeSurfer install.
      2. curl    — direct URL download (fallback; links may be stale).
      3. Manual  — prints instructions and exits if both methods fail.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Check which files are already present and valid
    needed = []
    for fname, url in MODEL_FILES:
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 10_000_000:
            print(f"[skip] {fname} already present ({os.path.getsize(dest)/1e6:.0f} MB).")
        else:
            needed.append((fname, url))

    if not needed:
        return

    print(f"\n[4/4] Fetching {len(needed)} model weight file(s) …")

    # ── Method 1: Docker ──────────────────────────────────────────────────────
    if shutil.which("docker"):
        print("  Docker found — pulling freesurfer/freesurfer:7.4.1 …")
        print("  (first run downloads ~1.5 GB; subsequent runs use cache)")

        # Make sure the image is available
        pull = subprocess.run(
            ["docker", "pull", "freesurfer/freesurfer:7.4.1"],
            capture_output=False,
        )
        if pull.returncode != 0:
            print("  WARNING: docker pull failed — trying curl fallback …")
        else:
            # Copy each needed file out of the container
            models_in_container = "/usr/local/freesurfer/models"
            copy_cmd = " && ".join(
                f"cp {models_in_container}/{fname} /output/"
                for fname, _ in needed
            )
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{MODELS_DIR}:/output",
                    "freesurfer/freesurfer:7.4.1",
                    "bash", "-c", copy_cmd,
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                # Verify files landed correctly
                ok = all(
                    os.path.isfile(os.path.join(MODELS_DIR, fname))
                    and os.path.getsize(os.path.join(MODELS_DIR, fname)) > 10_000_000
                    for fname, _ in needed
                )
                if ok:
                    for fname, _ in needed:
                        sz = os.path.getsize(os.path.join(MODELS_DIR, fname)) / 1e6
                        print(f"  ✓ {fname} ({sz:.0f} MB) — extracted via Docker.")
                    return
                else:
                    print("  WARNING: Docker ran but files look empty — trying curl …")
            else:
                print(f"  WARNING: Docker copy failed: {result.stderr.strip()[:200]}")
                print("  Trying curl fallback …")
    else:
        print("  Docker not found — trying curl …")

    # ── Method 2: curl / urllib fallback ─────────────────────────────────────
    FALLBACK_URLS = [
        # primary UCL dropbox (may be stale)
        "https://www.dropbox.com/s/hrx4y4eeewfd4ld/{fname}?dl=1",
        "https://www.dropbox.com/s/2bqf0z2dxnkuucl/{fname}?dl=1",
        # MGH FTP
        "https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/SynthSeg/{fname}",
        # HuggingFace
        "https://huggingface.co/freesurfer/synthseg/resolve/main/{fname}",
    ]

    any_failed = False
    for fname, primary_url in needed:
        dest = os.path.join(MODELS_DIR, fname)
        tmp  = dest + ".tmp"

        urls_to_try = [primary_url] + [
            u.replace("{fname}", fname) for u in FALLBACK_URLS
            if u.replace("{fname}", fname) != primary_url
        ]

        downloaded = False
        for url in urls_to_try:
            print(f"  Trying: {url}")
            try:
                def _progress(block, block_size, total):
                    if total > 0:
                        pct = min(100, block * block_size * 100 // total)
                        print(f"\r  {fname}: {pct}%", end="", flush=True)
                urllib.request.urlretrieve(url, tmp, reporthook=_progress)
                print()
                if os.path.getsize(tmp) < 10_000_000:
                    print(f"  File too small ({os.path.getsize(tmp)} bytes) — skipping URL.")
                    os.remove(tmp)
                    continue
                os.rename(tmp, dest)
                print(f"  ✓ {fname} ({os.path.getsize(dest)/1e6:.0f} MB) saved.")
                downloaded = True
                break
            except Exception as e:
                if os.path.isfile(tmp):
                    os.remove(tmp)
                print(f"  Failed: {e}")

        if not downloaded:
            any_failed = True
            print(f"  ✗ Could not download {fname} from any source.")

    # ── Method 3: Manual instructions ────────────────────────────────────────
    if any_failed:
        print()
        print("=" * 60)
        print("ACTION REQUIRED — model weights could not be downloaded")
        print("=" * 60)
        print()
        print("Please obtain the weights manually using one of these methods:")
        print()
        print("Option A — Docker (recommended, cross-platform):")
        print("  1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        print("  2. Re-run this script:  python setup_synthseg.py --skip-env")
        print()
        print("Option B — FreeSurfer (macOS/Linux):")
        print("  1. Download FreeSurfer 7.4.1: https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads")
        print("  2. Copy weights:")
        print(f"     cp $FREESURFER_HOME/models/synthseg_2.0.h5         {MODELS_DIR}/")
        print(f"     cp $FREESURFER_HOME/models/synthseg_2.0_robust.h5  {MODELS_DIR}/")
        print()
        print(f"Both files must be placed in: {MODELS_DIR}")
        print("=" * 60)
        sys.exit(1)
def step_verify():
    print("\n[verify] Running a quick import test inside the venv …")
    code = (
        "import sys; "
        "sys.path.insert(0, repr(REPO_DIR)); "
        "from SynthSeg.predict import predict; "
        "print('SynthSeg import OK')"
    ).replace("repr(REPO_DIR)", repr(REPO_DIR))
    result = subprocess.run(
        [_venv_python(), "-c", code],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  SynthSeg import OK.")
    else:
        print("  WARNING: import test failed:")
        print("  stdout:", result.stdout.strip())
        print("  stderr:", result.stderr.strip()[:400])
        print("  Setup may still work — check manually with:")
        print(f"    {_venv_python()} -c 'from SynthSeg.predict import predict'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="NEUROFLUX — SynthSeg 2.0 Setup")
    p.add_argument(
        "--python", default=sys.executable,
        help="Path to Python 3.8 interpreter (default: current interpreter)",
    )
    p.add_argument(
        "--skip-models", action="store_true",
        help="Skip model weight download (use if you supply weights manually)",
    )
    p.add_argument(
        "--skip-env", action="store_true",
        help="Skip venv creation and dependency install (jump straight to model download)",
    )
    args = p.parse_args()

    base_python = args.python
    print(f"Using base Python: {base_python}")
    _ver_result = subprocess.run(
        [base_python, "--version"], capture_output=True, text=True
    )
    py_ver = _ver_result.stdout.strip() or _ver_result.stderr.strip()
    print(f"Version: {py_ver}")
    if "3.8" not in py_ver:
        print(
            "\nWARNING: SynthSeg 2.0 requires Python 3.8 for tensorflow==2.2.0.\n"
            "Your interpreter reports a different version.  Setup will continue\n"
            "but TF installation may fail.  Consider:\n"
            "  python setup_synthseg.py --python /path/to/python3.8\n"
        )

    if not args.skip_env:
        step_create_venv(base_python)
        step_clone_repo()
        step_install_deps()
    else:
        print("[skip-env] Skipping venv / dependency setup.")
    if not args.skip_models:
        step_download_models()
    step_verify()

    print("\n" + "="*60)
    print("Setup complete.")
    print(f"  venv:   {ENV_DIR}")
    print(f"  repo:   {REPO_DIR}")
    print(f"  models: {MODELS_DIR}")
    print("\nYou can now run:")
    print("  python segment.py <input.nii.gz> [output_dir] [--robust]")
    print("="*60)


if __name__ == "__main__":
    main()
