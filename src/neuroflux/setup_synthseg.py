"""
NEURO//FLUX — SynthSeg 2.0 Setup  (setup_synthseg.py)
======================================================
Run once before first use.  Creates an isolated Python venv containing
SynthSeg and its modernised TensorFlow/Keras dependencies -- separate from the
main NEUROFLUX environment to avoid TF version conflicts.

What this script does
  1. Creates synthseg_env/       -- isolated venv (Python 3.10+)
  2. Clones https://github.com/D3njo/SynthSeg into synthseg_repo/
  3. pip-installs SynthSeg + its dependencies inside the venv
     On Apple Silicon (M1/M2/M3) tensorflow-metal is added automatically.
  4. Copies the pre-trained model weights from the local models/ folder
     into synthseg_repo/models/

Usage
  python setup_synthseg.py [--python /path/to/python3.10] [--skip-env] [--skip-models]

  --python     Path to Python 3.10+ interpreter (default: current interpreter)
  --skip-env   Skip venv creation + dependency install (jump straight to models)
  --skip-models  Skip model copy entirely

Python 3.10+ is required for the modernised SynthSeg fork (tensorflow==2.15).

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

SERVER_DIR        = os.path.dirname(os.path.abspath(__file__))
ENV_DIR           = os.path.join(SERVER_DIR, "synthseg_env")
REPO_DIR          = os.path.join(SERVER_DIR, "synthseg_repo")
REPO_URL          = "https://github.com/D3njo/SynthSeg.git"
MODELS_DIR        = os.path.join(REPO_DIR, "models")
LOCAL_MODELS_DIR  = os.path.join(SERVER_DIR, "..", "..", "models")

_RELEASE_BASE = (
    "https://github.com/D3njo/neuroflux/releases/download/v1.0-models"
)

# Model weight files — downloaded from GitHub Releases if not present locally
MODEL_FILES = [
    "synthseg_1.0.h5",
    "synthseg_2.0.h5",
    "synthseg_parc_2.0.h5",
    "synthseg_qc_2.0.h5",
    "synthseg_robust_2.0.h5",
]

# pip requirements for the modernised D3njo/SynthSeg fork inside its isolated venv.
# TF 2.15 is the last release with Keras 2.x (tf.keras) — supports Python 3.10+
# and Apple Silicon via the optional tensorflow-metal plugin.
VENV_REQUIREMENTS = [
    "tensorflow==2.15.*",
    "numpy>=1.24,<2.0",
    "scipy>=1.10",
    "h5py>=3.8",
    "nibabel>=5.0",
    "matplotlib>=3.7",
    "protobuf>=4.21,<5",
]

# On Apple Silicon the Metal plugin enables GPU acceleration
import platform as _platform
if _platform.system() == "Darwin" and _platform.machine() == "arm64":
    VENV_REQUIREMENTS.append("tensorflow-metal>=1.1")


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


def step_copy_models():
    """Provide SynthSeg model weights.

    Strategy (in order):
      1. Local  — copy from the project-root models/ folder if present.
      2. Remote — download from GitHub Releases (requires public repo).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_dir = os.path.normpath(LOCAL_MODELS_DIR)

    print(f"\n[4/4] Providing model weights …")

    any_failed = False
    for fname in MODEL_FILES:
        dest = os.path.join(MODELS_DIR, fname)

        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  [skip] {fname} already present ({os.path.getsize(dest)/1e6:.0f} MB).")
            continue

        # ── 1. Local copy ────────────────────────────────────────────────────
        src = os.path.join(local_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dest)
            print(f"  ✓ {fname} ({os.path.getsize(dest)/1e6:.0f} MB) copied from local folder.")
            continue

        # ── 2. GitHub Releases download ──────────────────────────────────────
        url = f"{_RELEASE_BASE}/{fname}"
        tmp = dest + ".tmp"
        print(f"  Downloading {fname} from GitHub Releases …")
        try:
            def _progress(block, block_size, total):
                if total > 0:
                    pct = min(100, block * block_size * 100 // total)
                    print(f"\r    {fname}: {pct}%", end="", flush=True)
            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
            print()
            if os.path.getsize(tmp) < 1_000_000:
                os.remove(tmp)
                raise RuntimeError(f"Downloaded file too small — is the repo public?")
            os.replace(tmp, dest)
            print(f"  ✓ {fname} ({os.path.getsize(dest)/1e6:.0f} MB) downloaded.")
        except Exception as e:
            if os.path.isfile(tmp):
                os.remove(tmp)
            print(f"  ✗ {fname}: {e}")
            any_failed = True

    if any_failed:
        print()
        print("=" * 60)
        print("ACTION REQUIRED — some model weights could not be obtained")
        print("=" * 60)
        print("Either:")
        print(f"  A) Place the .h5 files manually in: {local_dir}")
        print(f"  B) Ensure https://github.com/D3njo/neuroflux is set to public")
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
        help="Path to Python 3.10+ interpreter (default: current interpreter)",
    )
    p.add_argument(
        "--skip-models", action="store_true",
        help="Skip model weight copy",
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
    import re as _re
    _ver_match = _re.search(r"Python (\d+)\.(\d+)", py_ver)
    if _ver_match:
        _major, _minor = int(_ver_match.group(1)), int(_ver_match.group(2))
        if (_major, _minor) < (3, 10):
            print(
                "\nWARNING: The modernised SynthSeg fork requires Python 3.10+.\n"
                "Your interpreter reports a different version.  Setup will continue\n"
                "but TF installation may fail.  Consider:\n"
                "  python setup_synthseg.py --python /path/to/python3.10\n"
            )

    if not args.skip_env:
        step_create_venv(base_python)
        step_clone_repo()
        step_install_deps()
    else:
        print("[skip-env] Skipping venv / dependency setup.")
    if not args.skip_models:
        step_copy_models()
    step_verify()

    print("\n" + "="*60)
    print("Setup complete.  (SynthSeg fork: D3njo/SynthSeg, TF 2.15)")
    print(f"  venv:   {ENV_DIR}")
    print(f"  repo:   {REPO_DIR}")
    print(f"  models: {MODELS_DIR}")
    print("\nYou can now run:")
    print("  python segment.py <input.nii.gz> [output_dir] [--robust]")
    print("="*60)


if __name__ == "__main__":
    main()
