"""
NEURO//FLUX — Brain Segmentation Pipeline v1.6  (SynthSeg 2.0)
===============================================================
Cross-platform (Windows / macOS / Linux).
No FreeSurfer installation required.
No ANTs / antspynet / TensorFlow dependency in the main process.

Backend: SynthSeg 2.0 (Billot et al., Harvard/MGH, PNAS 2023)
  - Contrast- and resolution-agnostic 3-D U-Net
  - Domain-randomisation training: works out-of-the-box on any MRI contrast,
    any resolution, without retraining or fine-tuning
  - 95 FreeSurfer structures -> mapped to 7-class tissue + 10-class hemi
  - --robust model for clinical / low-SNR / large-spacing scans
  - QC scores and regional volumes written alongside NIfTI outputs
  - GPU auto-detected by SynthSeg; graceful CPU fallback

One-time setup
  python setup_synthseg.py

Pipeline
  SynthSeg predict -> label remap -> hemi split -> save outputs + summary.json

CLI
  python segment.py <input.nii[.gz]> [output_dir] [--robust] [--fast] [--threads N]

JSON protocol (stdout)
  {"step": "synthseg", "pct": 30, "msg": "..."}
  {"status": "done",  "outputs": {"original": "...", "seg_full": "...", ...}}
  {"status": "error", "msg": "..."}
"""

import sys
import os
import json
import time
import traceback
import argparse
import platform
import subprocess
import shutil
import tempfile
import threading

import numpy as np
import nibabel as nib

# Label mapping lives in labels.py next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import fs_to_tissue, fs_to_hemi, TISSUE_NAMES, HEMI_NAMES


# ── Progress / status emitters (same JSON protocol as v3.x) ──────────────────

def _emit(step, pct, msg):
    print(json.dumps({"step": step, "pct": pct, "msg": msg}), flush=True)

def _done(outputs):
    print(json.dumps({"status": "done", "outputs": outputs}), flush=True)

def _error(msg):
    print(json.dumps({"status": "error", "msg": msg}), flush=True)


# ── SynthSeg environment paths ────────────────────────────────────────────────

_SERVER_DIR    = os.path.dirname(os.path.abspath(__file__))
_SS_ENV        = os.path.join(_SERVER_DIR, "synthseg_env")
_SS_REPO       = os.path.join(_SERVER_DIR, "synthseg_repo")
_SS_SCRIPT     = os.path.join(_SS_REPO, "scripts", "commands", "SynthSeg_predict.py")


def _ss_python():
    """Path to the Python interpreter inside the SynthSeg venv."""
    if platform.system() == "Windows":
        return os.path.join(_SS_ENV, "Scripts", "python.exe")
    return os.path.join(_SS_ENV, "bin", "python")


def _check_env():
    """Raise a descriptive error if the SynthSeg venv is missing."""
    py  = _ss_python()
    scr = _SS_SCRIPT
    ok  = os.path.isfile(py) and os.path.isfile(scr)
    if not ok:
        missing = []
        if not os.path.isfile(py):  missing.append(f"Python env:  {py}")
        if not os.path.isfile(scr): missing.append(f"Script:      {scr}")
        raise RuntimeError(
            "SynthSeg environment not found.\n"
            "Run:  python setup_synthseg.py\n"
            "Missing:\n" + "\n".join(f"  {m}" for m in missing)
        )


# ── SynthSeg subprocess ───────────────────────────────────────────────────────

def _run_synthseg(
    input_path, seg_path, resampled_path,
    posteriors_path, volumes_path, qc_path,
    robust=False, fast=False, threads=1, use_ct=False,
):
    """
    Call SynthSeg_predict.py inside its isolated venv as a subprocess.

    All output paths are passed explicitly so we never depend on SynthSeg's
    internal naming conventions.  stderr is drained in a background thread
    to prevent OS pipe-buffer deadlock on long Python tracebacks.
    """
    cmd = [
        _ss_python(), _SS_SCRIPT,
        "--i",        input_path,
        "--o",        seg_path,
        "--resample", resampled_path,
        "--post",     posteriors_path,
        "--vol",      volumes_path,
        "--qc",       qc_path,
        "--threads",  str(threads),
    ]
    if robust:
        cmd.append("--robust")
    if fast and not robust:   # SynthSeg ignores --fast when --robust is active
        cmd.append("--fast")
    if use_ct:
        cmd.append("--ct")

    mode = "robust" if robust else "standard"
    _emit("synthseg", 8, f"SynthSeg 2.0 ({mode} mode, {threads} thread(s))…")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Drain stderr in background -- avoids OS pipe-buffer deadlock
    stderr_lines = []
    def _drain():
        for line in proc.stderr:
            stderr_lines.append(line)
    drain_t = threading.Thread(target=_drain, daemon=True)
    drain_t.start()

    # Forward SynthSeg stdout as progress events
    # SynthSeg prints things like: "predicting 1/1"  "saving seg..."
    _pct_map = {
        "predicting": 30,
        "saving":     70,
        "done":       85,
    }
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip()
        if not line:
            continue
        pct = next(
            (v for k, v in _pct_map.items() if k in line.lower()),
            None,
        )
        if pct is not None:
            _emit("synthseg", pct, line.strip())

    proc.wait()
    drain_t.join(timeout=5)

    if proc.returncode != 0:
        stderr_out = "".join(stderr_lines)[:800]
        raise RuntimeError(
            f"SynthSeg exited with code {proc.returncode}.\n"
            f"Stderr (last 800 chars):\n{stderr_out}"
        )


# ── Post-processing ───────────────────────────────────────────────────────────

def _remap_and_split(fs_seg_path, resampled_path, output_dir, input_path=None):
    """
    Load SynthSeg's FreeSurfer label volume, remap to NEUROFLUX classes,
    compute hemispheric segmentation, save both NIfTIs.

    Returns (original_path, seg_full_path, seg_hemi_path).
    """
    _emit("remap", 86, "Remapping FreeSurfer labels to NEUROFLUX tissue classes…")

    img    = nib.load(fs_seg_path)
    fs_arr = np.asarray(img.dataobj, dtype=np.int32)

    tissue_arr = fs_to_tissue(fs_arr)
    hemi_arr   = fs_to_hemi(fs_arr, tissue_arr)

    affine = img.affine
    header = img.header.copy()

    def _save(arr, name):
        path    = os.path.join(output_dir, name)
        out_img = nib.Nifti1Image(arr, affine, header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, path)
        return path

    # Log tissue volumes
    for lbl, name in TISSUE_NAMES.items():
        vox = int((tissue_arr == lbl).sum())
        _emit("remap", 88, f"  {name:12s}: {vox:>10,} vox")

    seg_full_path = _save(tissue_arr, "seg_full.nii.gz")
    _emit("remap", 90, "seg_full.nii.gz saved.")

    # Log hemi volumes
    for lbl, name in HEMI_NAMES.items():
        vox = int((hemi_arr == lbl).sum())
        _emit("hemi", 93, f"  {name:6s}: {vox:>10,} vox")

    seg_hemi_path = _save(hemi_arr, "seg_hemi.nii.gz")
    _emit("hemi", 95, "seg_hemi.nii.gz saved.")

    # "original" = the 1 mm isotropic T1 resampled by SynthSeg (--resample)
    original_path = os.path.join(output_dir, "original.nii.gz")
    if resampled_path and os.path.isfile(resampled_path):
        if os.path.abspath(resampled_path) != os.path.abspath(original_path):
            shutil.copy2(resampled_path, original_path)
    elif not os.path.isfile(original_path):
        # Fallback: copy the raw T1 input (resampled file not available)
        fallback = input_path if (input_path and os.path.isfile(input_path)) else None
        if fallback:
            shutil.copy2(fallback, original_path)

    return original_path, seg_full_path, seg_hemi_path


# ── QC / volume summary ───────────────────────────────────────────────────────

def _write_summary(volumes_path, qc_path, output_dir, elapsed_sec, robust, threads):
    """
    Merge SynthSeg's volumes CSV + QC CSV into a single summary.json.
    Returns path to summary.json.
    """
    import csv

    def _read_csv(path):
        if not path or not os.path.isfile(path):
            return {}
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return {}
        headers, values = rows[0], rows[1]
        result = {}
        for h, v in zip(headers[1:], values[1:]):   # skip subject-name column
            try:
                result[h] = float(v)
            except ValueError:
                result[h] = v
        return result

    summary = {
        "backend":   "SynthSeg 2.0",
        "mode":      "robust" if robust else "standard",
        "threads":   threads,
        "elapsed_s": round(elapsed_sec, 1),
        "volumes":   _read_csv(volumes_path),
        "qc_scores": _read_csv(qc_path),
    }

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path,
    output_dir=None,
    robust=False,
    fast=False,
    threads=1,
    use_ct=False,
):
    """
    Run the full NEUROFLUX v1.6 segmentation pipeline.

    Parameters
    ----------
    input_path : str        Path to a NIfTI file (.nii or .nii.gz).
    output_dir : str|None   Output directory.  Defaults to same dir as input.
    robust     : bool       Use SynthSeg-robust (clinical / low-SNR scans).
    fast       : bool       Disable some SynthSeg post-processing (~2x speed).
                            Ignored when robust=True.
    threads    : int        CPU threads for SynthSeg inference.
    use_ct     : bool       True for CT scans (clips to [0, 80] HU).

    Returns
    -------
    dict  keys: original, seg_full, seg_hemi, summary, seg_fs
    """
    t0 = time.time()

    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    def out(name):
        return os.path.join(output_dir, name)

    # ── 1. Pre-flight ─────────────────────────────────────────────────────────
    _emit("setup", 2, "Checking SynthSeg environment…")
    _check_env()

    try:
        img_check = nib.load(input_path)
        shape     = img_check.shape
        spacing   = tuple(round(float(v), 3) for v in img_check.header.get_zooms()[:3])
        ndim      = len(shape)
    except Exception as e:
        raise ValueError(f"Cannot read NIfTI: {e}") from e

    if ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI, got {ndim}-D: {input_path}")

    _emit("setup", 4, (
        f"Input: {os.path.basename(input_path)}  "
        f"shape={shape}  voxel={spacing} mm"
    ))

    # ── 2. SynthSeg inference ─────────────────────────────────────────────────
    # Intermediate files go into a tempdir; only the final remapped files
    # are written to output_dir to keep it clean.
    with tempfile.TemporaryDirectory(prefix="neuroflux_ss_") as tmp:
        fs_seg_path     = os.path.join(tmp, "synthseg_fs.nii.gz")
        resampled_path  = os.path.join(tmp, "resampled.nii.gz")
        posteriors_path = os.path.join(tmp, "posteriors.nii.gz")
        volumes_path    = os.path.join(tmp, "volumes.csv")
        qc_path         = os.path.join(tmp, "qc.csv")

        _run_synthseg(
            input_path=input_path,
            seg_path=fs_seg_path,
            resampled_path=resampled_path,
            posteriors_path=posteriors_path,
            volumes_path=volumes_path,
            qc_path=qc_path,
            robust=robust,
            fast=fast,
            threads=threads,
            use_ct=use_ct,
        )

        if not os.path.isfile(fs_seg_path):
            raise RuntimeError(
                "SynthSeg produced no output file — check the error above."
            )

        _emit("synthseg", 85, "SynthSeg inference complete.")

        # ── 3. Label remap + hemi split ───────────────────────────────────────
        original_path, seg_full_path, seg_hemi_path = _remap_and_split(
            fs_seg_path=fs_seg_path,
            resampled_path=resampled_path,
            output_dir=output_dir,
            input_path=input_path,
        )

        # ── 4. QC summary ─────────────────────────────────────────────────────
        _emit("qc", 96, "Writing summary.json…")
        elapsed = time.time() - t0
        summary_path = _write_summary(
            volumes_path=volumes_path,
            qc_path=qc_path,
            output_dir=output_dir,
            elapsed_sec=elapsed,
            robust=robust,
            threads=threads,
        )

        # Keep raw FreeSurfer label volume for any downstream analysis
        fs_out = out("seg_fs_labels.nii.gz")
        shutil.copy2(fs_seg_path, fs_out)

    outputs = {
        "original":  original_path,
        "seg_full":  seg_full_path,
        "seg_hemi":  seg_hemi_path,
        "summary":   summary_path,
        "seg_fs":    fs_out,
    }

    _emit("done", 100, f"All outputs written in {time.time()-t0:.0f}s.")
    _done(outputs)
    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="NEURO//FLUX v1.6 — SynthSeg 2.0 brain segmentation",
    )
    p.add_argument("input",
                   help="Input NIfTI (.nii / .nii.gz)")
    p.add_argument("output", nargs="?", default=None,
                   help="Output directory (default: same directory as input)")
    p.add_argument("--robust", action="store_true",
                   help="SynthSeg-robust model — recommended for clinical / "
                        "low-SNR / large-slice-spacing scans (slightly slower)")
    p.add_argument("--fast", action="store_true",
                   help="Skip some SynthSeg post-processing for ~2x speed "
                        "(ignored when --robust is active)")
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for SynthSeg inference (default: 1)")
    p.add_argument("--ct", action="store_true",
                   help="Input is a CT scan in Hounsfield units")
    return p


def main():
    args = _build_parser().parse_args()
    t0   = time.time()
    try:
        result = run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            robust=args.robust,
            fast=args.fast,
            threads=args.threads,
            use_ct=args.ct,
        )
        print(f"\nDone in {time.time()-t0:.0f}s")
        for k, v in result.items():
            print(f"  {k:14s}  {v}")
    except Exception as e:
        _error(str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
