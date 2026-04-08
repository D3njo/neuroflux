"""
NEURO//FLUX — Brain Segmentation Pipeline v2.0  (SynthSeg 2.0)
===============================================================
Cross-platform (Windows / macOS / Linux).
No FreeSurfer installation required.
SynthSeg is now bundled directly as neuroflux.synthseg — no separate venv needed.

Backend: SynthSeg 2.0 (Billot et al., Harvard/MGH, PNAS 2023)
  - Contrast- and resolution-agnostic 3-D U-Net
  - Domain-randomisation training: works on any MRI contrast / resolution
  - 95 FreeSurfer structures -> mapped to 7-class tissue + 10-class hemi
  - --robust model for clinical / low-SNR / large-spacing scans
  - QC scores and regional volumes written alongside NIfTI outputs
  - GPU auto-detected; graceful CPU fallback

One-time setup (model weights only)
  neuroflux-setup            # or: python -m neuroflux.setup_synthseg

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
import pathlib
import shutil
import tempfile

import numpy as np
import nibabel as nib

from neuroflux.labels import fs_to_tissue, fs_to_hemi, TISSUE_NAMES, HEMI_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = pathlib.Path(__file__).parent
_DATA_DIR   = _HERE / "synthseg" / "data" / "labels_classes_priors"
# Model weights live two levels up from src/neuroflux/ → repo root / models/
_MODEL_DIR  = _HERE.parent.parent / "models"


# ── Progress / status emitters ────────────────────────────────────────────────

def _emit(step, pct, msg):
    print(json.dumps({"step": step, "pct": pct, "msg": msg}), flush=True)

def _done(outputs):
    print(json.dumps({"status": "done", "outputs": outputs}), flush=True)

def _error(msg):
    print(json.dumps({"status": "error", "msg": msg}), flush=True)


# ── TF configuration (must run before any TF import) ─────────────────────────

def _configure_tf(threads: int):
    """
    Configure TensorFlow before it initialises.
    - Suppress noisy startup logs.
    - Enable memory growth so TF/Metal does not pre-allocate all RAM
      (critical on unified-memory / low-RAM systems like Apple Silicon or
       ChromeOS Flex with 8 GB).
    - Wire the --threads argument to TF's inter/intra-op thread counts.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # device already initialised


# ── Model weight check ────────────────────────────────────────────────────────

def _check_models(robust: bool):
    """Raise a descriptive error when required .h5 weights are missing."""
    needed = [
        "synthseg_2.0.h5" if not robust else "synthseg_robust_2.0.h5",
        "synthseg_qc_2.0.h5",
    ]
    missing = [f for f in needed if not (_MODEL_DIR / f).is_file()]
    if missing:
        raise RuntimeError(
            "Model weights not found. Run setup first:\n"
            "  neuroflux-setup\n"
            "Missing in {}:\n".format(_MODEL_DIR)
            + "\n".join(f"  {f}" for f in missing)
        )


# ── SynthSeg direct call ──────────────────────────────────────────────────────

def _run_synthseg(
    input_path, seg_path, resampled_path,
    posteriors_path, volumes_path, qc_path,
    robust=False, fast=False, threads=1, use_ct=False,
):
    """
    Call SynthSeg's predict() directly in-process.
    Replaces the old subprocess approach (no separate venv needed).
    """
    # TF must be configured before the first TF import
    _configure_tf(threads)

    # Lazy import — keeps TF out of module-load time
    from neuroflux.synthseg.predict_synthseg import predict as _ss_predict

    mode = "robust" if robust else "standard"
    _emit("synthseg", 8, f"SynthSeg 2.0 ({mode} mode, {threads} thread(s))…")

    _ss_predict(
        path_images               = input_path,
        path_segmentations        = seg_path,
        path_model_segmentation   = str(_MODEL_DIR / (
            "synthseg_robust_2.0.h5" if robust else "synthseg_2.0.h5"
        )),
        labels_segmentation       = str(_DATA_DIR / "synthseg_segmentation_labels_2.0.npy"),
        robust                    = robust,
        fast                      = fast or robust,
        v1                        = False,
        n_neutral_labels          = 19,
        labels_denoiser           = str(_DATA_DIR / "synthseg_denoiser_labels_2.0.npy"),
        path_posteriors           = posteriors_path,
        path_resampled            = resampled_path,
        path_volumes              = volumes_path,
        do_parcellation           = False,
        path_model_parcellation   = str(_MODEL_DIR / "synthseg_parc_2.0.h5"),
        labels_parcellation       = str(_DATA_DIR / "synthseg_parcellation_labels.npy"),
        path_qc_scores            = qc_path,
        path_model_qc             = str(_MODEL_DIR / "synthseg_qc_2.0.h5"),
        labels_qc                 = str(_DATA_DIR / "synthseg_qc_labels_2.0.npy"),
        cropping                  = None,
        ct                        = use_ct,
        names_segmentation        = str(_DATA_DIR / "synthseg_segmentation_names_2.0.npy"),
        names_parcellation        = str(_DATA_DIR / "synthseg_parcellation_names.npy"),
        names_qc                  = str(_DATA_DIR / "synthseg_qc_names_2.0.npy"),
        topology_classes          = str(_DATA_DIR / "synthseg_topological_classes_2.0.npy"),
        verbose                   = False,
    )

    _emit("synthseg", 85, "SynthSeg inference complete.")


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

    for lbl, name in TISSUE_NAMES.items():
        vox = int((tissue_arr == lbl).sum())
        _emit("remap", 88, f"  {name:12s}: {vox:>10,} vox")

    seg_full_path = _save(tissue_arr, "seg_full.nii.gz")
    _emit("remap", 90, "seg_full.nii.gz saved.")

    for lbl, name in HEMI_NAMES.items():
        vox = int((hemi_arr == lbl).sum())
        _emit("hemi", 93, f"  {name:6s}: {vox:>10,} vox")

    seg_hemi_path = _save(hemi_arr, "seg_hemi.nii.gz")
    _emit("hemi", 95, "seg_hemi.nii.gz saved.")

    original_path = os.path.join(output_dir, "original.nii.gz")
    if resampled_path and os.path.isfile(resampled_path):
        if os.path.abspath(resampled_path) != os.path.abspath(original_path):
            shutil.copy2(resampled_path, original_path)
    elif not os.path.isfile(original_path):
        fallback = input_path if (input_path and os.path.isfile(input_path)) else None
        if fallback:
            shutil.copy2(fallback, original_path)

    return original_path, seg_full_path, seg_hemi_path


# ── QC / volume summary ───────────────────────────────────────────────────────

def _write_summary(volumes_path, qc_path, output_dir, elapsed_sec, robust, threads):
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
        for h, v in zip(headers[1:], values[1:]):
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
    Run the full NEUROFLUX v2.0 segmentation pipeline.

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
    _emit("setup", 2, "Checking model weights…")
    _check_models(robust)

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
        description="NEURO//FLUX v2.0 — SynthSeg 2.0 brain segmentation",
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
