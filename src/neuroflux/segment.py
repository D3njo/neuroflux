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

import argparse
import io
import json
import os
import pathlib
import platform
import shutil
import sys
import tempfile
import time
import traceback

import nibabel as nib
import numpy as np

from neuroflux.labels import HEMI_NAMES, TISSUE_NAMES, fs_to_hemi, fs_to_tissue

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE     = pathlib.Path(__file__).parent
_DATA_DIR = _HERE / "synthseg" / "data" / "labels_classes_priors"


def _resolve_model_dir() -> pathlib.Path:
    """
    Locate the model weights directory.

    Resolution order:
      1. NEUROFLUX_MODELS_DIR environment variable  (explicit override)
      2. <repo_root>/models/   — works when installed with  pip install -e .
      3. ~/.local/share/neuroflux/models/            (XDG / standard user dir)

    The first directory that contains at least one .h5 file wins.
    Falls back to (3) if none of the candidates contain weights yet
    (neuroflux-setup has not been run), so the error message shows the
    right path to populate.
    """
    candidates = []

    env = os.environ.get("NEUROFLUX_MODELS_DIR")
    if env:
        candidates.append(pathlib.Path(env))

    # editable install: __file__ is src/neuroflux/segment.py
    # _HERE = src/neuroflux/ → ../../models = <repo_root>/models
    repo_models = (_HERE / ".." / ".." / "models").resolve()
    candidates.append(repo_models)

    # standard user data directory (works for pip install without -e)
    xdg = pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local" / "share"))
    candidates.append(xdg / "neuroflux" / "models")

    for p in candidates:
        if p.is_dir() and any(p.glob("*.h5")):
            return p

    # nothing found yet — return the user dir so setup writes there
    return candidates[-1]


_MODEL_DIR = _resolve_model_dir()


# ── Progress / status emitters ────────────────────────────────────────────────

def _emit(step, pct, msg):
    print(json.dumps({"step": step, "pct": pct, "msg": msg}), flush=True)

def _done(outputs):
    print(json.dumps({"status": "done", "outputs": outputs}), flush=True)

def _error(msg):
    print(json.dumps({"status": "error", "msg": msg}), flush=True)


# ── TF configuration (must run before any TF import) ─────────────────────────

_TF_CONFIGURED = False

def _configure_tf(threads: int, low_memory: bool = False):
    """
    Configure TensorFlow before it initialises.
    - Suppress noisy startup logs.
    - Enable memory growth so TF/Metal does not pre-allocate all RAM
      (critical on unified-memory / low-RAM systems like Apple Silicon or
       ChromeOS Flex with 8 GB).
    - Wire the --threads argument to TF's inter/intra-op thread counts.
    - low_memory: unused here — RAM savings are applied in _run_synthseg instead.

    This function is idempotent — calling it multiple times is safe.
    """
    global _TF_CONFIGURED
    if _TF_CONFIGURED:
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Suppress CUDA plugin registration warnings (cuDNN, cuFFT, cuBLAS)
    # These are harmless but noisy; redirect stderr temporarily
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        import tensorflow as tf
    finally:
        sys.stderr = old_stderr

    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # device already initialised

    # low_memory RAM savings are handled in _run_synthseg (fast mode, cropping, no QC),
    # not here — mixed_float16 has no effect on CPU TensorFlow and is not set.

    # ── Apple Silicon hint ────────────────────────────────────────────────────
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print(
                "[neuroflux] Apple Silicon detected but no Metal GPU found.\n"
                "            Install the GPU plugin for faster inference:\n"
                '            pip install "neuroflux[metal]"',
                file=sys.stderr,
            )

    _TF_CONFIGURED = True


# ── Model weight check + version check ───────────────────────────────────────

def _check_models(robust: bool):
    """
    Raise a descriptive error when required .h5 weights are missing or
    are clearly wrong files (size < 1 MB — e.g. a stale placeholder).
    """
    needed = [
        "synthseg_2.0.h5" if not robust else "synthseg_robust_2.0.h5",
        "synthseg_qc_2.0.h5",
    ]
    missing, corrupt = [], []
    for fname in needed:
        p = _MODEL_DIR / fname
        if not p.is_file():
            missing.append(fname)
        elif p.stat().st_size < 1_000_000:
            corrupt.append(fname)

    if missing or corrupt:
        lines = []
        if missing:
            lines += ["Missing:"] + [f"  {f}" for f in missing]
        if corrupt:
            lines += ["Too small (re-download):"] + [f"  {f}" for f in corrupt]
        raise RuntimeError(
            "Model weights not found or corrupt. Run setup first:\n"
            "  neuroflux-setup\n"
            "Models dir: {}\n".format(_MODEL_DIR) + "\n".join(lines)
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_to_brain_fov(input_path: str, tmp_dir: str, margin_mm: float = 25.0) -> str:
    """
    Pre-crop a scan to the FOV that contains only the skull/brain.

    Many clinical scans include neck, face and large empty FOV.  SynthSeg's
    center crop then lands in the neck rather than the brain.

    Strategy: compute the cross-sectional area profile along each axis.
    The skull/brain produces a prominent peak (large area) while neck and
    vertex have much smaller cross-sections.  We find that axis and crop
    to the contiguous region around the peak that stays above 40 % of the
    maximum — which reliably isolates the skull dome.

    Returns the path to the pre-cropped file, or the original path if the
    scan is already well-centered (< 10 % removed) or detection fails.
    """
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    margin_vox = np.ceil(margin_mm / voxel_sizes).astype(int)

    # Non-background mask (> 2nd percentile of non-zero voxels)
    nonzero = data[data > 0]
    if len(nonzero) < 1000:
        return input_path
    mask = data > np.percentile(nonzero, 2)

    # Cross-sectional area profile along each axis
    profiles = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        profiles.append(mask.sum(axis=other).astype(float))

    # The skull/brain axis has the highest peak-to-mean ratio:
    # brain dome >> neck cross-section
    ratios = [p.max() / (p.mean() + 1e-6) for p in profiles]
    si_axis = int(np.argmax(ratios))
    profile = profiles[si_axis]

    # Expand from the peak outward while area stays > 40 % of maximum
    threshold = profile.max() * 0.40
    peak_idx = int(np.argmax(profile))

    start = peak_idx
    while start > 0 and profile[start - 1] > threshold:
        start -= 1
    end = peak_idx
    while end < len(profile) - 1 and profile[end + 1] > threshold:
        end += 1

    # Add margin
    start = max(0, start - int(margin_vox[si_axis]))
    end   = min(data.shape[si_axis] - 1, end + int(margin_vox[si_axis]))

    # Skip if less than 10 % removed — not worth rewriting the file
    if (end - start + 1) >= 0.90 * data.shape[si_axis]:
        del data, mask
        return input_path

    slices = [slice(None)] * 3
    slices[si_axis] = slice(start, end + 1)
    cropped = data[tuple(slices)].copy()

    # Free full-resolution array before saving — critical on low-RAM systems
    del data, mask
    import gc; gc.collect()

    # Shift affine origin to account for removed slices
    new_affine = affine.copy()
    offset = np.zeros(3)
    offset[si_axis] = float(start)
    new_affine[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]

    out_path = os.path.join(tmp_dir, "brain_fov.nii.gz")
    nib.save(nib.Nifti1Image(cropped, new_affine, img.header), out_path)
    del cropped
    gc.collect()
    return out_path


def _has_real_swap() -> bool:
    """
    Return True only if the system has real disk-backed swap (file or partition).
    zram (compressed-RAM swap used by ChromeOS/Android) is excluded because it
    does not provide additional memory — it compresses existing RAM and cannot
    absorb the large tensors produced by SynthSeg without triggering OOM anyway.

    /proc/swaps columns: Filename  Type  Size  Used  Priority
    Real swap examples:
      /swapfile          file       ...
      /dev/sda2          partition  ...
    zram example (ChromeOS):
      /dev/zram0         partition  ...   ← excluded
    """
    try:
        with open("/proc/swaps") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Filename"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                filename, swap_type = parts[0], parts[1]
                # Skip zram devices — they don't add usable memory headroom
                if "zram" in filename:
                    continue
                # File-backed or real partition swap
                if swap_type in ("file", "partition"):
                    return True
        # /proc/swaps was readable but contained no real swap entries
        return False
    except OSError:
        pass
    try:
        # macOS: dynamic paging always counts as real swap
        import subprocess
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=3)
        total_str = out.split("total =")[1].split()[0].rstrip("M")
        return float(total_str) > 0
    except Exception:
        pass
    # Windows or other unknown platform — assume real swap exists
    return True


# ── SynthSeg direct call ──────────────────────────────────────────────────────

def _run_synthseg(
    input_path, seg_path, resampled_path,
    posteriors_path, volumes_path, qc_path,
    robust=False, fast=False, threads=1, use_ct=False,
    low_memory=False,
):
    """
    Call SynthSeg's predict() directly in-process.
    Replaces the old subprocess approach (no separate venv needed).

    low_memory mode reduces peak RAM by:
      - forcing fast=True (single forward pass instead of normal+flipped average → ~50% less peak RAM)
      - cropping input to 192³ (vs. unconstrained padding → ~40% less input tensor RAM)
      - disabling QC model (saves ~200 MB model weights + activations)
    These are the only knobs that actually reduce memory on CPU/Intel GPU;
    mixed_float16 has no effect on CPU TensorFlow.
    """
    # Lazy import — TF is already configured by run_pipeline()
    from neuroflux.synthseg.predict_synthseg import predict as _ss_predict

    mode = "robust" if robust else "standard"

    if low_memory:
        # fast=True: skip the flipped-image second pass (halves peak activation RAM)
        fast = True
        # skip QC model and posteriors to avoid materializing 95-class posterior tensor (~800 MB)
        do_qc_path    = None
        do_posteriors = None

        # Only crop when the system has no swap space.
        # With swap the OS can page out tensors to disk instead of OOM-killing the process,
        # so cropping (which cuts off parts of large brains) is unnecessary.
        swap_available = _has_real_swap()
        if swap_available:
            cropping = None
            _emit("synthseg", 8,
                  f"SynthSeg 2.0 ({mode}, low_memory: fast+no-QC+no-posteriors, {threads} thread(s))…")
        else:
            # No swap: crop to 160³ as safety net against OOM-kill.
            # 160mm covers a typical brain (~170mm); larger brains may be clipped at edges.
            cropping = 160
            _emit("synthseg", 8,
                  f"SynthSeg 2.0 ({mode}, low_memory: fast+crop160+no-QC+no-posteriors [no swap], {threads} thread(s))…")
    else:
        cropping      = None
        do_qc_path    = qc_path
        do_posteriors = posteriors_path
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
        path_posteriors           = do_posteriors,
        path_resampled            = resampled_path,
        path_volumes              = volumes_path,
        do_parcellation           = False,
        path_model_parcellation   = str(_MODEL_DIR / "synthseg_parc_2.0.h5"),
        labels_parcellation       = str(_DATA_DIR / "synthseg_parcellation_labels.npy"),
        path_qc_scores            = do_qc_path,
        path_model_qc             = str(_MODEL_DIR / "synthseg_qc_2.0.h5"),
        labels_qc                 = str(_DATA_DIR / "synthseg_qc_labels_2.0.npy"),
        cropping                  = cropping,
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
    low_memory=False,
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
    low_memory : bool       Enable mixed float16 precision to halve model RAM.
                            Useful on 8 GB systems (ChromeOS Flex, older M1).

    Returns
    -------
    dict  keys: original, seg_full, seg_hemi, summary, seg_fs
    """
    # Configure TensorFlow before any lazy imports.
    # This must happen first to avoid double-initialization when predict_synthseg imports TF.
    _configure_tf(threads, low_memory=low_memory)

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

    # Warn early if low_memory crop will clip this scan
    if low_memory and not _has_real_swap():
        vox = spacing  # mm per voxel
        size_mm = tuple(round(shape[i] * vox[i], 1) for i in range(3))
        clipped = [s for s in size_mm if s > 160]
        if clipped:
            _emit("setup", 5, (
                f"Warning: scan is {size_mm[0]}×{size_mm[1]}×{size_mm[2]} mm — "
                f"low_memory crops to 160 mm, edges may be clipped. "
                f"Add real swap to process the full scan."
            ))

    # ── 2. SynthSeg inference ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="neuroflux_ss_") as tmp:
        fs_seg_path     = os.path.join(tmp, "synthseg_fs.nii.gz")
        resampled_path  = os.path.join(tmp, "resampled.nii.gz")
        posteriors_path = os.path.join(tmp, "posteriors.nii.gz")
        volumes_path    = os.path.join(tmp, "volumes.csv")
        qc_path         = os.path.join(tmp, "qc.csv")

        # Pre-crop to brain FOV before SynthSeg so its center crop lands
        # on the brain even when the scan includes neck, face or large empty FOV.
        _emit("setup", 6, "Localising brain FOV…")
        seg_input = _crop_to_brain_fov(input_path, tmp)
        if seg_input != input_path:
            _emit("setup", 7, "Brain FOV detected — pre-cropped to skull region.")
        # Ensure pre-crop arrays are released before TF allocates model memory
        import gc; gc.collect()

        _run_synthseg(
            input_path=seg_input,
            seg_path=fs_seg_path,
            resampled_path=resampled_path,
            posteriors_path=posteriors_path,
            volumes_path=volumes_path,
            qc_path=qc_path,
            robust=robust,
            fast=fast,
            threads=threads,
            use_ct=use_ct,
            low_memory=low_memory,
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
    p.add_argument("--low-memory", action="store_true",
                   help="Enable mixed float16 precision to halve GPU/RAM usage "
                        "(recommended on 8 GB systems)")
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
            low_memory=args.low_memory,
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
