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

def _detect_fov_crop_params(
    input_path: str,
    margin_mm: float = 25.0,
) -> dict | None:
    """
    Detect whether the scan contains excess FOV (neck/face) and compute the
    crop parameters that would isolate the skull/brain region.

    Strategy: compute cross-sectional area profile along each axis.  The
    skull/brain produces a prominent peak while neck/vertex are much smaller.
    We find the SI axis (highest peak-to-mean ratio) and expand from the peak
    while area stays above 40 % of the maximum.

    Returns a dict with crop parameters, or None if the scan needs < 10 %
    removed (already well-centred — cropping not worthwhile).

    Dict keys
    ---------
    si_axis    : int            detected superior-inferior axis (0/1/2)
    start      : int            first kept slice along si_axis (after margin)
    end        : int            last kept slice along si_axis  (after margin)
    n_slices   : int            original size along si_axis
    orig_shape : list[int]      full voxel dimensions of the scan
    removed_pct: float          fraction of slices removed (0-100)
    """
    import gc

    img        = nib.load(input_path)
    affine     = img.affine.copy()
    orig_shape = list(img.shape[:3])
    data       = img.get_fdata(dtype=np.float32)
    del img
    gc.collect()

    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    margin_vox  = np.ceil(margin_mm / voxel_sizes).astype(int)

    nonzero = data[data > 0]
    if len(nonzero) < 1000:
        del data, nonzero, affine
        gc.collect()
        return None
    thresh = float(np.percentile(nonzero, 2))
    del nonzero
    mask = data > thresh
    del data
    gc.collect()

    # Determine SI axis from affine (dominant world-space row for SI = row 2)
    # Falls back to profile-ratio method if affine looks degenerate.
    dom_map: dict[int, int] = {}
    for ax in range(3):
        dom = int(np.abs(affine[:3, ax]).argmax())
        dom_map[dom] = ax
    si_axis_aff = dom_map.get(2)  # NIfTI voxel axis whose dominant direction is SI

    # Cross-sectional area profiles along each axis
    profiles = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        profiles.append(mask.sum(axis=other).astype(float))

    ratios  = [p.max() / (p.mean() + 1e-6) for p in profiles]
    si_axis = si_axis_aff if si_axis_aff is not None else int(np.argmax(ratios))
    profile = profiles[si_axis]
    del profiles, ratios

    threshold = float(profile.max()) * 0.40
    peak_idx  = int(np.argmax(profile))
    del profile

    n_slices = orig_shape[si_axis]
    prof_si  = mask.sum(axis=tuple(i for i in range(3) if i != si_axis)).astype(float)
    del mask
    gc.collect()

    start = peak_idx
    while start > 0 and prof_si[start - 1] > threshold:
        start -= 1
    end = peak_idx
    while end < n_slices - 1 and prof_si[end + 1] > threshold:
        end += 1
    del prof_si

    # Add margin
    start = max(0, start - int(margin_vox[si_axis]))
    end   = min(n_slices - 1, end + int(margin_vox[si_axis]))

    removed_pct = round((1.0 - (end - start + 1) / n_slices) * 100.0, 1)

    if removed_pct < 10.0:
        return None

    return {
        "si_axis":     si_axis,
        "start":       start,
        "end":         end,
        "n_slices":    n_slices,
        "orig_shape":  orig_shape,
        "removed_pct": removed_pct,
    }


def _apply_fov_crop(
    input_path: str,
    out_path: str,
    si_axis: int,
    start: int,
    end: int,
) -> str:
    """
    Crop input_path along si_axis [start:end+1] and save to out_path.
    Returns out_path.  Frees the large array immediately after saving.
    """
    import gc

    img     = nib.load(input_path)
    affine  = img.affine.copy()
    header  = img.header.copy()
    data    = img.get_fdata(dtype=np.float32)
    del img
    gc.collect()

    slices       = [slice(None)] * 3
    slices[si_axis] = slice(start, end + 1)
    cropped      = data[tuple(slices)].copy()
    del data
    gc.collect()

    new_affine = affine.copy()
    offset = np.zeros(3)
    offset[si_axis] = float(start)
    new_affine[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]

    nib.save(nib.Nifti1Image(cropped, new_affine, header), out_path)
    del cropped, new_affine, affine, header
    gc.collect()

    return out_path


def _crop_to_brain_fov(input_path: str, tmp_dir: str, margin_mm: float = 25.0) -> str:
    """
    Detect and apply brain FOV crop if needed.  Used by the CLI pipeline when
    the server has not already pre-cropped the file.

    Returns the path to the cropped file, or the original path if no crop
    was needed.
    """
    params = _detect_fov_crop_params(input_path, margin_mm)
    if params is None:
        return input_path
    out_path = os.path.join(tmp_dir, "brain_fov.nii.gz")
    return _apply_fov_crop(
        input_path, out_path,
        params["si_axis"], params["start"], params["end"],
    )


def analyze_and_crop_fov(
    input_path: str,
    out_dir: str,
    margin_mm: float = 25.0,
) -> dict:
    """
    Public API — detect brain FOV, crop if needed, save to out_dir.

    Call this BEFORE starting TensorFlow so the numpy arrays are freed and
    the OS has time to reclaim pages before model weights are loaded.

    Returns
    -------
    {
      "cropped":     bool,       True when a crop was applied
      "path":        str,        path to the (possibly cropped) file
      "removed_pct": float,      % of slices removed (0 when cropped=False)
      "si_axis":     int | None, detected SI axis (0/1/2)
      "orig_shape":  list[int],  original voxel dimensions
      "crop_start":  int | None, first kept slice
      "crop_end":    int | None, last kept slice
    }
    """
    params = _detect_fov_crop_params(input_path, margin_mm)
    if params is None:
        try:
            img = nib.load(input_path)
            shape = list(img.shape[:3])
            del img
        except Exception:
            shape = []
        return {
            "cropped":     False,
            "path":        input_path,
            "removed_pct": 0.0,
            "si_axis":     None,
            "orig_shape":  shape,
            "crop_start":  None,
            "crop_end":    None,
        }

    os.makedirs(out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(input_path))[0]
    stem     = stem.removesuffix(".nii")          # handle double extension
    out_path = os.path.join(out_dir, f"{stem}_brain_fov.nii.gz")
    _apply_fov_crop(
        input_path, out_path,
        params["si_axis"], params["start"], params["end"],
    )
    return {
        "cropped":     True,
        "path":        out_path,
        "removed_pct": params["removed_pct"],
        "si_axis":     params["si_axis"],
        "orig_shape":  params["orig_shape"],
        "crop_start":  params["start"],
        "crop_end":    params["end"],
    }


def manual_crop_fov(
    input_path: str,
    out_path: str,
    si_axis: int,
    start_vox: int,
    end_vox: int,
) -> dict:
    """
    Public API — apply a user-specified FOV crop along si_axis.

    Parameters
    ----------
    input_path : str   Source NIfTI file.
    out_path   : str   Destination path for the cropped file.
    si_axis    : int   Axis to crop along (0/1/2).
    start_vox  : int   First voxel to keep (inclusive).
    end_vox    : int   Last voxel to keep  (inclusive).

    Returns the same structure as analyze_and_crop_fov.
    """
    img      = nib.load(input_path)
    n_slices = img.shape[si_axis]
    shape    = list(img.shape[:3])
    del img

    start_vox = max(0, int(start_vox))
    end_vox   = min(n_slices - 1, int(end_vox))
    if start_vox >= end_vox:
        return {
            "cropped": False, "path": input_path, "removed_pct": 0.0,
            "si_axis": si_axis, "orig_shape": shape,
            "crop_start": None, "crop_end": None,
        }

    _apply_fov_crop(input_path, out_path, si_axis, start_vox, end_vox)
    removed_pct = round((1.0 - (end_vox - start_vox + 1) / n_slices) * 100.0, 1)
    return {
        "cropped":     True,
        "path":        out_path,
        "removed_pct": removed_pct,
        "si_axis":     si_axis,
        "orig_shape":  shape,
        "crop_start":  start_vox,
        "crop_end":    end_vox,
    }


def manual_crop_fov_multi(
    input_path: str,
    out_path: str,
    crops: list,
) -> dict:
    """
    Apply multiple axis crops in sequence.

    crops: list of {"si_axis": int, "start_vox": int, "end_vox": int}
    Only axes where the user actually trimmed something are applied.
    Returns the same structure as manual_crop_fov.
    """
    import gc
    import shutil

    orig_img   = nib.load(input_path)
    orig_shape = list(orig_img.shape[:3])
    del orig_img
    gc.collect()

    current    = input_path
    applied    = []          # crops actually applied
    prev_tmp   = None        # track intermediate tmp files

    for crop in crops:
        ax    = int(crop["si_axis"])
        start = int(crop["start_vox"])
        end   = int(crop["end_vox"])

        img = nib.load(current)
        n   = img.shape[ax]
        del img
        gc.collect()

        start = max(0, start)
        end   = min(n - 1, end)
        if start <= 0 and end >= n - 1:
            continue  # full range → skip
        if start >= end:
            continue

        tmp = out_path + f".tmp_ax{ax}.nii.gz"
        _apply_fov_crop(current, tmp, ax, start, end)

        if prev_tmp and os.path.exists(prev_tmp):
            os.remove(prev_tmp)
        prev_tmp = tmp
        current  = tmp
        applied.append({"si_axis": ax, "start_vox": start, "end_vox": end, "n": n})

    if not applied:
        return {
            "cropped": False, "path": input_path, "removed_pct": 0.0,
            "si_axis": None, "orig_shape": orig_shape,
            "crop_start": None, "crop_end": None,
        }

    shutil.move(current, out_path)

    # Use the primary (first applied) crop for the summary fields
    p0  = applied[0]
    removed_pct = round(
        (1.0 - (p0["end_vox"] - p0["start_vox"] + 1) / p0["n"]) * 100.0, 1
    )
    return {
        "cropped":     True,
        "path":        out_path,
        "removed_pct": removed_pct,
        "si_axis":     p0["si_axis"],
        "orig_shape":  orig_shape,
        "crop_start":  p0["start_vox"],
        "crop_end":    p0["end_vox"],
        "crops":       applied,
    }


def get_fov_profiles(input_path: str) -> dict:
    """
    Public API — compute cross-sectional area profiles for all 3 axes.

    Used by the browser FOV crop UI to draw the profile graph.

    Returns
    -------
    {
      "profiles":  [[...], [...], [...]],   # one list per axis, length = n_slices
      "si_axis":   int,                     # detected SI axis (0/1/2)
      "shape":     [X, Y, Z],
      "voxel_mm":  [dx, dy, dz],
    }
    Each profile value is the cross-sectional area (number of non-background
    voxels) at that slice index along the corresponding axis.
    """
    import gc

    img         = nib.load(input_path)
    affine      = img.affine
    voxel_mm    = [round(float(v), 4) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0))]
    shape       = list(img.shape[:3])
    data        = img.get_fdata(dtype=np.float32)
    del img
    gc.collect()

    nonzero = data[data > 0]
    thresh  = float(np.percentile(nonzero, 2)) if len(nonzero) > 1000 else 0.0
    del nonzero
    mask = (data > thresh)
    del data
    gc.collect()

    profiles = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        profiles.append(mask.sum(axis=other).astype(int).tolist())
    del mask
    gc.collect()

    ratios  = [max(p) / (sum(p) / len(p) + 1e-6) for p in profiles]
    si_axis = int(ratios.index(max(ratios)))

    # axis_neg[i] = True when voxel-0 is at the anatomically HIGH end of that axis.
    axis_neg = []
    # dom_map: world axis (0=LR, 1=AP, 2=SI) → NIfTI voxel axis
    dom_map: dict[int, int] = {}
    for ax in range(3):
        dom = int(np.abs(affine[:3, ax]).argmax())  # dominant world-space row
        axis_neg.append(bool(affine[dom, ax] < 0))
        dom_map[dom] = ax

    lr_axis = dom_map.get(0, 0)
    ap_axis = dom_map.get(1, 1)
    si_axis_aff = dom_map.get(2, 2)  # affine-based SI (more reliable than profile ratio)

    return {
        "profiles":   profiles,
        "si_axis":    si_axis,      # profile-based SI axis (legacy)
        "si_axis_aff": si_axis_aff, # affine-based SI axis  (preferred)
        "ap_axis":    ap_axis,
        "lr_axis":    lr_axis,
        "shape":      shape,
        "voxel_mm":   voxel_mm,
        "axis_neg":   axis_neg,
    }


def _is_crostini() -> bool:
    """Detect if running inside a ChromeOS Crostini (Linux VM) container."""
    return os.path.exists("/dev/.cros_milestone")


def _has_real_swap() -> bool:
    """
    Return True only if the system has real disk-backed swap (file or partition).
    zram (compressed-RAM swap used by ChromeOS/Android) is excluded because it
    does not provide additional memory — it compresses existing RAM and cannot
    absorb the large tensors produced by SynthSeg without triggering OOM anyway.

    Note: Crostini containers show empty /proc/swaps even though the ChromeOS
    host has swap.  _is_crostini() is checked separately in _run_synthseg to
    use an intermediate crop (192mm) instead of the full-size path.
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


def _smart_crop(input_path: str) -> list:
    """
    Compute per-axis crop sizes (mm) tailored to brain anatomy.

    AP axis needs ~192mm (brain ~170mm + margin), LR and SI need only ~160mm.
    This gives a rectangular crop that covers the full brain while using ~30%
    less memory than a uniform 192³ cube.
    """
    img = nib.load(input_path)
    affine = img.affine
    del img

    # Find which voxel axis corresponds to AP (world row 1 = anterior-posterior)
    dom_map: dict[int, int] = {}
    for ax in range(3):
        dom = int(np.abs(affine[:3, ax]).argmax())
        dom_map[dom] = ax

    ap_axis = dom_map.get(1, 1)  # voxel axis aligned with A-P

    crop = [160, 160, 160]
    crop[ap_axis] = 192
    return crop


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
      - per-axis brain crop (AP=192mm, LR/SI=160mm) when no swap available
      - skipping posteriors file output (still computed internally for volumes, freed immediately)
      - freeing input tensor before postprocessing
    QC model is kept (~48 MB overhead).
    These are the only knobs that actually reduce memory on CPU/Intel GPU;
    mixed_float16 has no effect on CPU TensorFlow.
    """
    # Lazy import — TF is already configured by run_pipeline()
    from neuroflux.synthseg.predict_synthseg import predict as _ss_predict

    mode = "robust" if robust else "standard"

    if low_memory:
        # fast=True: skip the flipped-image second pass (halves peak activation RAM)
        fast = True
        # Skip posteriors file (the 37-channel probability tensor is ~725 MB; it is
        # still computed internally for volume calculation but freed immediately after).
        # QC is kept — its model is only ~48 MB and adds minimal activation overhead.
        do_qc_path    = qc_path
        do_posteriors = None

        # Only crop when the system has no swap space.
        # With swap the OS can page out tensors to disk instead of OOM-killing the process,
        # so cropping (which cuts off parts of large brains) is unnecessary.
        swap_available = _has_real_swap()
        if swap_available:
            cropping = None
            _emit("synthseg", 8,
                  f"SynthSeg 2.0 ({mode}, low_memory: fast+no-posteriors, {threads} thread(s))…")
        else:
            # No real swap: use per-axis rectangular crop sized to the brain.
            # AP axis needs ~192mm (brain is ~170mm), LR/SI only ~160mm.
            # This covers the full brain while using ~30% less RAM than a 192³ cube.
            cropping = _smart_crop(input_path)
            tag = "crostini" if _is_crostini() else "no swap"
            _emit("synthseg", 8,
                  f"SynthSeg 2.0 ({mode}, low_memory: fast+crop{cropping}+no-posteriors [{tag}], {threads} thread(s))…")
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

    import gc
    img    = nib.load(fs_seg_path)
    fs_arr = np.asarray(img.dataobj, dtype=np.int32)
    affine = img.affine.copy()
    header = img.header.copy()
    del img; gc.collect()

    tissue_arr = fs_to_tissue(fs_arr)
    hemi_arr   = fs_to_hemi(fs_arr, tissue_arr)
    del fs_arr; gc.collect()  # free the largest array ASAP

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
    del tissue_arr; gc.collect()
    _emit("remap", 90, "seg_full.nii.gz saved.")

    for lbl, name in HEMI_NAMES.items():
        vox = int((hemi_arr == lbl).sum())
        _emit("hemi", 93, f"  {name:6s}: {vox:>10,} vox")

    seg_hemi_path = _save(hemi_arr, "seg_hemi.nii.gz")
    del hemi_arr; gc.collect()
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
    skip_fov_crop=False,
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
        crop_vals = _smart_crop(input_path)
        clipped = any(size_mm[i] > crop_vals[i] for i in range(3))
        if clipped:
            _emit("setup", 5, (
                f"Warning: scan is {size_mm[0]}×{size_mm[1]}×{size_mm[2]} mm — "
                f"low_memory crops to {crop_vals[0]}×{crop_vals[1]}×{crop_vals[2]} mm, "
                f"edges may be clipped. Add real swap to process the full scan."
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
        # When skip_fov_crop=True the caller (server) already cropped the file.
        if skip_fov_crop:
            seg_input = input_path
        else:
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

        # ── Free TensorFlow memory before post-processing ────────────────────
        # TF holds the model weights + activations + graph in RAM; release
        # everything aggressively so _remap_and_split has enough headroom.
        try:
            from neuroflux.synthseg.predict_synthseg import _MODEL_CACHE
            _MODEL_CACHE.clear()
        except Exception:
            pass
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            # Drop any remaining TF tensor caches
            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass
        except Exception:
            pass
        import gc
        # Multiple GC passes — prevent lazy-freed TF objects from lingering
        for _ in range(3):
            gc.collect()

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
    p.add_argument("--skip-fov-crop", action="store_true",
                   help="Skip automatic brain FOV pre-crop (use when the file "
                        "was already cropped by the server's /analyze_fov endpoint)")
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
            skip_fov_crop=args.skip_fov_crop,
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
