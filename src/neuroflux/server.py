"""
NEURO//FLUX — Segmentation Bridge Server  v2.0  (SynthSeg 2.0)
===============================================================
Starts a local Flask server on  http://localhost:5050

Endpoints
---------
GET  /ping
     Health check — returns {"ok": true}

GET  /gpu
     Checks whether the SynthSeg venv can see a GPU.
     Returns: {"gpu": bool, "name": str|null, "note": str|null}

POST /upload
     Multipart file upload.  Returns: {"path": "/abs/path/to/file.nii.gz"}

POST /segment
     Body: {
       "input_path":  "/absolute/path/to/file.nii.gz",
       "output_dir":  "/optional/dir",        (optional)
       "robust":      false,                  (optional, SynthSeg-robust model)
       "fast":        false,                  (optional, skip some post-proc)
       "threads":     1,                      (optional, CPU thread count)
       "ct":          false,                  (optional, CT scan mode)
       "low_memory":  false                   (optional, use float16 precision to halve GPU/RAM usage)
     }
     Starts the segmentation pipeline in a background thread.
     Returns: {"job_id": "<uuid>", "output_dir": "..."}

GET  /status/<job_id>
     Server-Sent Events stream.
     Each event is a JSON line emitted by segment.py:
       {"step": "synthseg", "pct": 30, "msg": "predicting 1/1"}
       {"status": "done",  "outputs": {"original": "...", "seg_full": "...",
                                       "seg_hemi": "...", "summary": "..."}}
       {"status": "error", "msg": "..."}

GET  /file?path=<absolute_path>
     Serves any output NIfTI file back to the browser as a binary blob.
     Only paths inside the designated output directories are allowed.

Usage
-----
    pip install -r requirements_segment.txt
    python setup_synthseg.py          # one-time SynthSeg venv setup
    python server.py [--port 5050]
    # then open neuroflux.html in your browser
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque

import nibabel as nib
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file
from flask_cors import CORS

from neuroflux.mesh_export import iter_stl_export, stl_options_from_body

app   = Flask(__name__)
CORS(app)

# ── Upload size limit: 500 MB ────────────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

_PKG_DIR       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR      = os.path.join(_PKG_DIR, "data")   # bundled static assets
SEGMENT_SCRIPT = os.path.join(_PKG_DIR, "segment.py")
SERVER_DIR     = _PKG_DIR

# All generated output lives under a single top-level output/ directory.
# When installed editably (the expected mode), _PKG_DIR is src/neuroflux/
# and _PROJECT_ROOT resolves to the repository root.
_PROJECT_ROOT  = os.path.realpath(os.path.join(_PKG_DIR, "..", ".."))
_OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "output")

# job registry  { job_id: { "status", "queue", "outputs", "output_dir", "created" } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 1800   # 30 minutes


# ── helpers ───────────────────────────────────────────────────────────────────

def _cleanup_old_jobs():
    """Remove jobs older than _JOB_TTL_SECONDS from the registry."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items()
                   if now - j.get("created", 0) > _JOB_TTL_SECONDS
                   and j.get("status") in ("done", "error")]
        for jid in expired:
            del _jobs[jid]


def _safe_path(path: str) -> bool:
    """
    Reject path traversal attempts.
    Only allow paths that:
      1. End with .nii or .nii.gz
      2. Exist on disk
      3. Reside inside allowed directories (server dir, temp uploads, outputs)
    """
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return False
    if not (path.endswith(".nii") or path.endswith(".nii.gz")):
        return False
    # Allow: anything under the server directory, or under the system temp dir.
    # Use trailing os.sep to prevent prefix attacks:
    # e.g. /foo/neuro must not match /foo/neuro_other/file.nii
    allowed_roots = [
        os.path.realpath(_OUTPUT_DIR),
        os.path.realpath(tempfile.gettempdir()),
    ]
    return any(
        path.startswith(root + os.sep) or path == root
        for root in allowed_roots
    )


def _run_job(
    job_id: str,
    input_path: str,
    output_dir: str,
    robust: bool = False,
    fast: bool = False,
    threads: int = 1,
    ct: bool = False,
    low_memory: bool = False,
    skip_fov_crop: bool = False,
) -> None:
    """Run segment.py in a subprocess, feed stdout into the SSE queue."""
    q: queue.Queue = _jobs[job_id]["queue"]

    try:
        # Build CLI command — forward all SynthSeg options
        cmd = [sys.executable, SEGMENT_SCRIPT, input_path, output_dir,
               "--threads", str(max(1, threads))]
        if robust:
            cmd.append("--robust")
        if fast and not robust:   # SynthSeg ignores --fast when --robust is active
            cmd.append("--fast")
        if ct:
            cmd.append("--ct")
        if low_memory:
            cmd.append("--low-memory")
        if skip_fov_crop:
            cmd.append("--skip-fov-crop")

        # Store proc reference so the watchdog can kill it on timeout
        with _jobs_lock:
            _jobs[job_id]["proc"] = None

        # Set TensorFlow environment variables to suppress noisy logs
        # and avoid CUDA initialization conflicts between processes
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        with _jobs_lock:
            _jobs[job_id]["proc"] = proc

        # Drain stderr in a background thread to avoid OS pipe-buffer deadlock
        # (SynthSeg can print long Python tracebacks to stderr)
        # Filter out only the specific, harmless CUDA plugin registration warnings
        stderr_lines: list = []
        def _drain_stderr():
            for line in proc.stderr:
                # Skip ONLY the exact benign CUDA warnings (cuDNN/cuFFT/cuBLAS plugin reinitialization)
                # These are harmless but noisy from TensorFlow initialization
                if ("Unable to register" in line and "factory" in line and
                    any(x in line for x in ["cuDNN", "cuFFT", "cuBLAS", "cuSOLVER"])):
                    continue
                stderr_lines.append(line)
                # Mirror to terminal in real time so errors are visible immediately
                print(line, end="", file=sys.stderr, flush=True)
        drain_t = threading.Thread(target=_drain_stderr, daemon=True)
        drain_t.start()

        stdout_tail: deque[str] = deque(maxlen=80)

        # Stream stdout line by line into the SSE queue
        for raw in iter(proc.stdout.readline, ""):
            line = raw.strip()
            if not line:
                continue
            stdout_tail.append(line)
            q.put(line)
            try:
                msg = json.loads(line)
                if msg.get("status") == "done":
                    with _jobs_lock:
                        _jobs[job_id]["status"]  = "done"
                        _jobs[job_id]["outputs"] = msg.get("outputs", {})
                elif msg.get("status") == "error":
                    with _jobs_lock:
                        _jobs[job_id]["status"] = "error"
            except json.JSONDecodeError:
                pass

        proc.wait()
        drain_t.join(timeout=5)

        # If process exited with an error but never emitted a JSON error message
        if proc.returncode != 0:
            stderr_out = "".join(stderr_lines).strip()
            stdout_out = "\n".join(stdout_tail).strip()
            details = []
            if stderr_out:
                details.append(f"stderr:\n{stderr_out}")
            if stdout_out:
                details.append(f"stdout tail:\n{stdout_out}")
            if details:
                msg_text = f"Process exited {proc.returncode}.\n\n" + "\n\n".join(details)
            else:
                msg_text = f"Process exited {proc.returncode} with no stdout/stderr output."
            # Always print to terminal so it's visible regardless of UI state
            print(f"\n[neuroflux] JOB ERROR ({job_id}):\n{msg_text}\n",
                  file=sys.stderr, flush=True)
            err_msg = json.dumps({"status": "error", "msg": msg_text})
            q.put(err_msg)
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"

    except Exception as exc:
        msg_text = str(exc)
        print(f"\n[neuroflux] JOB EXCEPTION ({job_id}):\n{msg_text}\n",
              file=sys.stderr, flush=True)
        err_msg = json.dumps({"status": "error", "msg": msg_text})
        q.put(err_msg)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
    finally:
        q.put(None)  # sentinel → close SSE stream


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return jsonify({"ok": True})


@app.get("/gpu")
def gpu_check():
    """
    Check whether the SynthSeg venv can see a GPU (via its TF 2.2 install).
    Runs a tiny subprocess so the main server process never imports TF.
    Result is cached for the lifetime of the server process.
    """
    global _gpu_cache
    with _gpu_cache_lock:
        if _gpu_cache is not None:
            return jsonify(_gpu_cache)

    # Probe GPU inside the SynthSeg venv
    ss_env  = os.path.join(SERVER_DIR, "synthseg_env")
    if platform.system() == "Windows":
        ss_python = os.path.join(ss_env, "Scripts", "python.exe")
    else:
        ss_python = os.path.join(ss_env, "bin", "python")

    result = {"gpu": False, "name": None, "note": None}

    if not os.path.isfile(ss_python):
        result["note"] = "SynthSeg venv not found — run setup_synthseg.py"
    else:
        probe = (
            "import json; "
            "import tensorflow as tf; "
            "gpus = tf.config.list_physical_devices('GPU'); "
            "n = gpus[0].name.split('/')[-1] if gpus else None; "
            "print(json.dumps({'gpu': bool(gpus), 'count': len(gpus), 'name': n}))"
        )
        try:
            r = subprocess.run(
                [ss_python, "-c", probe],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip():
                result = json.loads(r.stdout.strip())
                if not result.get("gpu"):
                    if platform.system() == "Windows":
                        result["note"] = (
                            "CPU mode — TF 2.2 has no native Windows GPU. "
                            "Use WSL2+CUDA for GPU acceleration."
                        )
            else:
                result["note"] = f"GPU probe failed: {r.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            result["note"] = "GPU probe timed out."
        except Exception as e:
            result["note"] = str(e)

    with _gpu_cache_lock:
        _gpu_cache = result
    return jsonify(result)


_gpu_cache      = None
_gpu_cache_lock = threading.Lock()


@app.get("/json")
def serve_json():
    """
    Serve a JSON result file (e.g. summary.json) from an allowed output directory.
    GET /json?path=/abs/path/to/summary.json
    """
    path = request.args.get("path", "").strip()
    if not path:
        abort(400, "path required")
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        abort(404, "file not found")
    if not path.endswith(".json"):
        abort(403, "only .json files allowed")
    allowed_roots = [
        os.path.realpath(_OUTPUT_DIR),
        os.path.realpath(tempfile.gettempdir()),
    ]
    if not any(path.startswith(root + os.sep) or path == root for root in allowed_roots):
        abort(403, "access denied")
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/demo")
def serve_demo():
    """Serve the bundled default_t1.nii sample scan."""
    for name in ("default_t1.nii", "default_t1.nii.gz"):
        demo_path = os.path.join(_DATA_DIR, name)
        if os.path.isfile(demo_path):
            return send_file(demo_path, mimetype="application/octet-stream")
    return jsonify({"error": "demo scan not found"}), 404




@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    fname = os.path.basename((f.filename or "upload.nii.gz").replace("\\", "/"))
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        fname += ".nii.gz"
    tmp_dir = os.path.join(tempfile.gettempdir(), "neuroflux_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, fname)
    f.save(save_path)
    return jsonify({"path": save_path})

def _stem(input_path: str) -> str:
    """Return filename without .nii or .nii.gz extension."""
    name = os.path.basename(input_path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


@app.post("/analyze_fov")
def analyze_fov():
    """
    Detect and apply brain FOV pre-crop on a previously-uploaded NIfTI.
    Call this after /upload, BEFORE /segment, so all numpy arrays are freed
    long before TensorFlow starts loading model weights.

    Body: {"path": "/abs/path/to/file.nii.gz", "margin_mm": 25}
    Returns: {
      "cropped":     bool,
      "path":        str,        path to cropped (or original) file
      "removed_pct": float,
      "si_axis":     int|null,   0=X 1=Y 2=Z
      "orig_shape":  [X, Y, Z],
      "crop_start":  int|null,
      "crop_end":    int|null,
    }
    """
    body   = request.get_json(force=True, silent=True) or {}
    path   = body.get("path", "").strip()
    margin = float(body.get("margin_mm", 25.0))

    if not path or not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 400

    from neuroflux.segment import analyze_and_crop_fov

    fov_dir = os.path.join(tempfile.gettempdir(), "neuroflux_fov")
    try:
        result = analyze_and_crop_fov(path, fov_dir, margin_mm=margin)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.post("/fov_profile")
def fov_profile():
    """
    Return cross-sectional area profiles for all three axes of a NIfTI file.
    Used by the browser FOV crop UI to draw the profile graph.

    Body: {"path": "/abs/path/to/file.nii.gz"}
    Returns: {"profiles": [[...],[...],[...]], "si_axis": int,
              "shape": [X,Y,Z], "voxel_mm": [dx,dy,dz]}
    """
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 400
    from neuroflux.segment import get_fov_profiles
    try:
        result = get_fov_profiles(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


@app.post("/manual_crop")
def manual_crop():
    """
    Apply a user-specified FOV crop along a given axis.

    Body: {
      "path":      "/abs/path/to/file.nii.gz",
      "si_axis":   int,   0/1/2
      "start_vox": int,   first voxel to keep (inclusive)
      "end_vox":   int,   last voxel to keep  (inclusive)
    }
    Returns: same structure as /analyze_fov
    """
    body      = request.get_json(force=True, silent=True) or {}
    path      = body.get("path", "").strip()
    si_axis   = body.get("si_axis")
    start_vox = body.get("start_vox")
    end_vox   = body.get("end_vox")

    if not path or not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 400

    fov_dir  = os.path.join(tempfile.gettempdir(), "neuroflux_fov")
    os.makedirs(fov_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(path))[0].removesuffix(".nii")
    out_path = os.path.join(fov_dir, f"{stem}_manual_crop.nii.gz")

    crops = body.get("crops")
    try:
        if crops:
            from neuroflux.segment import manual_crop_fov_multi
            result = manual_crop_fov_multi(path, out_path, crops)
        else:
            if si_axis is None or start_vox is None or end_vox is None:
                return jsonify({"error": "crops array or si_axis/start_vox/end_vox required"}), 400
            from neuroflux.segment import manual_crop_fov
            result = manual_crop_fov(path, out_path, int(si_axis), int(start_vox), int(end_vox))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.post("/check_seg")
def check_seg():
    """Check whether a previous segmentation exists for this file."""
    body      = request.get_json(force=True, silent=True) or {}
    filename  = body.get("filename", "").strip()

    if not filename:
        return jsonify({"exists": False})

    # Check for segmentation files in output/segmentation/<stem>/
    stem    = _stem(filename)
    seg_dir = os.path.join(_OUTPUT_DIR, "segmentation", stem)

    seg_full  = os.path.join(seg_dir, "seg_full.nii.gz")
    original  = os.path.join(seg_dir, "original.nii.gz")
    seg_hemi  = os.path.join(seg_dir, "seg_hemi.nii.gz")
    summary   = os.path.join(seg_dir, "summary.json")

    if os.path.isfile(seg_full) and os.path.isfile(original):
        outputs = {
            "seg_full":  seg_full,
            "original":  original,
        }
        seg_fs = os.path.join(seg_dir, "seg_fs_labels.nii.gz")
        if os.path.isfile(seg_hemi):   outputs["seg_hemi"]  = seg_hemi
        if os.path.isfile(summary):    outputs["summary"]   = summary
        if os.path.isfile(seg_fs):     outputs["seg_fs"]    = seg_fs
        return jsonify({
            "exists":  True,
            "outputs": outputs,
        })
    return jsonify({"exists": False})


@app.post("/segment")
def segment():
    body       = request.get_json(force=True, silent=True) or {}
    input_path = body.get("input_path", "").strip()
    output_dir = body.get("output_dir", "").strip() or None

    # SynthSeg 2.0 options
    robust         = bool(body.get("robust",         False))
    fast           = bool(body.get("fast",           False))
    threads        = max(1, int(body.get("threads",  1)))
    ct             = bool(body.get("ct",             False))
    low_memory     = bool(body.get("low_memory",     False))
    skip_fov_crop  = bool(body.get("skip_fov_crop",  False))

    if not input_path:
        return jsonify({"error": "input_path is required"}), 400
    if not os.path.isfile(input_path):
        return jsonify({"error": f"File not found: {input_path}"}), 400

    # Output dir: output/segmentation/<stem>/
    if output_dir is None:
        stem       = _stem(input_path)
        output_dir = os.path.join(_OUTPUT_DIR, "segmentation", stem)

    os.makedirs(output_dir, exist_ok=True)

    # Purge expired jobs from registry
    _cleanup_old_jobs()

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status":     "running",
            "queue":      queue.Queue(),
            "outputs":    {},
            "output_dir": output_dir,
            "created":    time.time(),
            "proc":       None,          # filled by _run_job after Popen
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_dir),
        kwargs={
            "robust": robust, "fast": fast, "threads": threads,
            "ct": ct, "low_memory": low_memory, "skip_fov_crop": skip_fov_crop,
        },
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "output_dir": output_dir})


@app.get("/job_status/<job_id>")
def job_status(job_id: str):
    """Simple JSON status check (non-SSE) for polling after connection loss."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "unknown"}), 404
    return jsonify({
        "status":  job.get("status", "running"),
        "outputs": job.get("outputs", {}),
    })


@app.delete("/segment/<job_id>")
def cancel_job(job_id: str):
    """Cancel a running segmentation job. Terminates the subprocess."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
        return jsonify({"cancelled": True})
    return jsonify({"cancelled": False, "note": "job already finished"})


@app.get("/status/<job_id>")
def status_stream(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    def generate():
        q = job["queue"]
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                # keep-alive comment so the browser doesn't close the connection
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/mask")
def get_mask():
    """
    Extract a binary mask for one label from seg_full.nii.gz and return
    it as a valid uncompressed NIfTI file.
    GET /mask?path=/abs/seg_full.nii.gz&label=1
    """
    seg_path = request.args.get("path", "").strip()
    label    = request.args.get("label", "")

    if not seg_path or not os.path.isfile(seg_path):
        abort(400, "path not found")
    try:
        label = int(label)
    except ValueError:
        abort(400, "label must be an integer")

    # Optional: exclude specific connected components
    excl_raw = request.args.get("excluded", "").strip()
    excluded_ids = set()
    if excl_raw:
        try:
            excluded_ids = {int(x) for x in excl_raw.split(",") if x.strip()}
        except ValueError:
            pass

    try:
        img     = nib.load(seg_path)
        arr     = np.round(img.get_fdata(dtype=np.float32)).astype(np.int32)
        mask    = (arr == label).astype(np.uint8)

        # Remove excluded components using connected component labeling
        if excluded_ids:
            from scipy.ndimage import label as nd_label
            # Get ALL labels' combined mask for component detection
            all_labels_raw = request.args.get("all_labels", "").strip()
            if all_labels_raw:
                try:
                    all_lbls = {int(x) for x in all_labels_raw.split(",") if x.strip()}
                except ValueError:
                    all_lbls = {label}
            else:
                all_lbls = {label}
            combined = np.zeros_like(arr, dtype=np.uint8)
            for lbl in all_lbls:
                combined[arr == lbl] = 1
            labeled, _ = nd_label(combined)
            # Zero out excluded components in this tissue's mask
            for comp_id in excluded_ids:
                mask[labeled == comp_id] = 0

        out_img = nib.Nifti1Image(mask, img.affine, img.header)
        out_img.header.set_data_dtype(np.uint8)

        buf = io.BytesIO()
        # Save as uncompressed .nii so the browser doesn't need to gunzip
        file_map = out_img.make_file_map({"image": buf, "header": buf})
        out_img.to_file_map(file_map)
        buf.seek(0)

        return Response(buf.read(), mimetype="application/octet-stream")
    except Exception as e:
        abort(500, str(e))


@app.post("/refine")
def refine():
    """
    Intensity-based boundary correction of tissue labels.

    For each tissue pair (e.g. GM/WM), voxels at the boundary between
    the two tissues are reclassified based on which tissue's T1 intensity
    distribution they match more closely (Gaussian log-likelihood).

    Body: {
      "seg_path":       "/path/to/seg_full.nii.gz",
      "nifti_path":     "/path/to/original.nii.gz",
      "pairs":          [[2,3], [2,4], ...],   # label pairs to correct
      "boundary_width": 3,                      # dilation radius in voxels
      "aggressiveness": 0.5                     # 0.1=conservative, 0.9=aggressive
    }
    Returns: { "refined_path": "...", "changed_voxels": N }
    """
    body     = request.get_json(force=True, silent=True) or {}
    seg_path   = body.get("seg_path",   "").strip()
    nifti_path = body.get("nifti_path", "").strip()
    pairs      = body.get("pairs", [[2, 3], [2, 4]])
    bw         = max(1, min(20, int(body.get("boundary_width", 3))))
    aggr       = max(0.05, min(0.95, float(body.get("aggressiveness", 0.5))))

    for p in [seg_path, nifti_path]:
        if not p or not os.path.isfile(p):
            return jsonify({"error": f"File not found: {p}"}), 400

    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure
    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {e}"}), 500

    seg_img  = nib.load(seg_path)
    t1_img   = nib.load(nifti_path)
    seg_arr  = np.asarray(seg_img.dataobj, dtype=np.uint8).copy()
    t1_arr   = np.asarray(t1_img.dataobj, dtype=np.float32)

    # Resample T1 to seg space if shapes differ (simple nearest-neighbour via affine)
    if t1_arr.shape != seg_arr.shape:
        try:
            from scipy.ndimage import affine_transform
            seg_affine = seg_img.affine
            t1_affine  = t1_img.affine
            # Map from seg voxel coords → T1 voxel coords
            M   = np.linalg.inv(t1_affine) @ seg_affine
            mat = M[:3, :3]
            off = M[:3,  3]
            t1_arr = affine_transform(t1_arr, mat, offset=off,
                                      output_shape=seg_arr.shape,
                                      order=1, mode='nearest').astype(np.float32)
        except Exception as e:
            return jsonify({"error": f"T1 resampling failed: {e}"}), 500

    struct = generate_binary_structure(3, 1)   # 6-connectivity
    total_changed = 0

    for lbl_a, lbl_b in pairs:
        lbl_a, lbl_b = int(lbl_a), int(lbl_b)
        mask_a = seg_arr == lbl_a
        mask_b = seg_arr == lbl_b

        if not mask_a.any() or not mask_b.any():
            continue

        # Boundary zone: voxels of label_a within bw voxels of label_b, and vice versa
        dil_a = binary_dilation(mask_a, structure=struct, iterations=bw)
        dil_b = binary_dilation(mask_b, structure=struct, iterations=bw)
        boundary = (dil_a & mask_b) | (dil_b & mask_a)

        # "Core" voxels: inside the mask but NOT in the boundary zone
        core_a = mask_a & ~boundary
        core_b = mask_b & ~boundary

        if core_a.sum() < 10 or core_b.sum() < 10:
            continue

        # Estimate Gaussian parameters from core voxels
        int_a = t1_arr[core_a].astype(np.float64)
        int_b = t1_arr[core_b].astype(np.float64)
        mu_a, std_a = int_a.mean(), max(int_a.std(), 1e-6)
        mu_b, std_b = int_b.mean(), max(int_b.std(), 1e-6)

        # aggressiveness shifts the decision boundary
        # 0.5 = pure Bayes; < 0.5 = favour keeping original; > 0.5 = more swaps
        log_prior_ab = np.log(aggr / (1.0 - aggr + 1e-9))

        bnd_intensities = t1_arr[boundary].astype(np.float64)
        bnd_indices     = np.argwhere(boundary)

        # Log-likelihood ratio: log p(x|a) - log p(x|b)
        ll_a = -0.5 * ((bnd_intensities - mu_a) / std_a) ** 2 - np.log(std_a)
        ll_b = -0.5 * ((bnd_intensities - mu_b) / std_b) ** 2 - np.log(std_b)
        llr  = ll_a - ll_b + log_prior_ab   # > 0 → assign to a, < 0 → assign to b

        current_labels = seg_arr[boundary]
        new_labels      = np.where(llr > 0, lbl_a, lbl_b).astype(np.uint8)
        changed_mask    = new_labels != current_labels
        total_changed  += int(changed_mask.sum())

        # Apply only changed voxels
        bnd_changed = bnd_indices[changed_mask]
        if bnd_changed.shape[0] > 0:
            seg_arr[bnd_changed[:,0], bnd_changed[:,1], bnd_changed[:,2]] = \
                new_labels[changed_mask]

    # ── Post-processing: fill holes created by boundary correction ────────
    from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

    affected_labels = list(set(int(lbl) for pair in pairs for lbl in pair))

    # 1. True 3D hole fill per affected label:
    #    binary_fill_holes fills internal cavities (background not reachable
    #    from the array border). Only claim voxels that are currently label-0.
    for lbl in affected_labels:
        mask    = (seg_arr == lbl)
        filled3d = binary_fill_holes(mask)
        interior = filled3d & ~mask & (seg_arr == 0)
        seg_arr[interior] = lbl

    # 2. For any remaining background gaps that are completely enclosed by
    #    tissue (e.g. label ≠ 0 on all sides but not caught above because
    #    multiple labels surround the gap), use nearest-label assignment.
    #    We identify enclosed background by flood-filling from the border.
    bg_mask  = (seg_arr == 0)
    if bg_mask.any():
        # background connected to the image border = external, keep as 0
        # background NOT connected to border = internal gap → fill
        filled_bg   = binary_fill_holes(~bg_mask)   # True where tissue OR enclosed gap
        enclosed_bg = filled_bg & bg_mask            # True only for enclosed gaps
        if enclosed_bg.any():
            tissue_mask = seg_arr > 0
            _, nearest  = distance_transform_edt(~tissue_mask, return_indices=True)
            seg_arr[enclosed_bg] = seg_arr[
                nearest[0][enclosed_bg],
                nearest[1][enclosed_bg],
                nearest[2][enclosed_bg],
            ]

    # 3. Morphological closing to bridge any hairline cracks (1-2 voxel gaps)
    full_mask = seg_arr > 0
    closed    = binary_closing(full_mask, structure=struct, iterations=1)
    cracks    = closed & ~full_mask
    if cracks.any():
        tissue_mask = seg_arr > 0
        _, nearest  = distance_transform_edt(~tissue_mask, return_indices=True)
        seg_arr[cracks] = seg_arr[
            nearest[0][cracks], nearest[1][cracks], nearest[2][cracks]
        ]

    # Save refined segmentation next to original
    seg_dir      = os.path.dirname(seg_path)
    refined_path = os.path.join(seg_dir, "seg_full_refined.nii.gz")
    nib.save(nib.Nifti1Image(seg_arr, seg_img.affine, seg_img.header), refined_path)

    return jsonify({"refined_path": refined_path, "changed_voxels": total_changed})


def _build_mesh_response(seg_arr, affine, labels,
                          excluded_comps=None, painted_points_mm=None,
                          brush_radius_mm=6.0, small_threshold=None):
    """
    Shared mesh-building logic for /preview3d.
    Now includes per-tissue vertex coloring for the 3D preview.
    """
    import trimesh
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as nd_label
    from skimage.measure import marching_cubes

    excluded_comps = excluded_comps or set()

    # Build combined binary float mask
    combined = np.zeros_like(seg_arr, dtype=np.float32)
    for _, lbl in labels.items():
        combined[seg_arr == int(lbl)] = 1.0

    # Remove excluded connected components
    if excluded_comps:
        binary = (combined > 0)
        lbl_mask, _ = nd_label(binary)
        for cid in excluded_comps:
            combined[lbl_mask == cid] = 0.0

    # Apply paint erasure
    if painted_points_mm:
        inv_affine = np.linalg.inv(affine)
        pts = np.array(painted_points_mm, dtype=np.float64)
        pts_hom = np.column_stack([pts, np.ones(len(pts))])
        pts_vox  = (pts_hom @ inv_affine.T)[:, :3]
        vox_size = float(np.abs(np.diag(affine[:3, :3])).mean())
        brush_vox = brush_radius_mm / vox_size
        shape = combined.shape
        for pv in pts_vox:
            x0 = max(0, int(pv[0] - brush_vox))
            x1 = min(shape[0], int(pv[0] + brush_vox) + 1)
            y0 = max(0, int(pv[1] - brush_vox))
            y1 = min(shape[1], int(pv[1] + brush_vox) + 1)
            z0 = max(0, int(pv[2] - brush_vox))
            z1 = min(shape[2], int(pv[2] + brush_vox) + 1)
            xi, yi, zi = np.mgrid[x0:x1, y0:y1, z0:z1]
            dist_sq = (xi - pv[0])**2 + (yi - pv[1])**2 + (zi - pv[2])**2
            combined[x0:x1, y0:y1, z0:z1][dist_sq <= brush_vox**2] = 0.0

    # Recompute components on mask
    binary_clean    = (combined > 0)
    labeled, n_comp = nd_label(binary_clean)

    # Remove tiny components (< 100 vox) from list and data
    MIN_PREVIEW_VOXELS = 100
    sizes = np.bincount(labeled.ravel())
    for cid in range(1, n_comp + 1):
        if sizes[cid] < MIN_PREVIEW_VOXELS:
            combined[labeled == cid] = 0
            labeled[labeled == cid] = 0
    binary_clean = combined > 0
    labeled, n_comp = nd_label(binary_clean)

    comp_info = []
    for i in range(1, n_comp + 1):
        comp_info.append({"id": i, "voxel_count": int((labeled == i).sum())})
    comp_info.sort(key=lambda x: -x["voxel_count"])
    main_vox = comp_info[0]["voxel_count"] if comp_info else 1
    thresh   = small_threshold if small_threshold is not None else int(main_vox * 0.005)
    for c in comp_info:
        c["is_main"]  = c["id"] == comp_info[0]["id"]
        c["is_small"] = c["voxel_count"] < thresh

    # ── Per-tissue meshing for preview ──────────────────────────────────────
    # Mesh each tissue label separately to avoid inter-tissue boundary artefacts,
    # then merge all into one trimesh for the preview renderer.
    # This is the same principle as the STL export pipeline.
    from trimesh import smoothing as tri_smooth

    # Tissue-specific sigma — matches STL export pipeline defaults
    # GM (2) and cerebellum (6) use lower sigma to preserve sulci/folia
    TISSUE_SIGMA = {1: 0.8, 2: 0.5, 3: 0.8, 4: 0.7, 5: 0.85, 6: 0.55}
    TISSUE_SIGMA_DEFAULT = 0.8  # fallback for unknown labels

    vox2mm = affine[:3, :3]
    origin = affine[:3,  3]

    tissue_meshes  = []   # (trimesh, tissue_label)
    unique_labels  = [int(lbl) for lbl in set(labels.values()) if int(lbl) > 0]

    if not unique_labels:
        return {"error": "No labels selected"}

    for lbl in unique_labels:
        mask = (seg_arr == lbl).astype(np.float32)
        if mask.sum() < 100:
            continue
        sigma = TISSUE_SIGMA.get(lbl, TISSUE_SIGMA_DEFAULT)
        smoothed = gaussian_filter(mask, sigma=sigma)
        if smoothed.max() < 0.3:
            continue
        try:
            verts, faces_mc, _, _ = marching_cubes(smoothed, level=0.5)
        except Exception:
            continue
        verts_mm = (verts @ vox2mm.T) + origin
        # process=False — preserve all faces, fix manually below
        m = trimesh.Trimesh(vertices=verts_mm, faces=faces_mc, process=False)
        try:
            tri_smooth.filter_taubin(m, lamb=0.5, nu=0.53, iterations=10)
            m.remove_degenerate_faces()
            m.remove_duplicate_faces()
            trimesh.repair.fix_normals(m)
            # Multiple fill_holes passes for deep sulci
            for _ in range(3):
                if m.is_watertight:
                    break
                trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
        except Exception:
            pass
        tissue_meshes.append((m, lbl))

    if not tissue_meshes:
        return {"error": "No mesh surface found"}

    # Build per-vertex tissue array BEFORE merge (trimesh concatenate preserves vertex order)
    vertex_tissue_parts = []
    mesh_parts = []
    for m, lbl in tissue_meshes:
        vertex_tissue_parts.append(np.full(len(m.vertices), lbl, dtype=np.int32))
        mesh_parts.append(m)

    mesh_obj = trimesh.util.concatenate(mesh_parts) if len(mesh_parts) > 1 else mesh_parts[0]

    # Decimate for preview (keep responsive)
    if len(mesh_obj.faces) > 150_000:
        try:
            mesh_obj = mesh_obj.simplify_quadric_decimation(face_count=150_000)
        except Exception:
            pass

    # Assign vertex tissue + component IDs via inverse affine lookup into seg_arr
    # This is the most reliable approach — works after decimation too.
    # For vertices landing on background (0), search 6-connected neighbours.
    try:
        inv_affine = np.linalg.inv(affine)
        v_hom = np.column_stack([mesh_obj.vertices,
                                  np.ones(len(mesh_obj.vertices))])
        v_vox  = (v_hom @ inv_affine.T)[:, :3]
        vi = np.clip(np.round(v_vox[:, 0]).astype(int), 0, seg_arr.shape[0] - 1)
        vj = np.clip(np.round(v_vox[:, 1]).astype(int), 0, seg_arr.shape[1] - 1)
        vk = np.clip(np.round(v_vox[:, 2]).astype(int), 0, seg_arr.shape[2] - 1)
        vertex_tissue = seg_arr[vi, vj, vk].astype(np.int32)
        # Fill background hits by searching adjacent voxels
        bg = vertex_tissue == 0
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            if not bg.any():
                break
            ni = np.clip(vi + dx, 0, seg_arr.shape[0]-1)
            nj = np.clip(vj + dy, 0, seg_arr.shape[1]-1)
            nk = np.clip(vk + dz, 0, seg_arr.shape[2]-1)
            t  = seg_arr[ni, nj, nk]
            fix = bg & (t > 0)
            vertex_tissue[fix] = t[fix]
            bg = vertex_tissue == 0
        vertex_comp = labeled[
            np.clip(vi, 0, labeled.shape[0]-1),
            np.clip(vj, 0, labeled.shape[1]-1),
            np.clip(vk, 0, labeled.shape[2]-1)
        ].astype(np.int32)
    except Exception:
        vertex_comp   = np.zeros(len(mesh_obj.vertices), dtype=np.int32)
        vertex_tissue = np.zeros(len(mesh_obj.vertices), dtype=np.int32)

    # Tissue presence index for frontend toggle panel
    tissue_mesh_ranges = {int(lbl): True for _, lbl in tissue_meshes}

    return {
        "vertices":           mesh_obj.vertices.flatten().tolist(),
        "faces":              mesh_obj.faces.flatten().tolist(),
        "vertex_components":  vertex_comp.tolist(),
        "vertex_tissues":     vertex_tissue.tolist(),
        "component_info":     comp_info,
        "n_components":       n_comp,
        "main_vox":           main_vox,
        "small_threshold":    thresh,
        "tissue_mesh_ranges": tissue_mesh_ranges,
    }


@app.post("/preview3d")
def preview3d():
    """
    Return connected component info for the 3D preview sidebar.

    When components_only=true (default for NiiVue-based preview), skips
    mesh generation entirely and just returns component stats.
    Legacy mesh generation is still available when components_only=false.

    Body: {
      "seg_path":        "...",
      "labels":          {...},
      "small_threshold": <int|null>,
      "components_only": true
    }
    """
    body            = request.get_json(force=True, silent=True) or {}
    seg_path        = body.get("seg_path", "").strip()
    labels          = body.get("labels", {})
    components_only = body.get("components_only", True)
    small_threshold = body.get("small_threshold", None)
    if small_threshold is not None:
        small_threshold = int(small_threshold)

    if not seg_path or not os.path.isfile(seg_path):
        return jsonify({"error": "seg_path not found"}), 400

    img     = nib.load(seg_path)
    seg_arr = np.round(img.get_fdata(dtype=np.float32)).astype(np.int32)

    if components_only:
        # Fast path — just compute connected components, no mesh
        from scipy.ndimage import label as nd_label
        combined = np.zeros_like(seg_arr, dtype=np.uint8)
        for lbl in labels.values():
            combined[seg_arr == int(lbl)] = 1
        labeled, n_comp = nd_label(combined)
        sizes = np.bincount(labeled.ravel())
        comp_info = []
        for i in range(1, n_comp + 1):
            comp_info.append({"id": i, "voxel_count": int(sizes[i])})
        comp_info.sort(key=lambda x: -x["voxel_count"])
        main_vox = comp_info[0]["voxel_count"] if comp_info else 1
        thresh   = small_threshold if small_threshold is not None else int(main_vox * 0.005)
        for c in comp_info:
            c["is_main"]  = c["id"] == comp_info[0]["id"]
            c["is_small"] = c["voxel_count"] < thresh
        return jsonify({
            "component_info":  comp_info,
            "n_components":    n_comp,
            "main_vox":        main_vox,
            "small_threshold": thresh,
        })

    # Legacy full mesh path (used by STL export pipeline)
    result = _build_mesh_response(
        seg_arr, img.affine, labels,
        small_threshold=small_threshold,
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.post("/export3d")
def export3d():
    """
    Export selected tissue labels from seg_full.nii.gz as NIfTI or STL.

    NIfTI:  one file per selected tissue  +  one combined file (if >1 selected)
    STL:    all selected tissues merged into a single .stl (no per-tissue suffix)

    Body: {
      "seg_path":  "/path/to/seg_full.nii.gz",
      "labels":    { "csf": 1, "gm": 2, ... },
      "format":    "nifti" | "stl",
      "filename":  "original_file.nii.gz",
      "timestamp": "2024-01-01-12-00-00"
    }
    Returns: { "saved": ["<path>", ...], "dir": "<dir>" }
    """
    body     = request.get_json(force=True, silent=True) or {}
    seg_path = body.get("seg_path", "").strip()
    labels   = body.get("labels", {})
    fmt      = body.get("format", "nifti").lower()

    if not seg_path or not os.path.isfile(seg_path):
        return jsonify({"error": "seg_path not found"}), 400
    if not labels:
        return jsonify({"error": "no labels selected"}), 400
    if fmt not in ("nifti", "stl"):
        return jsonify({"error": "format must be nifti or stl"}), 400

    filename = body.get("filename", "")
    stem     = _stem(filename) if filename else _stem(seg_path)
    ts       = body.get("timestamp", "export")
    out_dir  = os.path.join(_OUTPUT_DIR, "3d-files", stem)
    os.makedirs(out_dir, exist_ok=True)

    img     = nib.load(seg_path)
    seg_arr = np.round(img.get_fdata(dtype=np.float32)).astype(np.int32)
    affine  = img.affine
    header  = img.header.copy()

    saved = []
    label_items = [(k, int(v)) for k, v in labels.items()]
    excluded_comps = set(int(x) for x in body.get("excluded_components", []))

    # ── NIfTI export ──────────────────────────────────────────────────────
    if fmt == "nifti":
        combined = np.zeros_like(seg_arr)

        for key, lbl in label_items:
            mask = (seg_arr == lbl).astype(np.uint8)
            # Remove excluded connected components from mask
            if excluded_comps:
                from scipy.ndimage import label as nd_label
                lbl_mask, _ = nd_label(mask)
                for cid in excluded_comps:
                    mask[lbl_mask == cid] = 0
            combined[mask > 0] = lbl
            # Individual file
            out_path = os.path.join(out_dir, f"{ts}_{key}.nii.gz")
            nib.save(nib.Nifti1Image(mask, affine, header), out_path)
            saved.append(out_path)

        # Combined file (only if more than one tissue selected)
        if len(label_items) > 1:
            out_path = os.path.join(out_dir, f"{ts}_combined.nii.gz")
            nib.save(nib.Nifti1Image(combined, affine, header), out_path)
            saved.append(out_path)

        return jsonify({"saved": saved, "dir": out_dir})

    # ── STL export — streaming SSE progress ─────────────────────────────────
    elif fmt == "stl":
        options = stl_options_from_body(body)

        def _stl_gen_new():
            for event in iter_stl_export(
                seg_arr=seg_arr,
                affine=affine,
                label_items=label_items,
                out_dir=out_dir,
                timestamp=ts,
                excluded_components=excluded_comps,
                options=options,
            ):
                yield f'data: {json.dumps(event)}\n\n'

        return Response(
            _stl_gen_new(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


@app.post("/screenshot")
def save_screenshot():
    """
    Receive PNG data URLs from the browser and save them to
    <server_dir>/screenshots/  (created if missing).

    Body: {
      "timestamp": "2024-01-01T12-00-00",
      "views": {
        "composite":  "data:image/png;base64,...",
        "axial":      "data:image/png;base64,...",
        "sagittal":   "data:image/png;base64,...",
        "coronal":    "data:image/png;base64,...",
        "3d":         "data:image/png;base64,..."
      }
    }
    Returns: { "saved": ["<path>", ...] }
    """
    body = request.get_json(force=True, silent=True) or {}
    ts   = body.get("timestamp", "screenshot").replace(":", "-").replace("T", "_")
    views = body.get("views", {})
    if not views:
        return jsonify({"error": "no views provided"}), 400

    filename = body.get("filename", "")
    stem     = _stem(filename) if filename else "unknown"
    screenshots_dir = os.path.join(_OUTPUT_DIR, "screenshots", stem)
    os.makedirs(screenshots_dir, exist_ok=True)

    saved = []
    for name, data_url in views.items():
        if not data_url or not data_url.startswith("data:image/png;base64,"):
            continue
        raw  = base64.b64decode(data_url.split(",", 1)[1])
        fname = os.path.join(screenshots_dir, f"neuroflux_{ts}_{name}.png")
        with open(fname, "wb") as f:
            f.write(raw)
        saved.append(fname)

    return jsonify({"saved": saved, "dir": screenshots_dir})


@app.get("/file")
def serve_file():
    """
    Serve a NIfTI file to the browser.
    For .nii.gz files: decompress to raw .nii bytes so NiiVue can parse
    them without Content-Encoding negotiation issues.
    Usage: GET /file?path=/absolute/path/to/file.nii.gz
    """
    path = request.args.get("path", "").strip()
    if not path:
        abort(400, "path parameter required")
    if not _safe_path(path):
        abort(403, "Access denied or file not found")

    # Decompress .nii.gz → raw .nii bytes for NiiVue URL loading
    if path.endswith(".nii.gz"):
        try:
            import gzip
            with open(path, "rb") as fh:
                with gzip.open(fh) as gz:
                    raw = gz.read()
            return Response(raw, mimetype="application/octet-stream")
        except Exception as e:
            print(f"[/file] gzip decompress failed ({e}), serving raw", flush=True)
            # Fall through to direct serve (file may already be uncompressed)

    return send_file(path, mimetype="application/octet-stream")


# ── Static file serving ──────────────────────────────────────────────────────
# Serves neuroflux.html and assets (default_t1.nii, etc.) so the entire app
# runs through http://localhost:5050 — no more file:// CORS issues.

@app.get("/")
def serve_index():
    """Serve neuroflux.html as the main page."""
    html_path = os.path.join(_DATA_DIR, "neuroflux.html")
    if os.path.isfile(html_path):
        return send_file(html_path, mimetype="text/html")
    return "<h3>neuroflux.html not found</h3>", 404


@app.get("/stl_file")
def serve_stl_file():
    """
    Serve a generated STL file from an allowed output directory.
    GET /stl_file?path=/abs/path/to/mesh.stl
    """
    path = request.args.get("path", "").strip()
    if not path:
        abort(400, "path required")
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        abort(404, "file not found")
    if not path.endswith(".stl"):
        abort(403, "only .stl files allowed")
    allowed_roots = [
        os.path.realpath(_OUTPUT_DIR),
        os.path.realpath(tempfile.gettempdir()),
    ]
    if not any(path.startswith(root + os.sep) or path == root for root in allowed_roots):
        abort(403, "access denied")
    return send_file(path, mimetype="model/stl")


@app.get("/browse_folder")
def browse_folder():
    """
    Open a native OS folder-picker dialog and return the selected path.
    Uses tkinter (bundled with Python) — works on Windows/macOS/Linux with display.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Select DICOM folder")
        root.destroy()
        return jsonify({"folder": folder or None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/save_file")
def save_file():
    """
    Save raw bytes to 3d-files/<stem>/<filename>, or to a system temp file if temp=true.
    Query params: filename (required), stem (optional), temp (optional bool)
    """
    import tempfile
    filename = request.args.get("filename", "file.nii")
    stem     = request.args.get("stem", "ct_bone")
    use_temp = request.args.get("temp", "false").lower() == "true"
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
        tmp.write(request.data)
        tmp.close()
        return jsonify({"path": tmp.name, "filename": filename, "temp": True})
    out_dir  = os.path.join(_OUTPUT_DIR, "3d-files", stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        f.write(request.data)
    return jsonify({"path": out_path, "filename": filename, "temp": False})


@app.delete("/delete_file")
def delete_file():
    """Delete a file by path (used to clean up temp NIfTIs after STL export)."""
    path = (request.get_json(force=True, silent=True) or {}).get("path", "")
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.post("/ct_bone")
def ct_bone():
    """
    Extract a bone/tissue mask from a CT scan using HU thresholding.

    Body: {
      "ct_path":      "/abs/path/to/ct.nii(.gz)",
      "hu_min":       300,      # lower HU bound  (inclusive)
      "hu_max":       2000,     # upper HU bound  (inclusive)
      "n_components": 1,        # keep N largest components (0 = keep all)
      "z_min_pct":    0.0,      # optional: crop bottom  Z% of volume
      "z_max_pct":    100.0     # optional: crop top     Z% of volume
    }
    Returns raw uncompressed NIfTI bytes (binary uint8 mask).
    """
    from scipy.ndimage import label as nd_label

    body       = request.get_json(force=True, silent=True) or {}
    ct_path    = body.get("ct_path", "").strip()
    hu_min     = float(body.get("hu_min",  300))
    hu_max     = float(body.get("hu_max", 2000))
    n_comp     = int(body.get("n_components", 1))
    z_min_pct  = float(body.get("z_min_pct",   0))
    z_max_pct  = float(body.get("z_max_pct", 100))

    if not ct_path or not os.path.isfile(ct_path):
        return jsonify({"error": "ct_path not found"}), 400

    img = nib.load(ct_path)
    arr = img.get_fdata(dtype=np.float32)

    # Optional Z-slice crop for anatomical region isolation
    nz = arr.shape[2]
    z0 = max(0, int(nz * z_min_pct / 100))
    z1 = min(nz, int(np.ceil(nz * z_max_pct / 100)))
    roi = np.zeros_like(arr, dtype=np.uint8)
    roi[:, :, z0:z1] = ((arr[:, :, z0:z1] >= hu_min) &
                         (arr[:, :, z0:z1] <= hu_max)).astype(np.uint8)

    # Keep N largest connected components (0 = all)
    if n_comp > 0:
        labeled, n = nd_label(roi)
        if n > 0:
            sizes      = np.bincount(labeled.ravel())
            sizes[0]   = 0
            top        = np.argsort(sizes)[::-1][:n_comp]
            roi        = np.zeros_like(roi, dtype=np.uint8)
            for c in top:
                roi[labeled == c] = 1

    out      = nib.Nifti1Image(roi, img.affine, img.header)
    out.header.set_data_dtype(np.uint8)
    buf      = io.BytesIO()
    file_map = out.make_file_map({"image": buf, "header": buf})
    out.to_file_map(file_map)
    return Response(buf.getvalue(), mimetype="application/octet-stream")


@app.post("/dicom_series")
def dicom_series():
    """
    Scan a local folder for DICOM series and return metadata.
    Body: { "folder": "/abs/path" }
    Returns: [ { uid, folder, description, series_number, modality, count } ]
    """
    body   = request.get_json(force=True, silent=True) or {}
    folder = body.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "folder not found"}), 400

    try:
        import pydicom
    except ImportError:
        return jsonify({"error": "pydicom not installed — run: pip install pydicom"}), 500

    import hashlib as _hashlib
    cache_base = os.path.join(_OUTPUT_DIR, "ct_converted")

    series = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for fname in sorted(files):
            if fname.startswith('.'):
                continue
            fpath = os.path.join(root, fname)
            try:
                ds  = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                uid = str(getattr(ds, 'SeriesInstanceUID', '') or '')
                if uid not in series:
                    cache_key = uid.replace('.', '_') if uid else \
                                _hashlib.md5(root.encode('utf-8', errors='replace')).hexdigest()[:16]
                    cache_dir   = os.path.join(cache_base, cache_key)
                    cached_path = None
                    if os.path.isdir(cache_dir):
                        nii_files = [f for f in os.listdir(cache_dir)
                                     if f.endswith('.nii.gz') or f.endswith('.nii')]
                        if nii_files:
                            cached_path = os.path.join(cache_dir, sorted(nii_files)[0])
                    series[uid] = {
                        'uid':           uid,
                        'folder':        root,
                        'description':   str(getattr(ds, 'SeriesDescription', '') or ''),
                        'series_number': str(getattr(ds, 'SeriesNumber',      '') or ''),
                        'modality':      str(getattr(ds, 'Modality',          '') or ''),
                        'count':         0,
                        'cached_path':   cached_path,
                    }
                series[uid]['count'] += 1
            except Exception:
                continue

    result = sorted(series.values(), key=lambda x: x['series_number'])
    return jsonify(result)


@app.post("/dicom_convert_path")
def dicom_convert_path():
    """
    Convert a local DICOM folder directly to NIfTI (no upload needed).
    Body: { "folder": "/abs/path/to/series" }
    Returns: { "path": "/abs/path/to/output.nii.gz", "filename": "..." }
    """
    body   = request.get_json(force=True, silent=True) or {}
    folder = str(body.get("folder") or "").strip()
    uid    = str(body.get("uid")    or "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "folder not found"}), 400

    try:
        import dicom2nifti
    except ImportError:
        return jsonify({"error": "dicom2nifti not installed — run: pip install dicom2nifti"}), 500

    # Persistent cache dir: ct_converted/<key>/
    # Key = uid if available, else MD5 of the folder path (always unique per series)
    import hashlib
    cache_base = os.path.join(_OUTPUT_DIR, "ct_converted")
    if uid:
        cache_key = uid.replace('.', '_')
    else:
        cache_key = hashlib.md5(folder.encode('utf-8', errors='replace')).hexdigest()[:16]
    out_dir = os.path.join(cache_base, cache_key)
    os.makedirs(out_dir, exist_ok=True)

    # Return cached file if it already exists
    existing = sorted(f for f in os.listdir(out_dir) if f.endswith('.nii.gz') or f.endswith('.nii'))
    if existing:
        nii_path = os.path.join(out_dir, existing[0])
        return jsonify({"path": nii_path, "filename": existing[0], "cached": True})

    try:
        dicom2nifti.convert_directory(folder, out_dir, compression=True, reorient=True)
        nii_files = sorted(
            f for f in os.listdir(out_dir)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        )
        if not nii_files:
            return jsonify({"error": "Conversion produced no NIfTI output — "
                                     "is this a valid 3-D DICOM series?"}), 500
        nii_path = os.path.join(out_dir, nii_files[0])
        return jsonify({"path": nii_path, "filename": nii_files[0], "cached": False})
    except Exception as e:
        return jsonify({"error": f"DICOM conversion failed: {e}"}), 500


@app.post("/dicom_convert")
def dicom_convert():
    """
    Accept a multipart upload of DICOM slice files (.dcm), convert them to a
    single NIfTI-GZ file, and return its absolute path.

    POST body: multipart/form-data with field "files" (multiple .dcm files)
    Returns: {"path": "/abs/path/to/converted.nii.gz", "filename": "..."}
    """
    import shutil as _shutil
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    tmp_in  = tempfile.mkdtemp(prefix="nfx_dcm_in_")
    tmp_out = tempfile.mkdtemp(prefix="nfx_dcm_out_")
    try:
        for f in files:
            fname = os.path.basename((f.filename or "slice.dcm").replace("\\", "/")) or "slice.dcm"
            f.save(os.path.join(tmp_in, fname))

        try:
            import dicom2nifti
            dicom2nifti.convert_directory(
                tmp_in, tmp_out, compression=True, reorient=True
            )
        except Exception as e:
            return jsonify({"error": f"DICOM conversion failed: {e}"}), 500

        nii_files = sorted(
            f for f in os.listdir(tmp_out)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        )
        if not nii_files:
            return jsonify({"error": "Conversion produced no NIfTI output — "
                                     "is this a valid 3-D DICOM series?"}), 500

        nii_path = os.path.join(tmp_out, nii_files[0])
        return jsonify({"path": nii_path, "filename": nii_files[0]})

    finally:
        _shutil.rmtree(tmp_in, ignore_errors=True)
        # tmp_out intentionally kept — client will fetch the file from it


@app.get("/sessions")
def list_sessions():
    """
    List previous segmentation sessions from output/segmentation/.
    Returns a JSON array sorted by most-recent modification time first.
    Each entry: { stem, mtime, outputs }
    """
    seg_root = os.path.join(_OUTPUT_DIR, "segmentation")
    if not os.path.isdir(seg_root):
        return jsonify([])
    sessions = []
    for stem in os.listdir(seg_root):
        seg_dir  = os.path.join(seg_root, stem)
        if not os.path.isdir(seg_dir):
            continue
        seg_full = os.path.join(seg_dir, "seg_full.nii.gz")
        original = os.path.join(seg_dir, "original.nii.gz")
        if not (os.path.isfile(seg_full) and os.path.isfile(original)):
            continue
        seg_hemi = os.path.join(seg_dir, "seg_hemi.nii.gz")
        summary  = os.path.join(seg_dir, "summary.json")
        seg_fs   = os.path.join(seg_dir, "seg_fs_labels.nii.gz")
        outputs  = {"original": original, "seg_full": seg_full}
        if os.path.isfile(seg_hemi): outputs["seg_hemi"] = seg_hemi
        if os.path.isfile(summary):  outputs["summary"]  = summary
        if os.path.isfile(seg_fs):   outputs["seg_fs"]   = seg_fs
        sessions.append({
            "stem":    stem,
            "mtime":   os.path.getmtime(seg_full),
            "outputs": outputs,
        })
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return jsonify(sessions)


@app.get("/<path:filename>")
def serve_static(filename):
    """
    Serve static files (.html, .nii, .nii.gz, .js, .css, .png) from the
    server directory.  Only allows safe extensions — not a general file server.
    """
    ALLOWED_EXT = {'.html', '.htm', '.js', '.css', '.json', '.png', '.jpg', '.svg',
                   '.ico', '.nii', '.gz', '.woff', '.woff2', '.ttf'}
    # Security: block path traversal
    safe = os.path.normpath(filename)
    if safe.startswith('..') or safe.startswith('/'):
        abort(403)
    ext = os.path.splitext(safe)[-1].lower()
    if ext not in ALLOWED_EXT:
        abort(403, f"File type {ext} not served")
    # Check bundled data/ directory first (neuroflux.html, default_t1.nii, …)
    # then fall back to the package root for any other served files.
    filepath = os.path.join(_DATA_DIR, safe)
    if not os.path.isfile(filepath):
        filepath = os.path.join(SERVER_DIR, safe)
    if not os.path.isfile(filepath):
        abort(404)
    # Explicit mimetypes so browsers render HTML instead of downloading
    MIME_MAP = {
        '.html': 'text/html', '.htm': 'text/html',
        '.js':   'application/javascript', '.css': 'text/css',
        '.json': 'application/json', '.svg': 'image/svg+xml',
        '.png':  'image/png', '.jpg': 'image/jpeg',
    }
    mime = MIME_MAP.get(ext, 'application/octet-stream')
    return send_file(filepath, mimetype=mime)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Suppress noisy TensorFlow/oneDNN log messages
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    parser = argparse.ArgumentParser(description="NEURO//FLUX segmentation server")
    parser.add_argument("--port", type=int, default=5050, help="Port (default 5050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    args = parser.parse_args()

    print("\n  NEURO//FLUX Segmentation Server")
    print("  ================================")
    print(f"  Running on  http://{args.host}:{args.port}")
    print("")
    print(f"  → Open  http://localhost:{args.port}  in your browser")
    print("")
    print("  Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
