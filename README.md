<div align="center">

# `N E U R O // F L U X`

**⬡ Brain MRI Segmentation ⬡ SynthSeg 2.0 ⬡ Local-First ⬡**

![Python](https://img.shields.io/badge/python-3.10%2B-cyan?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-2.3%2B-cyan?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-magenta?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows%20·%20macOS%20·%20Linux-yellow?style=flat-square)

</div>

---

## ⬡ OVERVIEW

**NEURO//FLUX** is a local-first, browser-based MRI viewer and automated brain segmentation pipeline. Drop in a NIfTI scan, hit **RUN SEGMENTATION**, and get a fully interactive multi-panel viewer with tissue overlays, 3D export, and longitudinal comparison — all running on your own machine with no data leaving it.

Under the hood, segmentation is powered by **SynthSeg 2.0** (Billot et al., Harvard/MGH, PNAS 2023) — a contrast- and resolution-agnostic 3-D U-Net that works out-of-the-box on any MRI without retraining or fine-tuning. SynthSeg runs in its own isolated Python 3.8 / TensorFlow 2.2 environment (subprocess) to keep dependency conflicts away from the main server.

---

## ⬡ FEATURES

### Segmentation
| Mode | Labels | Description |
|------|--------|-------------|
| **Whole Brain** | 6 | CSF · Grey Matter · White Matter · Deep GM · Brainstem · Cerebellum |
| **Hemisphere** | 10 | Full lateral split — GM-L/R · WM-L/R · deep GM-L/R · CC · BS · CB |
| **Structures** | 37 | Individual FreeSurfer structures (raw `seg_fs_labels.nii.gz`) |

**Options:** `--robust` (clinical / low-SNR scans) · `--fast` (~2× speed) · `--ct` (CT Hounsfield mode) · `--threads N`

### Viewer
- **4-panel MRI viewer** — axial / sagittal / coronal / 3-D render via [NiiVue v0.47](https://github.com/niivue/niivue)
- **Client-side NIfTI validation** — warns on bad dimensions or voxel size on load
- **Measurement tools** — Pin A / Pin B + Euclidean distance in mm
- **DICOM import** — upload DICOM slices → auto-convert to NIfTI via `dicom2nifti`

### 3-D Export (STL)
- Combined or per-tissue STL meshes
- Hollow shell with configurable wall thickness
- Sulci enhancement (curvature-based, GM only)
- Cerebellum folia enhancement
- Up to 1.5 M faces
- **Inline STL viewer** (Three.js r134, lazy-loaded) opens after every export

### Workflow
- **Session history** — sidebar lists all completed segmentations sorted by date
- **Load session** — restores all overlays and summary data in one click
- **Longitudinal comparison** — side-by-side volume table with Δ and % change, CSV export
- **Normative reference** — Z-score overlay vs adult normative volumes (31 structures)
- **Batch processing** — queue multiple files for sequential auto-run
- **Intensity refinement** — Gaussian boundary correction between tissue pairs
- **NIfTI export** — per-tissue binary masks

---

## ⬡ ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│  Browser  (neuroflux.html — NiiVue + Three.js + vanilla JS) │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE  (localhost:5050)
┌────────────────────────▼────────────────────────────────┐
│  Flask Server  (server.py)                              │
│  · File upload / DICOM convert                          │
│  · Job queue + Server-Sent Events streaming             │
│  · Mesh generation (trimesh + scikit-image)             │
│  · STL / NIfTI export                                   │
└────────────────────────┬────────────────────────────────┘
                         │ subprocess
┌────────────────────────▼────────────────────────────────┐
│  Segmentation Pipeline  (segment.py)                    │
│  · Calls SynthSeg 2.0 inside isolated synthseg_env/     │
│  · Label remapping  (labels.py)                         │
│  · Hemisphere split                                     │
│  · QC + volume summary  (summary.json)                  │
└────────────────────────┬────────────────────────────────┘
                         │ subprocess
┌────────────────────────▼────────────────────────────────┐
│  SynthSeg 2.0  (synthseg_env/ — Python 3.8 + TF 2.2)    │
│  · SynthSeg_predict.py                                  │
│  · Models extracted from freesurfer/freesurfer:7.4.1    │
└─────────────────────────────────────────────────────────┘
```

**Environments**

| Env | Python | Purpose |
|-----|--------|---------|
| main (`neuro_venv`) | 3.10 | Flask server, nibabel, scipy, trimesh, pydicom |
| `synthseg_env/` | 3.8 | SynthSeg + TensorFlow 2.2 (auto-created by `neuroflux-setup`) |

---

## ⬡ INSTALLATION

### Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/)
- **uv** — fast Python package manager
- **Git** — required for cloning SynthSeg
- **Docker** — for extracting SynthSeg model weights (recommended)
- ~**3 GB** free disk space

**Install uv:**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### Step 1 — Clone and install

```bash
git clone https://github.com/<your-username>/NeuroFlux.git
cd NeuroFlux

# Create main environment
uv venv neuro_venv --python 3.10

# Activate it
source neuro_venv/bin/activate        # macOS / Linux
neuro_venv\Scripts\activate           # Windows

# Install NEUROFLUX
pip install -e .
```

---

### Step 2 — SynthSeg one-time setup

This step creates the isolated SynthSeg environment and downloads model weights
(~256 MB via Docker from `freesurfer/freesurfer:7.4.1`).

**Docker must be running.**

```bash
# Install Python 3.8 via uv (required for SynthSeg / TF 2.2)
uv python install 3.8

# macOS / Linux
neuroflux-setup --python $(uv python find 3.8)

# Windows (PowerShell)
neuroflux-setup --python (uv python find 3.8)

# Windows (cmd)
for /f "delims=" %i in ('uv python find 3.8') do set PY38=%i
neuroflux-setup --python "%PY38%"
```

<details>
<summary>▸ No Docker? Manual model installation</summary>

On a machine that has Docker, run:

```bash
mkdir -p ~/synthseg_models
docker run --rm \
  -v ~/synthseg_models:/output \
  freesurfer/freesurfer:7.4.1 \
  bash -c "cp /usr/local/freesurfer/models/synthseg_2.0.h5 /output/ && \
           cp /usr/local/freesurfer/models/synthseg_robust_2.0.h5 /output/"
```

Copy the files to `src/neuroflux/synthseg_repo/models/`, then run setup skipping the model download:

```bash
neuroflux-setup --python <path/to/python3.8> --skip-models
```

</details>

---

## ⬡ USAGE

### Start the server

```bash
# Activate the main environment first (if not already active)
source neuro_venv/bin/activate   # macOS / Linux
neuro_venv\Scripts\activate      # Windows

neuroflux-server
# → open http://localhost:5050 in your browser
```

Options: `--port 5051` · `--host 0.0.0.0`

### CLI segmentation (headless)

```bash
neuroflux-segment path/to/scan.nii.gz [output_dir] [--robust] [--fast] [--threads 4]
```

### Smoke test

```bash
neuroflux-segment src/neuroflux/data/default_t1.nii
```

---

## ⬡ PROJECT STRUCTURE

```
NeuroFlux/
├── src/
│   └── neuroflux/
│       ├── __init__.py          # package version
│       ├── server.py            # Flask bridge server (port 5050)
│       ├── segment.py           # segmentation pipeline CLI + API
│       ├── labels.py            # FreeSurfer → NEUROFLUX label mapping
│       ├── setup_synthseg.py    # one-time SynthSeg environment setup
│       └── data/
│           ├── neuroflux.html   # full browser UI (NiiVue + Three.js)
│           └── default_t1.nii  # bundled demo scan
├── pyproject.toml               # build config + dependencies
├── .gitignore
└── README.md
```

**Auto-created at runtime (gitignored):**

```
src/neuroflux/
├── synthseg_env/                # Python 3.8 venv — TF 2.2 + SynthSeg
├── synthseg_repo/               # SynthSeg 2.0 source + model weights
└── segmentation/                # output sessions
    └── <scan_stem>/
        ├── original.nii.gz      # 1 mm isotropic T1 resampled by SynthSeg
        ├── seg_full.nii.gz      # 6-class whole-brain segmentation
        ├── seg_hemi.nii.gz      # 10-class hemispheric segmentation
        ├── seg_fs_labels.nii.gz # raw FreeSurfer integer labels
        └── summary.json         # QC scores + per-structure volumes
```

---

## ⬡ SERVER API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ping` | Health check |
| `GET` | `/gpu` | GPU availability probe |
| `GET` | `/demo` | Serve bundled demo scan |
| `POST` | `/upload` | Upload NIfTI file |
| `POST` | `/segment` | Start segmentation job → `{job_id}` |
| `GET` | `/status/<job_id>` | SSE stream of pipeline progress |
| `GET` | `/job_status/<job_id>` | JSON job status (polling fallback) |
| `DELETE` | `/segment/<job_id>` | Cancel running job |
| `POST` | `/check_seg` | Check for existing segmentation |
| `GET` | `/sessions` | List all completed sessions |
| `GET` | `/mask` | Extract single-label binary NIfTI mask |
| `POST` | `/refine` | Intensity-based boundary correction |
| `POST` | `/export3d` | Generate STL mesh |
| `POST` | `/preview3d` | Generate 3-D preview mesh |
| `POST` | `/screenshot` | Render screenshot |
| `POST` | `/dicom_convert` | Convert DICOM series to NIfTI |
| `GET` | `/file` | Serve any output NIfTI by path |
| `GET` | `/<filename>` | Serve static assets |

---

## ⬡ OUTPUT FILES

| File | Description |
|------|-------------|
| `original.nii.gz` | Input scan resampled to 1 mm isotropic by SynthSeg |
| `seg_full.nii.gz` | Whole-brain 6-class tissue segmentation (labels 1–6) |
| `seg_hemi.nii.gz` | Hemispheric 10-class segmentation (labels 1–10) |
| `seg_fs_labels.nii.gz` | Raw FreeSurfer integer label volume (for Structures mode) |
| `summary.json` | QC scores, per-structure volumes, elapsed time, mode |

**`seg_full` label map:**

```
0 = background   1 = CSF          2 = grey matter    3 = white matter
4 = deep GM      5 = brainstem    6 = cerebellum
```

**`seg_hemi` label map:**

```
0 = background   1 = CSF     2 = GM-L    3 = GM-R    4 = WM-L    5 = WM-R
6 = corpus callosum          7 = dGM-L   8 = dGM-R   9 = brainstem   10 = cerebellum
```

---

## ⬡ DEPENDENCIES

**Main environment (Python ≥ 3.10)**

| Package | Purpose |
|---------|---------|
| `flask` + `flask-cors` | Bridge HTTP server |
| `nibabel` | NIfTI I/O |
| `numpy` | Array operations |
| `scipy` | Boundary refinement, resampling |
| `scikit-image` | Marching cubes mesh generation |
| `trimesh` | Mesh processing + STL export |
| `pydicom` + `dicom2nifti` | DICOM import pipeline |

**SynthSeg environment (Python 3.8, auto-created)**

| Package | Version | Notes |
|---------|---------|-------|
| `tensorflow` | 2.2.0 | Pinned — SynthSeg requirement |
| `keras` | 2.3.1 | Pinned |
| `protobuf` | 3.20.3 | Pinned — TF 2.2 compat |
| `nibabel` | ≥ 3.2 | |
| `numpy` | < 1.24 | TF 2.2 constraint |

---

## ⬡ KNOWN ISSUES & NOTES

- **Model weight downloads:** UCL Dropbox / MGH FTP links may be stale. Docker extraction from `freesurfer/freesurfer:7.4.1` is the recommended and most reliable method.
- **Windows GPU:** TF 2.2 has no native Windows GPU support. Use WSL2 + CUDA for GPU acceleration.
- **SynthSeg install warning:** `pip install -e` on `synthseg_repo` may emit a legacy `setup.py develop` deprecation warning — harmless.
- **`neuroflux.html` must be served via the Flask server**, not opened as `file://`. The server enforces CORS and handles all `/file`, `/segment`, and SSE endpoints.

---

## ⬡ LICENSE

MIT — see [LICENSE](LICENSE).

**Third-party components:**
- [SynthSeg](https://github.com/BBillot/SynthSeg) — Apache 2.0 (Billot et al., Harvard/MGH)
- [NiiVue](https://github.com/niivue/niivue) — BSD 2-Clause
- [Three.js](https://threejs.org/) — MIT

---

<div align="center">

```
⬡  NEURO//FLUX  ·  local-first  ·  no data leaves your machine  ⬡
```

</div>
