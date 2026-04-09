<div align="center">

# `N E U R O // F L U X`

**⬡ Brain MRI Segmentation ⬡ SynthSeg 2.0 ⬡ Local-First ⬡**

![Python](https://img.shields.io/badge/python-3.10%20·%203.11-cyan?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-cyan?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/flask-2.3%2B-cyan?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-magenta?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows%20·%20macOS%20·%20Linux%20·%20Apple%20Silicon-yellow?style=flat-square)

</div>

---

## ⬡ OVERVIEW

**NEURO//FLUX** is a local-first, browser-based MRI viewer and automated brain segmentation pipeline. Drop in a NIfTI scan, hit **RUN SEGMENTATION**, and get a fully interactive multi-panel viewer with tissue overlays, 3D export, and longitudinal comparison — all running on your own machine with no data leaving it.

Under the hood, segmentation is powered by **SynthSeg 2.0** (Billot et al., Harvard/MGH, PNAS 2023) — a contrast- and resolution-agnostic 3-D U-Net that works out-of-the-box on any MRI without retraining or fine-tuning. SynthSeg is bundled directly inside NeuroFlux and runs **in-process** on **TensorFlow 2.15**, with full **Apple Silicon (M1/M2/M3)** support via `tensorflow-metal`.

---

## ⬡ FEATURES

### Segmentation
| Mode | Labels | Description |
|------|--------|-------------|
| **Whole Brain** | 6 | CSF · Grey Matter · White Matter · Deep GM · Brainstem · Cerebellum |
| **Hemisphere** | 10 | Full lateral split — GM-L/R · WM-L/R · deep GM-L/R · CC · BS · CB |
| **Structures** | 37 | Individual FreeSurfer structures (raw `seg_fs_labels.nii.gz`) |

**Options:** `--robust` (clinical / low-SNR scans) · `--fast` (~2× speed) · `--ct` (CT Hounsfield mode) · `--threads N` · `--low-memory` (6–8 GB systems)

**Automatic brain FOV detection** — before inference, NeuroFlux analyses the cross-sectional area profile of the scan along each axis. The skull/brain produces a distinct peak (large area) while neck and vertex have much smaller cross-sections. The pipeline crops to the contiguous region around that peak, ensuring SynthSeg's center crop lands on the brain even for whole-head scans that include neck, face, or large empty FOV. Scans that are already well-centred are passed through unchanged.

**Visual FOV crop** — interactive 3-panel modal (axial / sagittal / coronal) with drag handles for manual per-axis cropping. Viewport-aware overlays account for NiiVue's aspect-ratio rendering. Undo / clear crop reload the original file.

### Viewer
- **4-panel MRI viewer** — axial / sagittal / coronal / 3-D render via [NiiVue v0.47](https://github.com/niivue/niivue)
- **Client-side NIfTI validation** — warns on bad dimensions or voxel size on load
- **Measurement tools** — Pin A / Pin B + Euclidean distance in mm
- **DICOM import** — upload DICOM slices → auto-convert to NIfTI via `dicom2nifti`

### 3-D Export (STL)
- Combined or per-tissue STL meshes
- **Quality presets** — four named presets tuned for different use-cases:

  | Preset | Best for | What it does |
  |--------|----------|--------------|
  | **Standard** | General use | Tissue-optimised σ + Taubin, moderate sulcal enhancement, HC smoothing |
  | **High Detail** | Resin / SLA, display models | Strong unsharp masking, deep HC passes — maximum sulci & folia |
  | **Print Ready** | FDM / PLA (0.4 mm nozzle) | Gentle polish, mild sulcal detail, optimised layer resolution |
  | **Smooth** | WM, CSF, presentations | Heavy Gaussian blur + Taubin — clean dome-like surfaces |

- **Sulcal/folial enhancement** — unsharp masking in volume space before marching cubes recovers detail blurred by Gaussian pre-smoothing; GM and cerebellum benefit most
- **HC smoothing** (`filter_humphrey`) after Taubin — feature-preserving pass that keeps concavities (sulci, fissures, folia) while reducing surface jaggedness
- **Watertight repair** — degenerate/duplicate face removal, 3-pass hole fill, normal fix; output is slicer-compatible by default
- **Flat base** option — clamps bottom 5th-percentile Z vertices flat for print-bed stability
- Hollow shell with configurable wall thickness
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
                         │ direct Python call (in-process)
┌────────────────────────▼────────────────────────────────┐
│  Segmentation Pipeline  (segment.py)                    │
│  · SynthSeg 2.0 — bundled as neuroflux.synthseg         │
│  · TensorFlow 2.15 — Python 3.10/3.11                   │
│  · Apple Silicon GPU via tensorflow-metal               │
│  · Label remapping  (labels.py)                         │
│  · Hemisphere split                                     │
│  · QC + volume summary  (summary.json)                  │
└─────────────────────────────────────────────────────────┘
```

**Single environment — no isolated venv needed**

| Env | Python | Purpose |
|-----|--------|---------|
| `neuro_venv` | 3.10 / 3.11 | Everything: Flask · SynthSeg · TensorFlow 2.15 |

---

## ⬡ INSTALLATION

### Prerequisites

- **Python 3.10 or 3.11** — [python.org](https://www.python.org/)
  (TF 2.15 has no wheels for Python 3.12+)
- **uv** — fast Python package manager (optional but recommended)
- ~**2 GB** free disk space (model weights + TF)

> **Apple Silicon (M1/M2/M3)?**  
> Install the `[metal]` extra after the main install for GPU acceleration:  
> `uv pip install "neuroflux[metal]"`

**Install uv (optional):**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### Step 1 — Clone and install

```bash
git clone https://github.com/D3njo/NeuroFlux.git
cd NeuroFlux

# Create environment with Python 3.11 (recommended)
uv venv neuro_venv --python 3.11
# — or without uv —
python3.11 -m venv neuro_venv

# Activate
source neuro_venv/bin/activate        # macOS / Linux
neuro_venv\Scripts\activate           # Windows

# Install NeuroFlux (includes SynthSeg + TensorFlow 2.15)
uv pip install -e .

# Apple Silicon: add Metal GPU plugin
uv pip install "neuroflux[metal]"        # M1/M2/M3 only
```

---

### Step 2 — Download model weights

SynthSeg is now bundled inside NeuroFlux — no separate environment needed.
Only the pre-trained model weights (~600 MB total) need to be downloaded once:

```bash
neuroflux-setup
```

<details>
<summary>▸ Manual model installation (offline)</summary>

Place the following `.h5` files into the `models/` folder at the repository root:

```
models/
├── synthseg_1.0.h5
├── synthseg_2.0.h5
├── synthseg_parc_2.0.h5
├── synthseg_qc_2.0.h5
└── synthseg_robust_2.0.h5
```

Then run:

```bash
neuroflux-setup --skip-models   # verify paths without downloading
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
├── models/                      # model weights (downloaded by neuroflux-setup)
│   ├── synthseg_2.0.h5
│   ├── synthseg_robust_2.0.h5
│   └── ...
├── src/
│   └── neuroflux/
│       ├── __init__.py          # package version
│       ├── server.py            # Flask bridge server (port 5050)
│       ├── segment.py           # segmentation pipeline CLI + API
│       ├── labels.py            # FreeSurfer → NEUROFLUX label mapping
│       ├── setup_synthseg.py    # downloads model weights on first run
│       ├── synthseg/            # SynthSeg 2.0 bundled (TF 2.15 / tf.keras)
│       │   ├── predict_synthseg.py
│       │   ├── ext/lab2im/      # image generation utilities
│       │   ├── ext/neuron/      # U-Net layers and models
│       │   └── data/labels_classes_priors/   # label .npy files (bundled)
│       └── data/
│           ├── neuroflux.html   # full browser UI (NiiVue + Three.js)
│           └── default_t1.nii  # bundled demo scan
├── tests/                       # pytest suite (fast + TF-backed)
├── pyproject.toml               # build config + dependencies
└── README.md
```

**Auto-created at runtime (gitignored):**

```
src/neuroflux/
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

**Single environment (Python 3.10 / 3.11)**

| Package | Purpose |
|---------|---------|
| `tensorflow 2.15` | SynthSeg inference (bundled) |
| `flask` + `flask-cors` | Bridge HTTP server |
| `nibabel` | NIfTI I/O |
| `numpy` | Array operations |
| `scipy` | Boundary refinement, resampling |
| `h5py` | Load `.h5` model weights |
| `scikit-image` | Marching cubes mesh generation |
| `trimesh` | Mesh processing + STL export |
| `pydicom` + `dicom2nifti` | DICOM import pipeline |

**Optional extras**

| Extra | Package | Purpose |
|-------|---------|---------|
| `[metal]` | `tensorflow-metal` | Apple Silicon (M1/M2/M3) GPU acceleration |

---

## ⬡ KNOWN ISSUES & NOTES

- **Python 3.12+:** TF 2.15 has no wheels for Python 3.12 or newer. Use Python 3.10 or 3.11. Support for 3.12+ will come once the Keras 3 migration is complete.
- **Apple Silicon:** install `pip install "neuroflux[metal]"` for GPU acceleration. Without it, inference runs on CPU (~1 min/scan).
- **Low RAM (6–8 GB):** Use the **Low Memory** option in the UI (or `--low-memory` on the CLI). This forces fast mode, disables QC, applies a per-axis brain-shaped crop (AP=192 mm, LR/SI=160 mm), skips posteriors file output, and frees intermediate tensors aggressively — cutting peak RAM by ~50%. If the process is still killed by the OS (exit code -9), your system has no swap space. Set up a swapfile:

  <details>
  <summary>▸ Linux / WSL2</summary>

  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile

  # Persist across reboots:
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```

  </details>

  <details>
  <summary>▸ ChromeOS / Crostini (Linux container)</summary>

  Swap cannot be created inside the container. Enable zram-based swap from the ChromeOS shell instead:

  1. Open **crosh**: `Ctrl`+`Alt`+`T` in the Chrome browser
  2. Run: `swap enable 4096`
  3. Reboot ChromeOS

  </details>

  <details>
  <summary>▸ macOS</summary>

  macOS manages swap automatically (dynamic paging). If you hit OOM, free up RAM by closing other applications — you cannot manually add a swapfile on macOS.

  </details>

  <details>
  <summary>▸ Windows</summary>

  Go to **System → Advanced system settings → Performance → Settings → Advanced → Virtual memory**, click *Change*, uncheck *Automatically manage*, set a custom size (e.g. 4096–8192 MB), and reboot.

  </details>
- **Windows GPU:** TF 2.15 supports CUDA on Windows. For CPU-only use, no extra setup is needed.
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
