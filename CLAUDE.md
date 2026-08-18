# CLAUDE.md — NeuroFlux project context

Quick reference for AI agents working in this repository.

---

## Architecture

```
Browser (neuroflux.html)
    │ HTTP / SSE (localhost:5050)
Flask Server (server.py)
    │ direct Python call (in-process)
Segmentation Pipeline (segment.py)
    └── SynthSeg 2.0 (bundled: src/neuroflux/synthseg/)
        └── TensorFlow 2.18 · Keras 3 · Python 3.10–3.12
```

Key modules:

| Module | Role |
|--------|------|
| `server.py` | Flask REST API + SSE job streaming |
| `segment.py` | SynthSeg 2.0 pipeline, CLI entry point |
| `labels.py` | FreeSurfer → tissue / hemisphere label remapping |
| `setup_synthseg.py` | Model-weight downloader |
| `synthseg/` | Keras-3-ported SynthSeg 2.0 fork (Billot `.h5` weights) |
| `synthseg/keras_compat.py` | Keras 2/3 shims (`tensor_static_shape`, `set_keras_shape`, …) |

Runtime output lives at `<project_root>/output/` (gitignored):

```
output/
├── segmentation/<scan_stem>/   # NIfTI outputs + summary.json
├── 3d-files/<scan_stem>/       # exported STL meshes
├── ct_converted/               # DICOM → NIfTI conversions
└── screenshots/                # server-side renders
```

---

## Python version constraint

**Python 3.10–3.12.**  TF 2.18 + `tensorflow-metal==1.2.0` is the Apple Silicon stack; TF >2.18 has no reliable Metal wheels.
Python 3.13 is deferred until TF/Metal support is clarified.

---

## Linting

```bash
ruff check src/ tests/
```

Rules active: `E` (pycodestyle errors), `F` (pyflakes), `W` (pycodestyle warnings), `I` (isort).  
`E501` (line length) and `E701` (single-line if) are suppressed.  
Vendored Billot style under `src/neuroflux/synthseg/ext/` remains excluded from ruff; changed fork files may be linted selectively.

---

## Testing

```bash
# Fast tests — no TensorFlow required (runs on all platforms / Python versions)
python -m pytest tests/ -v -m "not slow"

# TF-backed import tests — requires TF 2.18 (Python 3.10–3.12)
python -m pytest tests/ -v -m slow -k "not TestSynthSegInference"

# Full inference smoke-test — requires model weights in models/
python -m pytest tests/test_synthseg_import.py::TestSynthSegInference -v -m slow --timeout=300
```

Download model weights once before running inference tests:

```bash
python -m neuroflux.setup_synthseg
```

---

## Common pitfalls

- **Import order** — ruff enforces isort (`I001`).  Always keep stdlib → third-party → local import blocks sorted.
- **Keras 3** — set `KERAS_BACKEND=tensorflow` before any TF import (`segment._configure_tf` does this).  Do not use `TF_USE_LEGACY_KERAS`.  Inference uses eager `net(..., training=False)` — not `model.predict()` (Metal / 5D BN).
- **BatchNormalization** — Keras 3 rejects `fused=False`; `neuron/models._batch_norm_layer` falls back automatically.
- **Apple Metal** — install `pip install "neuroflux[metal]"` (pins TF 2.18 + metal 1.2.0).
- **Version** — `pyproject.toml` and `src/neuroflux/__init__.py` must always declare the same version string.
