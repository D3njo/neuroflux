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
        └── TensorFlow 2.15 · Python 3.10 / 3.11
```

Key modules:

| Module | Role |
|--------|------|
| `server.py` | Flask REST API + SSE job streaming |
| `segment.py` | SynthSeg 2.0 pipeline, CLI entry point |
| `labels.py` | FreeSurfer → tissue / hemisphere label remapping |
| `setup_synthseg.py` | Model-weight downloader |
| `synthseg/` | Vendored SynthSeg 2.0 (do **not** edit or lint) |

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

**Python 3.10 / 3.11 only.**  TF 2.15 has no wheels for Python 3.12+.
The CI `test-tf` and `test-inference` jobs are intentionally restricted to 3.10/3.11.

---

## Linting

```bash
ruff check src/ tests/
```

Rules active: `E` (pycodestyle errors), `F` (pyflakes), `W` (pycodestyle warnings), `I` (isort).  
`E501` (line length) and `E701` (single-line if) are suppressed.  
All files under `src/neuroflux/synthseg/` (except `__init__.py`) are excluded from ruff — they are vendored code.

---

## Testing

```bash
# Fast tests — no TensorFlow required (runs on all platforms / Python versions)
python -m pytest tests/ -v -m "not slow"

# TF-backed import tests — requires TF 2.15 (Python 3.10 / 3.11)
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
- **Vendored code** — never modify files in `src/neuroflux/synthseg/ext/` or the vendored predict/evaluate/metrics modules.
- **Version** — `pyproject.toml` and `src/neuroflux/__init__.py` must always declare the same version string.
