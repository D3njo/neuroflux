"""Shared filesystem path helpers for NeuroFlux.

Keep model-weight resolution in one place so ``neuroflux-setup`` writes
weights where ``neuroflux-segment`` will look for them.
"""

from __future__ import annotations

import os
import pathlib

_HERE = pathlib.Path(__file__).parent
_MIN_WEIGHT_BYTES = 1_000_000
_REQUIRED_WEIGHT = "synthseg_2.0.h5"


def repo_root() -> pathlib.Path:
    """Repository / install root (parent of ``src/`` in an editable install)."""
    return (_HERE / ".." / "..").resolve()


def _xdg_models_dir() -> pathlib.Path:
    xdg = pathlib.Path(
        os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local" / "share")
    )
    return xdg / "neuroflux" / "models"


def _candidate_dirs() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    env = os.environ.get("NEUROFLUX_MODELS_DIR")
    if env:
        candidates.append(pathlib.Path(env).expanduser())
    candidates.append(repo_root() / "models")
    candidates.append(_xdg_models_dir())
    return candidates


def _has_required_weights(path: pathlib.Path) -> bool:
    weight = path / _REQUIRED_WEIGHT
    try:
        return weight.is_file() and weight.stat().st_size > _MIN_WEIGHT_BYTES
    except OSError:
        return False


def setup_models_dir() -> pathlib.Path:
    """Writable directory where ``neuroflux-setup`` should download weights."""
    env = os.environ.get("NEUROFLUX_MODELS_DIR")
    if env:
        return pathlib.Path(env).expanduser()

    repo_models = repo_root() / "models"
    try:
        repo_models.mkdir(parents=True, exist_ok=True)
        return repo_models
    except OSError:
        pass

    return _xdg_models_dir()


def resolve_models_dir(*, require_weights: bool = False) -> pathlib.Path:
    """
    Locate the model weights directory.

    Priority:
      1. ``NEUROFLUX_MODELS_DIR`` (always honoured when set)
      2. ``<repo_root>/models``
      3. XDG user data dir (``~/.local/share/neuroflux/models``)

    If ``require_weights`` is True (segmentation) and the env var is unset,
    return the first remaining candidate that contains ``synthseg_2.0.h5``
    larger than 1 MB.  Otherwise return the setup write target so downloads
    land where inference will look.
    """
    env = os.environ.get("NEUROFLUX_MODELS_DIR")
    if env:
        return pathlib.Path(env).expanduser()

    if require_weights:
        for path in _candidate_dirs():
            if path.is_dir() and _has_required_weights(path):
                return path
        return setup_models_dir()
    return setup_models_dir()
