"""
Shared fixtures for NeuroFlux tests.
"""
import io
import json
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from neuroflux.server import app as flask_app


class _DummyProc:
    """Stand-in for subprocess.Popen so fast tests never spawn segment.py."""

    def __init__(self, *args, **kwargs):
        payload = json.dumps({"status": "done", "outputs": {}}) + "\n"
        self.stdout = io.StringIO(payload)
        self.stderr = io.StringIO("")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


@pytest.fixture
def client(monkeypatch):
    from neuroflux import server as server_mod

    with server_mod._jobs_lock:
        server_mod._jobs.clear()
    server_mod._gpu_cache = None
    monkeypatch.setattr(server_mod.subprocess, "Popen", _DummyProc)

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def nii_bytes():
    """Minimal valid NIfTI-1 file as raw bytes (10x10x10 int16 zeros)."""
    data = np.zeros((10, 10, 10), dtype=np.int16)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        tmp_path = f.name
    try:
        nib.save(img, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
