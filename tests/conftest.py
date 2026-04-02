"""
Shared fixtures for NeuroFlux tests.
"""
import io
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from neuroflux.server import app as flask_app


@pytest.fixture
def client():
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
