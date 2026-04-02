"""
Flask API tests for neuroflux.server.
No SynthSeg installation required — segmentation jobs are NOT started.
"""
import io
import json
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# /ping
# ---------------------------------------------------------------------------

class TestPing:
    def test_returns_200(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_body_ok_true(self, client):
        data = resp = client.get("/ping").get_json()
        assert data == {"ok": True}


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_nii_gz(self, client, nii_bytes):
        resp = client.post(
            "/upload",
            data={"file": (io.BytesIO(nii_bytes), "brain.nii.gz")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "path" in body
        assert body["path"].endswith(".nii.gz")

    def test_upload_nii(self, client, nii_bytes):
        resp = client.post(
            "/upload",
            data={"file": (io.BytesIO(nii_bytes), "brain.nii")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["path"].endswith(".nii")

    def test_upload_no_file_returns_400(self, client):
        resp = client.post("/upload", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_upload_non_nii_extension_renamed(self, client, nii_bytes):
        # A file with a non-NIfTI name gets .nii.gz appended
        resp = client.post(
            "/upload",
            data={"file": (io.BytesIO(nii_bytes), "scan.dcm")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        path = resp.get_json()["path"]
        assert path.endswith(".nii.gz")


# ---------------------------------------------------------------------------
# /check_seg
# ---------------------------------------------------------------------------

class TestCheckSeg:
    def test_no_filename_returns_not_exists(self, client):
        resp = client.post("/check_seg", json={})
        assert resp.status_code == 200
        assert resp.get_json()["exists"] is False

    def test_empty_filename_returns_not_exists(self, client):
        resp = client.post("/check_seg", json={"filename": ""})
        assert resp.status_code == 200
        assert resp.get_json()["exists"] is False

    def test_nonexistent_scan_returns_not_exists(self, client):
        resp = client.post("/check_seg", json={"filename": "ghost_scan.nii.gz"})
        assert resp.status_code == 200
        assert resp.get_json()["exists"] is False


# ---------------------------------------------------------------------------
# /segment  (input validation only — does NOT run SynthSeg)
# ---------------------------------------------------------------------------

class TestSegmentValidation:
    def test_missing_input_path_returns_400(self, client):
        resp = client.post("/segment", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_input_path_returns_400(self, client):
        resp = client.post("/segment", json={"input_path": ""})
        assert resp.status_code == 400

    def test_nonexistent_file_returns_400(self, client):
        resp = client.post(
            "/segment",
            json={"input_path": "/totally/nonexistent/file.nii.gz"},
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_file_creates_job(self, client, nii_bytes):
        # Write NIfTI to a real temp path so the server can stat it
        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz", delete=False
        ) as f:
            f.write(nii_bytes)
            tmp = f.name
        try:
            resp = client.post("/segment", json={"input_path": tmp})
            assert resp.status_code == 200
            body = resp.get_json()
            assert "job_id" in body
            assert "output_dir" in body
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# /job_status  and  DELETE /segment/<job_id>
# ---------------------------------------------------------------------------

class TestJobManagement:
    def test_unknown_job_status_returns_404(self, client):
        resp = client.get("/job_status/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_cancel_unknown_job_returns_404(self, client):
        resp = client.delete("/segment/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_created_job_appears_in_status(self, client, nii_bytes):
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(nii_bytes)
            tmp = f.name
        try:
            job_id = client.post(
                "/segment", json={"input_path": tmp}
            ).get_json()["job_id"]
            status_resp = client.get(f"/job_status/{job_id}")
            assert status_resp.status_code == 200
            assert "status" in status_resp.get_json()
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# /file  — path traversal protection
# ---------------------------------------------------------------------------

class TestFileSafety:
    def test_path_traversal_rejected(self, client):
        # Attempt to read a system file outside allowed dirs
        resp = client.get("/file?path=/etc/passwd")
        assert resp.status_code in (403, 404, 400)

    def test_missing_path_param_rejected(self, client):
        resp = client.get("/file")
        assert resp.status_code in (400, 404)

    def test_non_nifti_extension_rejected(self, client):
        # Even if a file exists, .txt is not allowed
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            tmp = f.name
        try:
            resp = client.get(f"/file?path={tmp}")
            assert resp.status_code in (403, 404, 400)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# /json  — summary file serving
# ---------------------------------------------------------------------------

class TestServeJson:
    def test_no_path_returns_400(self, client):
        resp = client.get("/json")
        assert resp.status_code == 400

    def test_nonexistent_json_returns_404(self, client):
        resp = client.get("/json?path=/nonexistent/summary.json")
        assert resp.status_code == 404

    def test_non_json_extension_forbidden(self, client):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"{}")
            tmp = f.name
        try:
            resp = client.get(f"/json?path={tmp}")
            assert resp.status_code in (403, 404)
        finally:
            os.unlink(tmp)
