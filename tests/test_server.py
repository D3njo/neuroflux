"""
Flask API tests for neuroflux.server.
No SynthSeg installation required — segmentation jobs are NOT started.
"""
import io
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# /ping
# ---------------------------------------------------------------------------

class TestPing:
    def test_returns_200(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200

    def test_body_ok_true(self, client):
        data = client.get("/ping").get_json()
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
            body = status_resp.get_json()
            assert "status" in body
            assert "msg" in body
        finally:
            os.unlink(tmp)

    def test_output_dir_is_ignored(self, client, nii_bytes):
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(nii_bytes)
            tmp = f.name
        try:
            body = client.post(
                "/segment",
                json={"input_path": tmp, "output_dir": "/tmp/should_not_use"},
            ).get_json()
            assert "should_not_use" not in body["output_dir"]
            assert "segmentation" in body["output_dir"]
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


# ---------------------------------------------------------------------------
# /gpu
# ---------------------------------------------------------------------------

class TestGpu:
    def test_probes_current_interpreter(self, client, monkeypatch):
        from neuroflux import server as server_mod

        calls = []

        class _Result:
            returncode = 0
            stdout = '{"gpu": false, "count": 0, "name": null}'
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result()

        monkeypatch.setattr(server_mod.subprocess, "run", fake_run)
        server_mod._gpu_cache = None
        data = client.get("/gpu").get_json()
        assert calls, "GPU probe did not run"
        assert calls[0][0] == sys.executable
        assert all("synthseg_env" not in str(c) for c in calls)
        assert data["gpu"] is False
        assert "venv not found" not in (data.get("note") or "")


# ---------------------------------------------------------------------------
# /delete_file  /  /save_file
# ---------------------------------------------------------------------------

class TestFileMutationSafety:
    def test_delete_outside_allowlist_forbidden(self, client):
        resp = client.delete("/delete_file", json={"path": sys.executable})
        assert resp.status_code == 403

    def test_delete_temp_file_ok(self, client):
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
            f.write(b"x")
            tmp = f.name
        try:
            resp = client.delete("/delete_file", json={"path": tmp})
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            assert not os.path.isfile(tmp)
        finally:
            if os.path.isfile(tmp):
                os.unlink(tmp)

    def test_save_file_rejects_bad_stem(self, client):
        resp = client.post(
            "/save_file?filename=a.nii&stem=../evil",
            data=b"nii",
            content_type="application/octet-stream",
        )
        assert resp.status_code == 400

    def test_upload_uses_unique_filename(self, client, nii_bytes):
        a = client.post(
            "/upload",
            data={"file": (io.BytesIO(nii_bytes), "brain.nii.gz")},
            content_type="multipart/form-data",
        ).get_json()["path"]
        b = client.post(
            "/upload",
            data={"file": (io.BytesIO(nii_bytes), "brain.nii.gz")},
            content_type="multipart/form-data",
        ).get_json()["path"]
        assert a != b
        assert a.endswith("brain.nii.gz")
        assert os.path.basename(a) != "brain.nii.gz"
