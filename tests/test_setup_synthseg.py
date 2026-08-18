"""Tests for neuroflux.setup_synthseg (no network)."""
import pathlib
import sys
import urllib.request

import pytest

from neuroflux.setup_synthseg import MODEL_FILES, _download_models, main


class TestModelFiles:
    def test_does_not_download_synthseg_1_0(self):
        assert "synthseg_1.0.h5" not in MODEL_FILES
        assert "synthseg_2.0.h5" in MODEL_FILES
        assert "synthseg_robust_2.0.h5" in MODEL_FILES
        assert "synthseg_qc_2.0.h5" in MODEL_FILES


class TestSetupCli:
    def test_skip_models(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["neuroflux-setup", "--skip-models"])
        main()
        out = capsys.readouterr().out
        assert "No downloads performed" in out
        assert str(tmp_path) in out


class TestDownload:
    def test_tiny_file_exits(self, monkeypatch, tmp_path):
        def fake_retrieve(url, dest, reporthook=None):
            pathlib.Path(dest).write_bytes(b"tiny")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
        with pytest.raises(SystemExit) as exc:
            _download_models(tmp_path)
        assert exc.value.code == 1
