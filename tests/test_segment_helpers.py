"""Fast unit tests for neuroflux.segment helpers (no TensorFlow)."""
import pathlib
import sys

import pytest

from neuroflux.segment import _build_parser, _check_models

_SEG_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "neuroflux" / "segment.py"


class TestCheckModels:
    def test_missing_weights_mentions_setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(tmp_path))
        with pytest.raises(RuntimeError, match="neuroflux-setup"):
            _check_models(False)

    def test_tiny_placeholder_is_corrupt(self, monkeypatch, tmp_path):
        (tmp_path / "synthseg_2.0.h5").write_bytes(b"x" * 100)
        (tmp_path / "synthseg_qc_2.0.h5").write_bytes(b"x" * 100)
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(tmp_path))
        with pytest.raises(RuntimeError, match="Too small"):
            _check_models(False)


class TestCli:
    def test_low_memory_help_does_not_mention_float16(self):
        help_text = _build_parser().format_help()
        assert "float16" not in help_text.lower()
        assert "crop" in help_text.lower() or "RAM" in help_text

    def test_main_clamps_threads_to_at_least_one(self, monkeypatch):
        from neuroflux import segment as seg

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(seg, "run_pipeline", fake_run)
        monkeypatch.setattr(
            sys, "argv", ["neuroflux-segment", "scan.nii", "--threads", "0"]
        )
        seg.main()
        assert captured["threads"] == 1

    def test_systemexit_wrapper_present(self):
        text = _SEG_SRC.read_text(encoding="utf-8")
        assert "except SystemExit" in text
        assert "SynthSeg inference failed" in text
