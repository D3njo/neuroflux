"""Fast unit tests for neuroflux.segment helpers (no TensorFlow)."""
import pathlib
import sys

import nibabel as nib
import numpy as np
import pytest

from neuroflux.segment import (
    _build_parser,
    _check_models,
    _smart_crop,
    effective_tf_threads,
    looks_like_hounsfield,
)

_SEG_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "neuroflux" / "segment.py"


class TestTfThreads:
    def test_windows_pins_to_one(self, monkeypatch):
        monkeypatch.setattr("neuroflux.segment.platform.system", lambda: "Windows")
        assert effective_tf_threads(4) == 1

    def test_other_os_keeps_requested_count(self, monkeypatch):
        monkeypatch.setattr("neuroflux.segment.platform.system", lambda: "Linux")
        assert effective_tf_threads(4) == 4
        assert effective_tf_threads(0) == 1


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


class TestSmartCrop:
    """Low-memory crop is always post-align RAS mm: [LR, AP, SI]."""

    _EXPECTED = [160, 192, 256]

    def _write_nifti(self, path, shape, affine):
        data = np.zeros(shape, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, affine), str(path))

    def test_returns_fixed_ras_mm_for_identity_1mm(self, tmp_path):
        path = tmp_path / "id_1mm.nii.gz"
        self._write_nifti(path, (200, 220, 180), np.eye(4))
        assert _smart_crop(str(path)) == self._EXPECTED

    def test_independent_of_2mm_spacing(self, tmp_path):
        path = tmp_path / "spacing_2mm.nii.gz"
        aff = np.diag([2.0, 2.0, 2.0, 1.0])
        # Small SI voxel count would previously clamp SI below 256
        self._write_nifti(path, (100, 110, 80), aff)
        assert _smart_crop(str(path)) == self._EXPECTED

    def test_independent_of_permuted_affine(self, tmp_path):
        path = tmp_path / "permuted.nii.gz"
        # Voxel axis 0→SI, 1→LR, 2→AP (columns are direction cosines × spacing)
        aff = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self._write_nifti(path, (90, 100, 110), aff)
        assert _smart_crop(str(path)) == self._EXPECTED

    def test_constant_regardless_of_path(self):
        assert _smart_crop("nonexistent_a.nii") == self._EXPECTED
        assert _smart_crop("nonexistent_b.nii.gz") == self._EXPECTED


class TestLooksLikeHounsfield:
    def test_ct_like_negative_and_high_max(self):
        im = np.array([-1000.0, -50.0, 0.0, 40.0, 800.0], dtype=np.float32)
        assert looks_like_hounsfield(im) is True

    def test_mri_like_non_negative(self):
        im = np.linspace(0.0, 1500.0, 100, dtype=np.float32)
        assert looks_like_hounsfield(im) is False

    def test_small_dynamic_range_not_hu(self):
        im = np.array([-5.0, 0.0, 50.0, 100.0], dtype=np.float32)
        assert looks_like_hounsfield(im) is False

    def test_empty_is_false(self):
        assert looks_like_hounsfield(np.array([], dtype=np.float32)) is False

    def test_preprocess_warns_when_ct_off(self):
        """Source contract: preprocess warns but does not set ct=True."""
        pred = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "neuroflux"
            / "synthseg"
            / "predict_synthseg.py"
        )
        text = pred.read_text(encoding="utf-8")
        assert "looks_like_hounsfield" in text
        assert "MRI percentile normalisation will crush soft tissue" in text
        # Must not silently flip the flag
        assert "ct = True" not in text
        assert "ct=True" not in text.split("looks_like_hounsfield")[1].split(
            "# normalise image"
        )[0]
