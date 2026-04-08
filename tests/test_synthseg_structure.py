"""
Structural tests for the neuroflux.synthseg subpackage.

These tests are intentionally TF-free: they verify that the package layout
is correct and that all expected modules and data files are present, without
triggering a TensorFlow import.  Fast — suitable for every CI run.
"""
import pathlib

import pytest

# Root of the installed package
_PKG_ROOT = pathlib.Path(__file__).parent.parent / "src" / "neuroflux"


# ── Package layout ────────────────────────────────────────────────────────────

class TestSubpackageLayout:
    """The synthseg/ tree must be present and contain all expected files."""

    SS = _PKG_ROOT / "synthseg"

    def test_synthseg_dir_exists(self):
        assert self.SS.is_dir()

    def test_ext_dir_exists(self):
        assert (self.SS / "ext").is_dir()

    def test_lab2im_dir_exists(self):
        assert (self.SS / "ext" / "lab2im").is_dir()

    def test_neuron_dir_exists(self):
        assert (self.SS / "ext" / "neuron").is_dir()

    def test_data_dir_exists(self):
        assert (self.SS / "data" / "labels_classes_priors").is_dir()

    @pytest.mark.parametrize("fname", [
        "__init__.py",
        "predict.py",
        "predict_synthseg.py",
        "predict_qc.py",
        "predict_denoiser.py",
        "evaluate.py",
        "metrics_model.py",
        "labels_to_image_model.py",
        "model_inputs.py",
    ])
    def test_core_module_present(self, fname):
        assert (self.SS / fname).is_file(), f"Missing: synthseg/{fname}"

    @pytest.mark.parametrize("fname", [
        "edit_tensors.py", "edit_volumes.py",
        "lab2im_model.py", "layers.py", "utils.py",
    ])
    def test_lab2im_module_present(self, fname):
        assert (self.SS / "ext" / "lab2im" / fname).is_file()

    @pytest.mark.parametrize("fname", [
        "layers.py", "models.py", "utils.py",
    ])
    def test_neuron_module_present(self, fname):
        assert (self.SS / "ext" / "neuron" / fname).is_file()

    @pytest.mark.parametrize("fname", [
        "synthseg_segmentation_labels_2.0.npy",
        "synthseg_segmentation_names_2.0.npy",
        "synthseg_denoiser_labels_2.0.npy",
        "synthseg_parcellation_labels.npy",
        "synthseg_qc_labels_2.0.npy",
        "synthseg_topological_classes_2.0.npy",
    ])
    def test_label_data_file_present(self, fname):
        assert (self.SS / "data" / "labels_classes_priors" / fname).is_file()


# ── Import hygiene ────────────────────────────────────────────────────────────

class TestNoStandaloneKeras:
    """All keras references in synthseg/ must go through tensorflow.keras."""

    SS = _PKG_ROOT / "synthseg"

    def _py_files(self):
        return list(self.SS.rglob("*.py"))

    def test_no_bare_import_keras(self):
        bad = []
        for fp in self._py_files():
            for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("import keras") or stripped.startswith("from keras"):
                    # Allow 'from tensorflow import keras' or 'from tensorflow.keras'
                    if "tensorflow" not in stripped:
                        bad.append(f"{fp.relative_to(self.SS)}:{i}: {stripped}")
        assert not bad, "Standalone keras imports found:\n" + "\n".join(bad)

    def test_no_absolute_synthseg_imports(self):
        """Internal imports must be relative, not 'from SynthSeg import ...'."""
        bad = []
        for fp in self._py_files():
            for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("from SynthSeg") or stripped.startswith("import SynthSeg"):
                    bad.append(f"{fp.relative_to(self.SS)}:{i}: {stripped}")
        assert not bad, "Absolute SynthSeg imports found:\n" + "\n".join(bad)

    def test_no_absolute_ext_imports(self):
        """'from ext.lab2im' / 'from ext.neuron' must be gone."""
        bad = []
        for fp in self._py_files():
            for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("from ext.") or stripped.startswith("import ext."):
                    bad.append(f"{fp.relative_to(self.SS)}:{i}: {stripped}")
        assert not bad, "Absolute ext.* imports found:\n" + "\n".join(bad)


# ── segment.py hygiene ────────────────────────────────────────────────────────

class TestSegmentModule:
    """segment.py must use package-qualified imports, not bare script imports."""

    SEG = _PKG_ROOT / "segment.py"

    def test_uses_neuroflux_labels_import(self):
        text = self.SEG.read_text(encoding="utf-8")
        assert "from neuroflux.labels import" in text, \
            "segment.py must use 'from neuroflux.labels import', not bare 'from labels import'"

    def test_no_subprocess_synthseg_call(self):
        text = self.SEG.read_text(encoding="utf-8")
        assert "synthseg_env" not in text, \
            "segment.py must not reference the old synthseg_env venv"
        assert "_SS_SCRIPT" not in text, \
            "segment.py must not use the old subprocess script path"

    def test_calls_predict_synthseg_directly(self):
        text = self.SEG.read_text(encoding="utf-8")
        assert "neuroflux.synthseg.predict_synthseg" in text, \
            "segment.py must import predict_synthseg directly"
