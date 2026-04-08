"""
TF-backed import tests for neuroflux.synthseg.

These tests verify that the SynthSeg subpackage imports correctly once
TensorFlow is available.  They are automatically skipped when TF is not
installed (e.g. in a lean dev environment without TF).

Marked with @pytest.mark.slow so they can be excluded from quick local runs:
    pytest -m "not slow"    # skip these
    pytest -m slow          # run only these
"""
import pathlib

import pytest

tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow not installed — skipping TF-backed synthseg import tests",
)

pytestmark = pytest.mark.slow

_MODEL_DIR = (
    pathlib.Path(__file__).parent.parent / "models"
)


# ── TF sanity ─────────────────────────────────────────────────────────────────

class TestTensorFlow:
    def test_tf_version_at_least_2_13(self):
        major, minor = (int(x) for x in tf.__version__.split(".")[:2])
        assert (major, minor) >= (2, 13), (
            f"TF {tf.__version__} is too old — need 2.13+ for Apple Silicon support"
        )

    def test_memory_growth_configurable(self):
        """Setting memory growth before first use should not raise."""
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            # Already initialised in this process — expect RuntimeError, not other errors
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # acceptable — device already initialised


# ── Core synthseg imports ─────────────────────────────────────────────────────

class TestSynthSegImports:
    """Each critical module must be importable without error."""

    def test_import_ext_neuron_utils(self):
        from neuroflux.synthseg.ext.neuron import utils  # noqa: F401

    def test_import_ext_neuron_layers(self):
        from neuroflux.synthseg.ext.neuron import layers  # noqa: F401

    def test_import_ext_neuron_models(self):
        from neuroflux.synthseg.ext.neuron import models  # noqa: F401

    def test_import_ext_lab2im_utils(self):
        from neuroflux.synthseg.ext.lab2im import utils  # noqa: F401

    def test_import_ext_lab2im_layers(self):
        from neuroflux.synthseg.ext.lab2im import layers  # noqa: F401

    def test_import_ext_lab2im_edit_volumes(self):
        from neuroflux.synthseg.ext.lab2im import edit_volumes  # noqa: F401

    def test_import_ext_lab2im_edit_tensors(self):
        from neuroflux.synthseg.ext.lab2im import edit_tensors  # noqa: F401

    def test_import_evaluate(self):
        from neuroflux.synthseg import evaluate  # noqa: F401

    def test_import_predict(self):
        from neuroflux.synthseg import predict  # noqa: F401

    def test_import_predict_synthseg(self):
        from neuroflux.synthseg import predict_synthseg  # noqa: F401

    def test_import_predict_qc(self):
        from neuroflux.synthseg import predict_qc  # noqa: F401

    def test_predict_function_callable(self):
        from neuroflux.synthseg.predict_synthseg import predict
        assert callable(predict)


# ── Inference smoke-test (skipped if model weights absent) ───────────────────

@pytest.mark.skipif(
    not (_MODEL_DIR / "synthseg_2.0.h5").is_file(),
    reason="Model weights not present — run 'neuroflux-setup' first",
)
class TestSynthSegInference:
    """
    End-to-end inference on a tiny synthetic NIfTI volume.

    Uses CPU only; passes --threads 1 to keep memory low.
    Runtime: ~60-120 s on CPU without optimised BLAS.
    """

    def test_run_pipeline_cpu(self, tmp_path):
        import numpy as np
        import nibabel as nib
        from neuroflux.segment import run_pipeline

        # 64³ volume — small enough for a CPU smoke test
        data = np.random.randint(0, 300, (64, 64, 64), dtype=np.int16)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        img = nib.Nifti1Image(data, affine)
        input_path = str(tmp_path / "synthetic.nii.gz")
        nib.save(img, input_path)

        outputs = run_pipeline(
            input_path=input_path,
            output_dir=str(tmp_path),
            robust=False,
            fast=True,
            threads=1,
        )

        assert "seg_full" in outputs
        assert pathlib.Path(outputs["seg_full"]).is_file()
        assert "summary" in outputs
        assert pathlib.Path(outputs["summary"]).is_file()
