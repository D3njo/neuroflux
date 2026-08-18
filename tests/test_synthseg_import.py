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
    def test_tf_version_at_least_2_18(self):
        major, minor = (int(x) for x in tf.__version__.split(".")[:2])
        assert (major, minor) >= (2, 18), (
            f"TF {tf.__version__} is too old — need 2.18+ for Keras 3 / Python 3.12"
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
class TestSynthSegModelParity:
    """Fast weight-load + eager-inference parity checks (no full NIfTI pipeline)."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("robust", [False, True])
    def test_build_and_predict_histogram(self, robust):
        import numpy as np

        from neuroflux.synthseg.ext.lab2im import utils
        from neuroflux.synthseg.predict_synthseg import _model_predict, build_model

        data_dir = (
            pathlib.Path(__file__).parent.parent
            / "src"
            / "neuroflux"
            / "synthseg"
            / "data"
            / "labels_classes_priors"
        )
        labels_seg = np.load(data_dir / "synthseg_segmentation_labels_2.0.npy")
        labels_seg, _ = utils.get_list_labels(label_list=labels_seg)
        labels_seg, _ = np.unique(labels_seg, return_index=True)
        labels_den = np.unique(
            utils.get_list_labels(np.load(data_dir / "synthseg_denoiser_labels_2.0.npy"))[0]
        )

        model_path = _MODEL_DIR / (
            "synthseg_robust_2.0.h5" if robust else "synthseg_2.0.h5"
        )
        bundle = build_model(
            path_model_segmentation=str(model_path),
            path_model_parcellation=None,
            path_model_qc=None,
            input_shape_qc=224,
            labels_segmentation=labels_seg,
            labels_denoiser=labels_den,
            labels_parcellation=None,
            labels_qc=None,
            sigma_smoothing=0.5,
            flip_indices=None,
            robust=robust,
            do_parcellation=False,
            do_qc=False,
        )

        vol = np.random.randn(1, 48, 48, 48, 1).astype(np.float32)
        posteriors, = _model_predict(bundle.seg_net, vol)
        assert posteriors.shape == (1, 48, 48, 48, len(labels_seg))
        assert np.isfinite(posteriors).all()
        # Posteriors should be non-degenerate (not all background)
        assert posteriors.max() > 0.01


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

    @pytest.mark.timeout(600)   # 10 min cap — Windows CPU without MKL can exceed 5 min
    def test_run_pipeline_cpu(self, tmp_path):
        import nibabel as nib
        import numpy as np

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
