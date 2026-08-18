"""Slow tests for Keras acceleration paths (requires TensorFlow + model weights)."""

import pathlib

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

pytestmark = pytest.mark.slow

_MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"


@pytest.mark.skipif(
    not (_MODEL_DIR / "synthseg_2.0.h5").is_file(),
    reason="Model weights not present — run 'neuroflux-setup' first",
)
class TestKerasAccelParity:
    @pytest.mark.timeout(180)
    def test_eager_vs_compiled_seg_outputs_match(self):
        from neuroflux.accel import (
            benchmark_predict,
            configure_acceleration,
            dice_coefficient,
            get_compiled_predictor,
            reset_acceleration_state,
        )
        from neuroflux.synthseg.ext.lab2im import utils
        from neuroflux.synthseg.predict_synthseg import _model_predict, build_model

        reset_acceleration_state()
        configure_acceleration(no_accel=True, tf_module=tf)

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

        bundle = build_model(
            path_model_segmentation=str(_MODEL_DIR / "synthseg_2.0.h5"),
            path_model_parcellation=None,
            path_model_qc=None,
            input_shape_qc=224,
            labels_segmentation=labels_seg,
            labels_denoiser=labels_den,
            labels_parcellation=None,
            labels_qc=None,
            sigma_smoothing=0.5,
            flip_indices=None,
            robust=False,
            do_parcellation=False,
            do_qc=False,
        )

        vol = np.random.randn(1, 64, 64, 64, 1).astype(np.float32)
        bucket = tuple(vol.shape[1:4])

        eager_out, = _model_predict(bundle.seg_net, vol)
        compiled = get_compiled_predictor(bundle.seg_net, bucket)
        compiled_out, = compiled(vol)

        eager_labels = eager_out.argmax(-1)
        compiled_labels = compiled_out.argmax(-1)
        dice = dice_coefficient(eager_labels.ravel(), compiled_labels.ravel())
        assert dice > 0.99

        bench = benchmark_predict(lambda x: compiled(x), vol, warmup=0, repeats=1)
        assert bench.wall_s > 0
        assert bench.output_shapes == (tuple(compiled_out.shape),)

    @pytest.mark.timeout(180)
    def test_qc_split_model_builds(self):
        from neuroflux.accel import configure_acceleration, reset_acceleration_state
        from neuroflux.synthseg.ext.lab2im import utils
        from neuroflux.synthseg.predict_synthseg import build_model

        reset_acceleration_state()
        configure_acceleration(no_accel=True, tf_module=tf)

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
        labels_seg, unique_idx = np.unique(labels_seg, return_index=True)
        labels_den = np.unique(
            utils.get_list_labels(np.load(data_dir / "synthseg_denoiser_labels_2.0.npy"))[0]
        )
        labels_qc = utils.get_list_labels(
            np.load(data_dir / "synthseg_qc_labels_2.0.npy")
        )[0][unique_idx]

        bundle = build_model(
            path_model_segmentation=str(_MODEL_DIR / "synthseg_2.0.h5"),
            path_model_parcellation=None,
            path_model_qc=str(_MODEL_DIR / "synthseg_qc_2.0.h5"),
            input_shape_qc=224,
            labels_segmentation=labels_seg,
            labels_denoiser=labels_den,
            labels_parcellation=None,
            labels_qc=labels_qc,
            sigma_smoothing=0.5,
            flip_indices=None,
            robust=False,
            do_parcellation=False,
            do_qc=True,
        )
        assert bundle.qc_net is not None
        assert bundle.qc_prep is not None
