"""Tests for Keras 2/3 compatibility helpers in the SynthSeg fork."""

import pytest

pytestmark = pytest.mark.slow


def test_tensor_static_shape_on_keras_tensor():
    pytest.importorskip("tensorflow")
    import keras

    from neuroflux.synthseg.keras_compat import tensor_static_dim, tensor_static_shape

    x = keras.Input(shape=(64, 64, 64, 1))
    assert tensor_static_shape(x) == [None, 64, 64, 64, 1]
    assert tensor_static_dim(x, -1) == 1


def test_model_output_tensor_single_output():
    pytest.importorskip("tensorflow")
    import keras

    from neuroflux.synthseg.keras_compat import (
        model_input_tensor,
        model_output_tensor,
        tensor_static_shape,
    )

    inp = keras.Input(shape=(8, 8, 8, 1))
    out = keras.layers.Conv3D(3, 3, padding="same")(inp)
    net = keras.Model(inp, out)
    assert tensor_static_shape(model_output_tensor(net))[-1] == 3
    assert tensor_static_shape(model_input_tensor(net))[-1] == 1


def test_set_keras_shape_no_op_on_plain_tensor():
    import numpy as np

    from neuroflux.synthseg.keras_compat import set_keras_shape

    arr = np.zeros((2, 2))
    set_keras_shape(arr, (2, 2))  # must not raise
