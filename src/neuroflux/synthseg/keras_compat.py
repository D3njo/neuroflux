"""
Keras 2 / Keras 3 compatibility helpers for the NeuroFlux SynthSeg fork.

TensorFlow 2.18 ships Keras 3 as the default API.  The vendored Billot code was
written against Keras 2 idioms (_keras_shape, model.output shape introspection,
K.switch, fused BatchNorm).  These helpers keep the inference path working on
both stacks without rewriting every layer.
"""

from __future__ import annotations

import tensorflow as tf


def model_output_tensor(model):
    """Return the primary output tensor from a Keras functional model."""
    out = model.output
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def model_input_tensor(model):
    """Return the primary input tensor from a Keras functional model."""
    inp = model.input
    if isinstance(inp, (list, tuple)):
        return inp[0]
    return inp


def tensor_shape_list(tensor):
    """Return ``tensor.shape`` as a plain Python list (Keras 2/3 compatible)."""
    shape = tensor.shape
    if hasattr(shape, "as_list"):
        return list(shape.as_list())
    return list(shape)


def tensor_static_shape(tensor):
    """Return the static shape of a Keras tensor as a plain Python list."""
    shape = getattr(tensor, "shape", None)
    if shape is None and hasattr(tensor, "get_shape"):
        shape = tensor.get_shape()
    if hasattr(shape, "as_list"):
        return list(shape.as_list())
    return list(shape)


def tensor_static_dim(tensor, index):
    return tensor_static_shape(tensor)[index]


def set_keras_shape(tensor, shape):
    """Assign legacy ``_keras_shape`` when the runtime still accepts it."""
    try:
        tensor._keras_shape = tuple(shape)
    except (AttributeError, TypeError):
        pass
    return tensor


def backend_switch(condition, then_expression, else_expression):
    """Scalar branch compatible with legacy ``K.switch``."""
    cond = tf.cast(condition, tf.bool)
    if getattr(cond.shape, "rank", None) not in (None, 0):
        cond = tf.reshape(cond, [])
    return tf.where(cond, then_expression, else_expression)


def backend_epsilon():
    return tf.keras.backend.epsilon()
