"""
Adaptive Keras-3 acceleration policy for SynthSeg inference.

Selects XLA / mixed-precision only where they help (NVIDIA CUDA).  CPU and
Apple Metal default to eager FP32 with shape-bucket padding for RAM + graph reuse.
"""

from __future__ import annotations

import enum
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Pad buckets — divisible by 2**5 (U-Net has 5 levels).  Keeps XLA graphs stable.
PAD_BUCKETS: tuple[int, ...] = (160, 192, 224, 256)
_UNET_LEVELS = 5
_BUCKET_DIVISOR = 2 ** _UNET_LEVELS


class AcceleratorKind(enum.Enum):
    CPU = "cpu"
    NVIDIA = "nvidia"
    METAL = "metal"


@dataclass(frozen=True)
class AccelPolicy:
    kind: AcceleratorKind
    use_xla: bool
    use_mixed_precision: bool
    use_bucket_padding: bool = True
    accel_mode: str = "eager"  # "xla" | "eager"
    compute_dtype: str = "float32"  # "float16" | "float32"

    def as_dict(self) -> dict:
        return {
            "accelerator": self.kind.value,
            "accel": self.accel_mode,
            "dtype": self.compute_dtype,
            "bucket_padding": self.use_bucket_padding,
            "mixed_precision": self.use_mixed_precision,
            "xla": self.use_xla,
        }


_POLICY: AccelPolicy | None = None
_PREDICTOR_CACHE: dict[tuple[int, tuple[int, ...] | None], Callable] = {}


def bucket_pad_shape(
    spatial_sizes: list[int] | np.ndarray,
    *,
    min_pad: list[int] | np.ndarray | None = None,
) -> list[int]:
    """Pick the smallest PAD_BUCKET >= each spatial size (and min_pad if set)."""
    sizes = list(spatial_sizes)
    mins = list(min_pad) if min_pad is not None else [0] * len(sizes)
    out: list[int] = []
    for i, size in enumerate(sizes):
        needed = max(int(size), int(mins[i]))
        chosen = None
        for bucket in PAD_BUCKETS:
            if bucket >= needed and bucket % _BUCKET_DIVISOR == 0:
                chosen = bucket
                break
        if chosen is None:
            from neuroflux.synthseg.ext.lab2im import utils

            chosen = utils.find_closest_number_divisible_by_m(
                max(needed, PAD_BUCKETS[-1]), _BUCKET_DIVISOR, "higher"
            )
        out.append(int(chosen))
    return out


def detect_accelerator(tf_module) -> AcceleratorKind:
    """Classify the active TensorFlow GPU backend."""
    gpus = tf_module.config.list_physical_devices("GPU")
    if not gpus:
        return AcceleratorKind.CPU
    name = (gpus[0].name or "").lower()
    if "metal" in name:
        return AcceleratorKind.METAL
    return AcceleratorKind.NVIDIA


def resolve_policy(*, no_accel: bool = False, tf_module=None) -> AccelPolicy:
    """Build the acceleration policy for this process."""
    if no_accel or os.environ.get("NEUROFLUX_NO_ACCEL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return AccelPolicy(
            kind=AcceleratorKind.CPU,
            use_xla=False,
            use_mixed_precision=False,
            accel_mode="eager",
            compute_dtype="float32",
        )

    if tf_module is None:
        import tensorflow as tf

        tf_module = tf

    kind = detect_accelerator(tf_module)

    if kind == AcceleratorKind.NVIDIA:
        xla_env = os.environ.get("NEUROFLUX_XLA", "").strip().lower()
        use_xla = xla_env not in ("0", "false", "no")
        return AccelPolicy(
            kind=kind,
            use_xla=use_xla,
            use_mixed_precision=True,
            accel_mode="xla" if use_xla else "eager",
            compute_dtype="float16" if use_xla else "float32",
        )

    if kind == AcceleratorKind.METAL:
        xla_env = os.environ.get("NEUROFLUX_XLA", "").strip().lower()
        use_xla = xla_env in ("1", "true", "yes")
        return AccelPolicy(
            kind=kind,
            use_xla=use_xla,
            use_mixed_precision=False,
            accel_mode="xla" if use_xla else "eager",
            compute_dtype="float32",
        )

    return AccelPolicy(
        kind=AcceleratorKind.CPU,
        use_xla=False,
        use_mixed_precision=False,
        accel_mode="eager",
        compute_dtype="float32",
    )


def configure_acceleration(*, no_accel: bool = False, tf_module=None) -> AccelPolicy:
    """Detect hardware and apply global Keras policies.  Idempotent."""
    global _POLICY
    if _POLICY is not None:
        return _POLICY

    if tf_module is None:
        import tensorflow as tf

        tf_module = tf

    policy = resolve_policy(no_accel=no_accel, tf_module=tf_module)
    _POLICY = policy

    if policy.use_mixed_precision:
        import keras

        keras.mixed_precision.set_global_policy("mixed_float16")

    return policy


def get_policy() -> AccelPolicy:
    if _POLICY is None:
        return resolve_policy(no_accel=True)
    return _POLICY


def reset_acceleration_state() -> None:
    """Clear cached policy/predictors (tests)."""
    global _POLICY
    _POLICY = None
    _PREDICTOR_CACHE.clear()


def _peak_rss_bytes() -> int | None:
    """Best-effort peak RSS for the current process (Unix only)."""
    if sys.platform == "win32":
        return None
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(ru.ru_maxrss)
        return int(ru.ru_maxrss) * 1024
    except Exception:
        return None

def label_histogram(seg: np.ndarray) -> dict[int, int]:
    """Compact label histogram for parity checks."""
    arr = np.asarray(seg, dtype=np.int32)
    unique, counts = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def dice_coefficient(a: np.ndarray, b: np.ndarray, *, ignore_zero: bool = True) -> float:
    """Mean Dice over labels present in either volume."""
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    labels = np.unique(np.concatenate([a.ravel(), b.ravel()]))
    if ignore_zero:
        labels = labels[labels != 0]
    if labels.size == 0:
        return 1.0
    scores = []
    for lab in labels:
        ma = a == lab
        mb = b == lab
        inter = np.logical_and(ma, mb).sum()
        denom = ma.sum() + mb.sum()
        if denom == 0:
            continue
        scores.append(2.0 * inter / denom)
    return float(np.mean(scores)) if scores else 1.0


@dataclass
class InferenceBenchmark:
    wall_s: float
    peak_rss_bytes: int | None
    output_shapes: tuple


def benchmark_predict(
    predict_fn: Callable,
    inputs: Any,
    *,
    warmup: int = 1,
    repeats: int = 2,
) -> InferenceBenchmark:
    """Time a predict callable after optional warmup."""
    for _ in range(warmup):
        predict_fn(inputs)
    t0 = time.perf_counter()
    last = predict_fn(inputs)
    for _ in range(repeats - 1):
        last = predict_fn(inputs)
    elapsed = time.perf_counter() - t0
    shapes = tuple(
        tuple(o.shape) for o in (last if isinstance(last, tuple) else (last,))
    )
    return InferenceBenchmark(
        wall_s=elapsed / max(repeats, 1),
        peak_rss_bytes=_peak_rss_bytes(),
        output_shapes=shapes,
    )


def _eager_predict(net, inputs):
    """Default eager forward (Metal-safe)."""
    from neuroflux.synthseg.predict_synthseg import _model_predict

    return _model_predict(net, inputs)


def get_compiled_predictor(net, bucket_shape: tuple[int, int, int] | None):
    """
    Return a callable(inputs) -> tuple of numpy arrays.

    Uses XLA tf.function when policy allows; falls back to eager on failure.
    """
    policy = get_policy()
    if not policy.use_xla:
        return lambda inputs: _eager_predict(net, inputs)

    cache_key = (id(net), bucket_shape)
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    import tensorflow as tf

    if bucket_shape is not None:
        @tf.function(jit_compile=True)
        def _compiled(x):
            return net(x, training=False)

        def _run(inputs):
            try:
                outputs = _compiled(inputs)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                return tuple(
                    o.numpy() if hasattr(o, "numpy") else np.asarray(o) for o in outputs
                )
            except Exception:
                return _eager_predict(net, inputs)

        _PREDICTOR_CACHE[cache_key] = _run
        return _run

    def _run_dynamic(inputs):
        try:
            fn = tf.function(lambda x: net(x, training=False), jit_compile=True)
            outputs = fn(inputs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            return tuple(
                o.numpy() if hasattr(o, "numpy") else np.asarray(o) for o in outputs
            )
        except Exception:
            return _eager_predict(net, inputs)

    _PREDICTOR_CACHE[cache_key] = _run_dynamic
    return _run_dynamic
