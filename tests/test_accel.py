"""Fast tests for adaptive acceleration policy (no TensorFlow graph required)."""

from neuroflux.accel import (
    PAD_BUCKETS,
    AcceleratorKind,
    bucket_pad_shape,
    dice_coefficient,
    label_histogram,
    reset_acceleration_state,
    resolve_policy,
)


class TestBucketPadding:
    def test_bucket_rounds_up_to_nearest_bucket(self):
        assert bucket_pad_shape([100, 100, 100]) == [160, 160, 160]
        assert bucket_pad_shape([180, 150, 150]) == [192, 160, 160]

    def test_bucket_respects_min_pad(self):
        assert bucket_pad_shape([64, 64, 64], min_pad=[192, 160, 160]) == [192, 160, 160]

    def test_buckets_divisible_by_32(self):
        for shape in bucket_pad_shape([200, 200, 200]):
            assert shape % 32 == 0
        assert set(PAD_BUCKETS) == {160, 192, 224, 256}


class TestAccelPolicy:
    def setup_method(self):
        reset_acceleration_state()

    def teardown_method(self):
        reset_acceleration_state()

    def test_no_accel_forces_eager_fp32(self):
        policy = resolve_policy(no_accel=True)
        assert policy.accel_mode == "eager"
        assert policy.compute_dtype == "float32"
        assert policy.use_xla is False
        assert policy.use_mixed_precision is False

    def test_cpu_default_is_eager_with_buckets(self):
        policy = resolve_policy(no_accel=True)
        assert policy.kind == AcceleratorKind.CPU
        assert policy.use_bucket_padding is True


class TestParityHelpers:
    def test_label_histogram_and_dice(self):
        a = label_histogram([0, 1, 1, 2])
        assert a == {0: 1, 1: 2, 2: 1}
        seg_a = [0, 1, 1, 2]
        seg_b = [0, 1, 2, 2]
        assert 0.0 < dice_coefficient(seg_a, seg_b) < 1.0
