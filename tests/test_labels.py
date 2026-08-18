"""
Unit tests for neuroflux.labels — pure NumPy, no SynthSeg required.
"""
import numpy as np

from neuroflux.labels import (
    HEMI_NAMES,
    TISSUE_NAMES,
    fs_to_hemi,
    fs_to_tissue,
)


class TestFsToTissue:
    def _arr(self, *labels):
        return np.array(labels, dtype=np.int32).reshape(len(labels), 1, 1)

    def test_background(self):
        assert fs_to_tissue(self._arr(0))[0, 0, 0] == 0

    def test_csf_lateral_ventricle(self):
        assert fs_to_tissue(self._arr(4))[0, 0, 0] == 1   # Left-Lateral-Ventricle

    def test_csf_third_ventricle(self):
        assert fs_to_tissue(self._arr(14))[0, 0, 0] == 1

    def test_csf_choroid_plexus_right(self):
        assert fs_to_tissue(self._arr(63))[0, 0, 0] == 1

    def test_cortical_gm_left(self):
        assert fs_to_tissue(self._arr(3))[0, 0, 0] == 2   # Left-Cerebral-Cortex

    def test_cortical_gm_right(self):
        assert fs_to_tissue(self._arr(42))[0, 0, 0] == 2  # Right-Cerebral-Cortex

    def test_wm_left(self):
        assert fs_to_tissue(self._arr(2))[0, 0, 0] == 3   # Left-Cerebral-WM

    def test_wm_hypointensities(self):
        assert fs_to_tissue(self._arr(77))[0, 0, 0] == 3

    def test_wm_corpus_callosum(self):
        for label in [251, 252, 253, 254, 255]:
            assert fs_to_tissue(self._arr(label))[0, 0, 0] == 3, f"CC label {label}"

    def test_deep_gm_thalamus_left(self):
        assert fs_to_tissue(self._arr(10))[0, 0, 0] == 4  # Left-Thalamus

    def test_deep_gm_hippocampus_right(self):
        assert fs_to_tissue(self._arr(53))[0, 0, 0] == 4  # Right-Hippocampus

    def test_brainstem(self):
        assert fs_to_tissue(self._arr(16))[0, 0, 0] == 5  # Brain-Stem

    def test_brainstem_alt_labels(self):
        for label in [170, 173, 174, 175, 178]:
            assert fs_to_tissue(self._arr(label))[0, 0, 0] == 5, f"BS label {label}"

    def test_cerebellum_left_cortex(self):
        assert fs_to_tissue(self._arr(8))[0, 0, 0] == 6   # Left-Cerebellum-Cortex

    def test_cerebellum_right_wm(self):
        assert fs_to_tissue(self._arr(46))[0, 0, 0] == 6  # Right-Cerebellum-WM

    def test_unknown_label_maps_to_background(self):
        assert fs_to_tissue(self._arr(999))[0, 0, 0] == 0

    def test_return_dtype_is_uint8(self):
        result = fs_to_tissue(np.zeros((5, 5, 5), dtype=np.int32))
        assert result.dtype == np.uint8

    def test_vectorised_batch(self):
        """All six tissue classes appear correctly in a mixed volume."""
        labels = np.array([0, 4, 3, 2, 10, 16, 8], dtype=np.int32)
        vol = labels.reshape(7, 1, 1)
        tissue = fs_to_tissue(vol).flatten()
        expected = [0, 1, 2, 3, 4, 5, 6]
        assert list(tissue) == expected

    def test_out_of_range_label_clipped_to_background(self):
        # Labels >= 512 are clipped; no valid mapping → background
        arr = np.array([600], dtype=np.int32).reshape(1, 1, 1)
        assert fs_to_tissue(arr)[0, 0, 0] == 0


class TestFsToHemi:
    def _make(self, fs_labels):
        fs = np.array(fs_labels, dtype=np.int32).reshape(len(fs_labels), 1, 1)
        tissue = fs_to_tissue(fs)
        return fs, tissue, fs_to_hemi(fs, tissue)

    def test_background_stays_zero(self):
        _, _, hemi = self._make([0])
        assert hemi[0, 0, 0] == 0

    def test_csf_bilateral(self):
        _, _, hemi = self._make([4])   # CSF
        assert hemi[0, 0, 0] == 1

    def test_gm_left(self):
        _, _, hemi = self._make([3])
        assert hemi[0, 0, 0] == 2

    def test_gm_right(self):
        _, _, hemi = self._make([42])
        assert hemi[0, 0, 0] == 3

    def test_wm_left(self):
        _, _, hemi = self._make([2])
        assert hemi[0, 0, 0] == 4

    def test_wm_right(self):
        _, _, hemi = self._make([41])
        assert hemi[0, 0, 0] == 5

    def test_wm_hypointensities_split_by_ras_x(self):
        fs = np.array([77, 77], dtype=np.int32).reshape(2, 1, 1)
        tissue = fs_to_tissue(fs)
        affine = np.eye(4)
        affine[0, 3] = -0.5  # voxel 0 at x=-0.5, voxel 1 at x=+0.5
        hemi = fs_to_hemi(fs, tissue, affine=affine)
        assert hemi[0, 0, 0] == 4
        assert hemi[1, 0, 0] == 5

    def test_wm_hypointensities_midplane_without_affine(self):
        fs = np.array([77, 77], dtype=np.int32).reshape(2, 1, 1)
        tissue = fs_to_tissue(fs)
        hemi = fs_to_hemi(fs, tissue)
        assert hemi[0, 0, 0] == 4
        assert hemi[1, 0, 0] == 5

    def test_corpus_callosum(self):
        _, _, hemi = self._make([253])
        assert hemi[0, 0, 0] == 6

    def test_deep_gm_left(self):
        _, _, hemi = self._make([10])  # Left-Thalamus
        assert hemi[0, 0, 0] == 7

    def test_deep_gm_right(self):
        _, _, hemi = self._make([49])  # Right-Thalamus
        assert hemi[0, 0, 0] == 8

    def test_brainstem(self):
        _, _, hemi = self._make([16])
        assert hemi[0, 0, 0] == 9

    def test_cerebellum(self):
        _, _, hemi = self._make([8])   # Left-Cerebellum-Cortex
        assert hemi[0, 0, 0] == 10

    def test_output_dtype_is_uint8(self):
        fs = np.zeros((4, 4, 4), dtype=np.int32)
        tissue = fs_to_tissue(fs)
        result = fs_to_hemi(fs, tissue)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        fs = np.zeros((3, 5, 7), dtype=np.int32)
        tissue = fs_to_tissue(fs)
        hemi = fs_to_hemi(fs, tissue)
        assert hemi.shape == (3, 5, 7)


class TestNameDicts:
    def test_tissue_names_has_six_classes(self):
        assert set(TISSUE_NAMES.keys()) == {1, 2, 3, 4, 5, 6}

    def test_hemi_names_has_ten_classes(self):
        assert set(HEMI_NAMES.keys()) == set(range(1, 11))

    def test_tissue_names_values_are_strings(self):
        assert all(isinstance(v, str) for v in TISSUE_NAMES.values())

    def test_hemi_names_values_are_strings(self):
        assert all(isinstance(v, str) for v in HEMI_NAMES.values())
