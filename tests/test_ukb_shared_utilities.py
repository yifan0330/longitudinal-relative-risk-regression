"""Unit tests for shared UKB input, map, and statistical utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from UKB_validation.io import (
    load_aligned_map,
    load_empirical_visits,
    load_voxel_ids,
    values_at_voxels,
)
from UKB_validation.mapping import values_to_map
from UKB_validation.stats import benjamini_hochberg, odds_ratio_to_relative_risk


class UkbIoTests(unittest.TestCase):
    """Validate input shape, alignment, and voxel-ID contracts."""

    def test_voxel_loading_validates_and_truncates_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("1\n3\n5\n", encoding="ascii")
            np.testing.assert_array_equal(load_voxel_ids(path, max_voxels=2), [1, 3])
            path.write_text("0\n2\n", encoding="ascii")
            with self.assertRaises(ValueError):
                load_voxel_ids(path)

    def test_map_loading_checks_file_alignment_and_dimensions(self) -> None:
        affine = np.eye(4)
        anatomical = nib.Nifti1Image(np.zeros((2, 2, 1)), affine)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 1)), affine), path)
            np.testing.assert_allclose(load_aligned_map(path, anatomical), 1.0)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 1)), affine * 2), path)
            with self.assertRaises(ValueError):
                load_aligned_map(path, anatomical)

    def test_empirical_visit_loader_rejects_mismatched_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesions_atleast6_CVR.npz"
            np.savez(path, lesions_vis1=np.zeros((2, 3)), lesions_vis2=np.zeros((1, 3)))
            with self.assertRaises(ValueError):
                load_empirical_visits(Path(directory), 1)

    def test_voxel_extraction_rejects_non3d_and_out_of_range_inputs(self) -> None:
        with self.assertRaises(ValueError):
            values_at_voxels(np.zeros((2, 2)), np.array([1]))
        with self.assertRaises(ValueError):
            values_at_voxels(np.zeros((2, 2, 1)), np.array([5]))


class UkbMappingAndStatisticsTests(unittest.TestCase):
    """Validate map round trips and edge cases in statistical transformations."""

    def test_map_builder_rejects_length_and_range_errors(self) -> None:
        with self.assertRaises(ValueError):
            values_to_map(np.array([1.0]), np.array([1, 2]), (2, 2, 1))
        with self.assertRaises(ValueError):
            values_to_map(np.array([1.0]), np.array([0]), (2, 2, 1))
        result = values_to_map(np.array([]), np.array([], dtype=int), (2, 2, 1))
        self.assertTrue(np.isnan(result).all())

    def test_bh_adjustment_handles_failed_tests_and_empty_arrays(self) -> None:
        np.testing.assert_allclose(
            benjamini_hochberg(np.array([0.01, np.nan, 0.04])),
            [0.03, 1.0, 0.06],
        )
        self.assertEqual(benjamini_hochberg(np.array([])).size, 0)

    def test_odds_ratio_conversion_is_elementwise(self) -> None:
        result = odds_ratio_to_relative_risk(
            np.array([1.0, 2.0]), np.array([0.0, 0.25])
        )
        np.testing.assert_allclose(result, [1.0, 1.6])


if __name__ == "__main__":
    unittest.main()
