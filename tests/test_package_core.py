"""Fast regression tests for the public package helpers.

These tests intentionally avoid UKB files, plotting backends, multiprocessing, and
long simulation runs.  They validate the deterministic transformations that connect
the package's simulation and real-data workflows.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from RR_OR_GEE_PGEE.config import (
    BASE_SCENARIO,
    full_scenarios,
    rep_seed,
    smoke_scenarios,
)
from RR_OR_GEE_PGEE.coverage import add_interval_columns, summarize_coverage
from RR_OR_GEE_PGEE.data_generation import model_matrix
from RR_OR_GEE_PGEE.methods import (
    METHOD_OR_GEE,
    METHOD_RR_GEE,
    zhang_yu_rr,
)
from UKB_validation.cli import FIGURE_COMMANDS, TABLE_COMMANDS
from UKB_validation.figure3.plot_ukb_empirical_maps import (
    reconstruct_maps,
    slice_indices_to_cut_coords,
)
from UKB_validation.paths import ExperimentPaths
from UKB_validation.ukb_python_experiment import (
    UKBDesign,
    _initial_beta,
    _objective_and_gradient,
    _poisson_outcome_chunk,
    load_voxel_ids,
    model_is_penalized,
    model_is_poisson,
    model_result_dir,
)


class SimulationConfigurationTests(unittest.TestCase):
    """Test scenario construction and reproducibility conventions."""

    def test_scenario_properties_and_seed_validation(self) -> None:
        self.assertEqual(BASE_SCENARIO.scenario_type, "standard")
        self.assertAlmostEqual(BASE_SCENARIO.true_rr, np.exp(1.6))
        self.assertEqual(rep_seed(7), 7)
        with self.assertRaises(ValueError):
            rep_seed(0)

    def test_full_and_smoke_scenario_grids(self) -> None:
        scenarios = full_scenarios()
        smoke = smoke_scenarios()
        self.assertEqual(len(scenarios), len({scenario.name for scenario in scenarios}))
        self.assertTrue(any(scenario.confounded for scenario in scenarios))
        self.assertTrue(all(scenario.n_replications == 100 for scenario in smoke))
        self.assertEqual(
            {scenario.name for scenario in scenarios},
            {scenario.name for scenario in smoke},
        )

    def test_design_matrix_accepts_historical_and_clear_time_column_names(self) -> None:
        data = pd.DataFrame(
            {"X1i": [0, 1], "obstime": [0.2, 0.4], "yij": [0, 1]}
        )
        np.testing.assert_allclose(
            model_matrix(data),
            [[1.0, 0.0, 0.2], [1.0, 1.0, 0.4]],
        )
        data["X2"] = [3.0, 4.0]
        np.testing.assert_allclose(
            model_matrix(data),
            [[1.0, 0.0, 3.0], [1.0, 1.0, 4.0]],
        )


class SimulationSummaryTests(unittest.TestCase):
    """Test RR/OR transformations and coverage summaries."""

    def test_zhang_yu_transformation_handles_invalid_inputs(self) -> None:
        expected = 1.0 / (0.25 + 0.75 * np.exp(-np.log(2.0)))
        self.assertAlmostEqual(zhang_yu_rr(np.log(2.0), 0.25), expected)
        self.assertTrue(np.isnan(zhang_yu_rr(np.nan, 0.25)))
        self.assertTrue(np.isnan(zhang_yu_rr(0.1, 1.1)))

    def test_interval_columns_use_log_rr_and_transformed_or_intervals(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "scenario": "toy",
                    "method": METHOD_RR_GEE,
                    "log_effect": np.log(2.0),
                    "se_log_effect": 0.1,
                    "p0_hat": np.nan,
                    "true_rr": 2.0,
                    "converged": True,
                    "finite": True,
                },
                {
                    "scenario": "toy",
                    "method": METHOD_OR_GEE,
                    "log_effect": np.log(2.0),
                    "se_log_effect": 0.1,
                    "p0_hat": 0.25,
                    "true_rr": 1.0,
                    "converged": True,
                    "finite": True,
                },
                {
                    "scenario": "toy",
                    "method": METHOD_RR_GEE,
                    "log_effect": 0.0,
                    "se_log_effect": 1.0,
                    "p0_hat": np.nan,
                    "true_rr": 1.0,
                    "converged": False,
                    "finite": True,
                },
            ]
        )
        result = add_interval_columns(rows)
        self.assertGreater(result.loc[0, "ci_upper"], result.loc[0, "ci_lower"])
        self.assertGreater(result.loc[1, "ci_upper"], result.loc[1, "ci_lower"])
        self.assertTrue(pd.isna(result.loc[2, "ci_lower"]))
        summary = summarize_coverage(result)
        self.assertEqual(summary["coverage_n"].tolist(), [1, 1])


class RealDataHelperTests(unittest.TestCase):
    """Test model classification, paths, voxel ordering, and optimizer helpers."""

    def test_model_classification_and_result_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(model_result_dir(root, "rr-pgee"), root / "rr_pgee")
            self.assertTrue(model_is_poisson("rr-gee"))
            self.assertFalse(model_is_poisson("or-gee"))
            self.assertTrue(model_is_penalized("or-pgee"))
            with self.assertRaises(ValueError):
                model_result_dir(root, "unknown")

            paths = ExperimentPaths.from_package_dir(root / "UKB_validation")
            self.assertEqual(
                paths.model_result_dir("or-pgee"),
                root / "UKB_validation" / "UKB" / "python_results" / "or_pgee",
            )

    def test_voxel_ids_are_positive_unique_and_one_based(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voxel_ids.dat"
            path.write_text("1\n4\n", encoding="ascii")
            np.testing.assert_array_equal(load_voxel_ids(path), [1, 4])
            path.write_text("1\n1\n", encoding="ascii")
            with self.assertRaises(ValueError):
                load_voxel_ids(path)

    def test_poisson_outcome_chunk_preserves_subject_visit_order(self) -> None:
        design = UKBDesign(
            X=np.zeros((4, 2)),
            X_clusters=np.zeros((2, 2, 2)),
            lesions1=np.array([[1, 0], [0, 1], [1, 1]]),
            lesions2=np.array([[0, 1], [1, 0], [1, 0]]),
            voxel_ids=np.array([1, 2, 3]),
        )
        chunk = _poisson_outcome_chunk(design, 1, 3)
        np.testing.assert_array_equal(
            chunk,
            [[0, 1], [1, 1], [1, 1], [0, 0]],
        )

    def test_initial_beta_and_poisson_objective_gradient(self) -> None:
        y = np.array([0.0, 1.0, 1.0])
        self.assertAlmostEqual(_initial_beta(y, poisson=True, p=2)[0], np.log(2.0 / 3.0))
        self.assertAlmostEqual(_initial_beta(y, poisson=False, p=2)[0], np.log(2.0))
        X = np.column_stack([np.ones(3), [0.0, 1.0, 1.0]])
        beta = np.array([0.0, 0.0])
        objective, gradient = _objective_and_gradient(beta, y, X, poisson=True, firth=False)
        self.assertAlmostEqual(objective, 3.0)
        np.testing.assert_allclose(gradient, [1.0, 0.0])

    def test_map_reconstruction_uses_fortran_voxel_order(self) -> None:
        visit_1 = np.array([[1, 0], [0, 1]], dtype=float)
        visit_2 = np.array([[0, 1], [1, 1]], dtype=float)
        sqrt_map, rr_map = reconstruct_maps(visit_1, visit_2, np.array([1, 2]), (2, 2, 1))
        self.assertAlmostEqual(sqrt_map[0, 0, 0], np.sqrt(0.5))
        self.assertAlmostEqual(sqrt_map[1, 0, 0], np.sqrt(0.5))
        self.assertAlmostEqual(rr_map[0, 0, 0], 1.0)
        self.assertAlmostEqual(rr_map[1, 0, 0], 2.0)

    def test_slice_coordinates_validate_image_bounds(self) -> None:
        image = nib.Nifti1Image(np.zeros((2, 2, 3)), np.diag([2.0, 2.0, 2.0, 1.0]))
        self.assertEqual(slice_indices_to_cut_coords((0, 2), image), [0.0, 4.0])
        with self.assertRaises(ValueError):
            slice_indices_to_cut_coords((3,), image)


class CommandInventoryTests(unittest.TestCase):
    """Ensure the reproducibility CLI targets the canonical package entry points."""

    def test_commands_use_root_level_real_data_modules(self) -> None:
        commands = FIGURE_COMMANDS + TABLE_COMMANDS
        self.assertTrue(commands)
        self.assertTrue(
            all(command.module.startswith("UKB_validation.") for command in commands)
        )
        nested_modules = {
            "figure3",
            "figure4",
            "figure5",
            "figure6",
            "figure7",
            "figureB3",
            "table5",
            "table6",
            "table7",
        }
        self.assertFalse(
            any(command.module.split(".")[1] in nested_modules for command in commands)
        )


if __name__ == "__main__":
    unittest.main()
