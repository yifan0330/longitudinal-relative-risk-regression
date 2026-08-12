"""Unit tests for real-data experiment utilities."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from UKB_validation.cli import (
    FIGURE_COMMANDS,
    TABLE_COMMANDS,
    _commands_with_cache_policy,
    _fit_arguments,
    build_parser,
)
from UKB_validation.paths import ExperimentPaths
from UKB_validation.ukb_python_experiment import (
    _objective_and_gradient,
    default_n_jobs,
    model_is_penalized,
    model_is_poisson,
)


class PathAndModelTests(unittest.TestCase):
    """Validate model labels and historical result paths."""

    def test_model_labels_cover_all_supported_link_and_penalty_combinations(self) -> None:
        models = ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
        self.assertEqual(
            [(model_is_poisson(model), model_is_penalized(model)) for model in models],
            [(True, False), (True, True), (False, False), (False, True)],
        )

    def test_model_result_paths_normalize_hyphens(self) -> None:
        paths = ExperimentPaths.from_package_dir(Path("/tmp/project/UKB_validation"))
        self.assertEqual(
            paths.model_result_dir("rr-pgee"),
            Path("/tmp/project/UKB_validation/UKB/python_results/rr_pgee"),
        )

    def test_default_worker_count_is_positive_and_bounded(self) -> None:
        self.assertGreaterEqual(default_n_jobs(), 1)
        self.assertLessEqual(default_n_jobs(), 8)


class CliTests(unittest.TestCase):
    """Validate reproducibility command parsing and cache policy."""

    def test_parser_accepts_all_workflow_stages(self) -> None:
        for stage in ("fit", "figures", "tables", "all"):
            self.assertEqual(build_parser().parse_args([stage]).command, stage)

    def test_fit_arguments_validate_worker_count(self) -> None:
        args = build_parser().parse_args(["fit", "--n-jobs", "2"])
        self.assertEqual(_fit_arguments(args), ("--models", "all", "--use-cache", "--n-jobs", "2"))
        invalid = build_parser().parse_args(["fit", "--n-jobs", "0"])
        with self.assertRaises(ValueError):
            _fit_arguments(invalid)

    def test_rerun_policy_removes_only_cache_arguments(self) -> None:
        commands = _commands_with_cache_policy(FIGURE_COMMANDS + TABLE_COMMANDS, rerun_models=True)
        self.assertTrue(commands)
        self.assertTrue(
            all("--use-cache" not in command.arguments for command in commands)
        )
        self.assertTrue(
            any("--models" in command.arguments for command in commands)
        )


class OptimizerHelperTests(unittest.TestCase):
    """Validate the logistic objective and analytical gradient."""

    def test_logistic_gradient_matches_finite_difference(self) -> None:
        beta = np.array([0.2, -0.4])
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y = np.array([0.0, 1.0, 1.0])
        analytical = _objective_and_gradient(beta, y, X, False, False)[1]
        epsilon = 1e-6
        numerical = np.array(
            [
                (
                    _objective_and_gradient(beta + epsilon * basis, y, X, False, False)[0]
                    - _objective_and_gradient(beta - epsilon * basis, y, X, False, False)[0]
                )
                / (2.0 * epsilon)
                for basis in np.eye(2)
            ]
        )
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
