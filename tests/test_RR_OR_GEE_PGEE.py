"""Unit tests for the OR-PGEE comparison package."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from RR_OR_GEE_PGEE.config import BASE_SCENARIO, Scenario, full_scenarios, rep_seed
from RR_OR_GEE_PGEE.coverage import coverage_from_saved_payload
from RR_OR_GEE_PGEE.data_generation import model_matrix
from RR_OR_GEE_PGEE.methods import (
    METHOD_RR_GEE,
    FitRecord,
    estimate_p0_observed,
    zhang_yu_rr,
)


class ConfigurationTests(unittest.TestCase):
    """Validate simulation configuration and reproducibility helpers."""

    def test_scenario_values_are_not_mutated_by_grid_construction(self) -> None:
        original_beta = BASE_SCENARIO.beta.copy()
        scenarios = full_scenarios(confounding_shift=0.75)

        self.assertEqual(scenarios[-1].confounding_shift, 0.75)
        np.testing.assert_array_equal(BASE_SCENARIO.beta, original_beta)
        self.assertEqual(rep_seed(1), 1)
        with self.assertRaises(ValueError):
            rep_seed(-1)

    def test_scenario_rejects_no_implicit_duplicate_names(self) -> None:
        scenarios = full_scenarios()
        names = [scenario.name for scenario in scenarios]
        self.assertEqual(len(names), len(set(names)))
        self.assertIsInstance(scenarios[0], Scenario)


class DataAndMethodTests(unittest.TestCase):
    """Validate deterministic data and estimator helper behavior."""

    def test_model_matrix_prefers_explicit_x2_column(self) -> None:
        data = pd.DataFrame(
            {"X1i": [0, 1], "X2": [4.0, 5.0], "obstime": [9.0, 10.0]}
        )
        np.testing.assert_allclose(model_matrix(data), [[1.0, 0.0, 4.0], [1.0, 1.0, 5.0]])

    def test_observed_baseline_probability_uses_only_unexposed_rows(self) -> None:
        data = pd.DataFrame({"X1i": [0, 0, 1, 1], "yij": [0, 1, 1, 1]})
        self.assertAlmostEqual(estimate_p0_observed(data), 0.5)
        self.assertTrue(
            np.isnan(
                estimate_p0_observed(
                    pd.DataFrame({"X1i": [1], "yij": [1]})
                )
            )
        )

    def test_fit_record_preserves_failure_metadata(self) -> None:
        record = FitRecord(
            method=METHOD_RR_GEE,
            converged=False,
            finite=False,
            log_effect=np.nan,
            se_log_effect=np.nan,
            rr_estimate=np.nan,
            p0_hat=np.nan,
            iterations=np.nan,
            alpha=np.nan,
            phi=np.nan,
            bec_count=np.nan,
            failure_reason="test failure",
        )
        self.assertEqual(record.failure_reason, "test failure")
        self.assertFalse(record.converged)

    def test_saved_rr_payload_excludes_nonconverged_replications(self) -> None:
        payload = {
            "coefficients": np.array([[0.0, np.log(2.0)], [0.0, np.nan]]),
            "ses": np.array([[0.0, 0.1], [0.0, 0.1]]),
            "converged": np.array([True, False]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload, stream)

            result = coverage_from_saved_payload(
                path,
                np.log(2.0),
                method_key=METHOD_RR_GEE,
                coefficient_key="coefficients",
                se_key="ses",
                convergence_key="converged",
            )

        self.assertEqual(result["coverage_eligible"].tolist(), [True, False])
        self.assertEqual(result["coverage_eligible"].tolist(), [True, False])
        self.assertTrue(result.loc[0, "covered"])

    def test_zhang_yu_identity_at_null_effect(self) -> None:
        self.assertAlmostEqual(zhang_yu_rr(0.0, 0.2), 1.0)


if __name__ == "__main__":
    unittest.main()
