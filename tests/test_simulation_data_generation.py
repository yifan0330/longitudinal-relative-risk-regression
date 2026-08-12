"""Unit tests for synthetic simulation data construction."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from RR_OR_GEE_PGEE.config import Scenario
from RR_OR_GEE_PGEE.data_generation import (
    generate_standard_datasets_broadcast,
    model_matrix,
)


def _fake_historical_generators():
    """Provide deterministic correlated-Bernoulli dependencies for fast tests."""

    def xch(size: int, _rho: float) -> np.ndarray:
        return np.eye(size)

    def simulate_correlated_bernoulli(probabilities, _corr, rng):
        return (rng.random(len(probabilities)) < probabilities).astype(int)

    return None, simulate_correlated_bernoulli, xch


class BroadcastGenerationTests(unittest.TestCase):
    """Validate the dependency-light broadcast data-generation path."""

    def setUp(self) -> None:
        self.scenario = Scenario(
            name="toy",
            beta=np.array([-2.0, 0.4, 0.1]),
            n_subjects=3,
            n_visits=2,
            x2_values=np.array([0.2, 0.4]),
        )

    def test_broadcast_output_has_subject_visit_rows_and_binary_outcomes(self) -> None:
        with patch(
            "RR_OR_GEE_PGEE.data_generation._historical_generators",
            return_value=_fake_historical_generators(),
        ):
            result = generate_standard_datasets_broadcast(
                self.scenario, [1, 2], seed=123
            )
        self.assertEqual(result.data.shape[0], 2 * 3 * 2)
        self.assertEqual(result.data["id"].nunique(), 3)
        self.assertEqual(set(result.data["yij"].unique()), {0, 1})
        self.assertEqual(result.diagnostics["replication"].tolist(), [1, 2])
        self.assertEqual(result.diagnostics["n_observations"].tolist(), [6, 6])

    def test_broadcast_generation_is_reproducible_for_same_seed(self) -> None:
        with patch(
            "RR_OR_GEE_PGEE.data_generation._historical_generators",
            return_value=_fake_historical_generators(),
        ):
            first = generate_standard_datasets_broadcast(self.scenario, [3], seed=9)
            second = generate_standard_datasets_broadcast(self.scenario, [3], seed=9)
        np.testing.assert_array_equal(first.data["yij"], second.data["yij"])
        np.testing.assert_allclose(
            first.diagnostics["event_rate"], second.diagnostics["event_rate"]
        )

    def test_broadcast_rejects_confounded_and_empty_inputs(self) -> None:
        confounded = Scenario(
            name="confounded", beta=self.scenario.beta, confounded=True
        )
        with self.assertRaises(ValueError):
            generate_standard_datasets_broadcast(confounded, [1])
        with self.assertRaises(ValueError):
            generate_standard_datasets_broadcast(self.scenario, [])

    def test_model_matrix_reports_missing_required_columns(self) -> None:
        with self.assertRaises(KeyError):
            model_matrix(self.scenario_data_without_time())

    def scenario_data_without_time(self):
        import pandas as pd

        return pd.DataFrame({"X1i": [0, 1], "yij": [0, 1]})


if __name__ == "__main__":
    unittest.main()
