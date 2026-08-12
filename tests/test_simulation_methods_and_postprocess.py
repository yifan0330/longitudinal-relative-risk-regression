"""Unit tests for estimator contracts and simulation summary tables."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from RR_OR_GEE_PGEE.config import BASE_SCENARIO
from RR_OR_GEE_PGEE.methods import (
    METHOD_RR_GEE,
    METHOD_ORDER,
    fit_all_methods,
    fit_rr_method,
    zhang_yu_rr,
)
from RR_OR_GEE_PGEE.postprocess import _table2_convergence, _table3_rr_bias_mse


class MethodContractTests(unittest.TestCase):
    """Ensure invalid and failed fits remain explicit to callers."""

    def test_unknown_fit_engine_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_all_methods(pd.DataFrame(), BASE_SCENARIO, fit_engine="unknown")

    def test_failing_rr_fitter_returns_failure_record(self) -> None:
        data = pd.DataFrame(
            {
                "X1i": [0, 1],
                "X2": [0.2, 0.4],
                "yij": [0, 1],
            }
        )

        def failing_fitter(**_kwargs):
            raise RuntimeError("synthetic failure")

        record = fit_rr_method(METHOD_RR_GEE, data, BASE_SCENARIO, failing_fitter)
        self.assertFalse(record.converged)
        self.assertFalse(record.finite)
        self.assertIn("synthetic failure", record.failure_reason)

    def test_zhang_yu_handles_boundary_baseline_risks(self) -> None:
        self.assertAlmostEqual(zhang_yu_rr(0.0, 0.0), 1.0)
        self.assertAlmostEqual(zhang_yu_rr(0.0, 1.0), 1.0)
        self.assertTrue(np.isnan(zhang_yu_rr(0.0, np.nan)))


class SummaryTableTests(unittest.TestCase):
    """Check finite/convergence flags and common summary calculations."""

    def setUp(self) -> None:
        rows = []
        for method in METHOD_ORDER:
            for replication in (1, 2):
                rows.append(
                    {
                    "scenario": "toy",
                    "scenario_type": "standard",
                    "method": method,
                    "replication": replication,
                    "bec_count": 2.0 if replication == 1 else 11.0,
                    "converged": replication == 1,
                    "finite": True,
                    "coverage_eligible": replication == 1,
                    "rr_estimate": 2.0 if replication == 1 else 1.0,
                    "true_rr": 1.5,
                    "true_beta_b": np.log(1.5),
                    }
                )
        self.rows = pd.DataFrame(rows)

    def test_convergence_table_partitions_bec_and_success(self) -> None:
        table = _table2_convergence(self.rows)
        row = table.iloc[0]
        self.assertEqual(row["n_replications"], 2)
        self.assertEqual(row["converged_bec_le_10"], 1)
        self.assertEqual(row["non_converged_bec_gt_10"], 1)

    def test_rr_summary_uses_all_rows_for_unconditional_metrics(self) -> None:
        table = _table3_rr_bias_mse(self.rows)
        row = table[table["method"] == METHOD_RR_GEE].iloc[0]
        self.assertEqual(row["n_used"], 1)
        self.assertAlmostEqual(row["mean_rr"], 2.0)
        self.assertEqual(row["n_unconditional"], 1)


if __name__ == "__main__":
    unittest.main()
