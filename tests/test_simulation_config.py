"""Unit tests for simulation configuration and deterministic scenario grids."""

from __future__ import annotations

import unittest

import numpy as np

from RR_OR_GEE_PGEE.config import (
    BASE_SCENARIO,
    DEFAULT_X2,
    Scenario,
    full_scenarios,
    smoke_scenarios,
)


class ScenarioTests(unittest.TestCase):
    """Check public scenario properties and defensive configuration behavior."""

    def test_scenario_properties_have_expected_scientific_meaning(self) -> None:
        scenario = Scenario(name="toy", beta=np.array([-2.0, 0.5, 0.1]))
        self.assertAlmostEqual(scenario.true_beta_b, 0.5)
        self.assertAlmostEqual(scenario.true_rr, np.exp(0.5))
        self.assertEqual(scenario.scenario_type, "standard")

    def test_confounded_scenario_properties_and_shift_format(self) -> None:
        scenario = full_scenarios(confounding_shift=0.75)[-1]
        self.assertTrue(scenario.confounded)
        self.assertEqual(scenario.scenario_type, "confounded")
        self.assertEqual(scenario.name, "confounded_shift_0_75")
        self.assertAlmostEqual(scenario.confounding_shift, 0.75)

    def test_scenario_defaults_do_not_share_mutable_x2_arrays(self) -> None:
        first = Scenario(name="first", beta=BASE_SCENARIO.beta)
        second = Scenario(name="second", beta=BASE_SCENARIO.beta)
        first.x2_values[0] = 99.0
        self.assertEqual(second.x2_values[0], DEFAULT_X2[0])
        self.assertEqual(BASE_SCENARIO.x2_values[0], DEFAULT_X2[0])

    def test_smoke_grid_preserves_names_and_reduces_replications(self) -> None:
        full = full_scenarios()
        smoke = smoke_scenarios()
        self.assertEqual([item.name for item in full], [item.name for item in smoke])
        self.assertTrue(all(item.n_replications == 100 for item in smoke))


if __name__ == "__main__":
    unittest.main()
