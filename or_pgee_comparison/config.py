"""Configuration for the OR-PGEE comparison simulations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_BETA = np.asarray([-4.0, 1.6, 0.2])
DEFAULT_X2 = np.asarray([0.2, 0.4, 0.6, 0.8])
DEFAULT_N_SUBJECTS = 50
DEFAULT_PROP = 0.2
DEFAULT_RHO = 0.4
DEFAULT_R = 1_000
SMOKE_R = 100
TOL = 1e-4
MAX_ITER = 25
STRICT_ORIGINAL_IRLS = True
TRUE_PHI = 1.0


@dataclass(frozen=True)
class Scenario:
    """A single simulation scenario."""

    name: str
    beta: np.ndarray
    n_subjects: int = DEFAULT_N_SUBJECTS
    n_visits: int = 4
    prop: float = DEFAULT_PROP
    rho: float = DEFAULT_RHO
    n_replications: int = DEFAULT_R
    x2_values: np.ndarray = field(default_factory=lambda: DEFAULT_X2.copy())
    confounded: bool = False
    confounding_shift: float = 0.0
    x2_floor: float = 0.01
    probability_clip_threshold: float = 0.01

    @property
    def true_beta_b(self) -> float:
        return float(self.beta[1])

    @property
    def true_rr(self) -> float:
        return float(np.exp(self.true_beta_b))

    @property
    def scenario_type(self) -> str:
        return "confounded" if self.confounded else "standard"


BASE_SCENARIO = Scenario(name="base", beta=DEFAULT_BETA)

def full_scenarios(*, confounding_shift: float = 0.50) -> tuple[Scenario, ...]:
    """Return the full Table 2/3/4-style scenario grid plus one confounded case."""
    scenarios: list[Scenario] = [BASE_SCENARIO]
    scenarios.extend(
        replace(BASE_SCENARIO, name=f"beta0_{value:g}", beta=np.asarray([value, 1.6, 0.2]))
        for value in (-3.0, -2.0)
    )
    scenarios.extend(
        replace(BASE_SCENARIO, name=f"beta_b_{value:g}", beta=np.asarray([-4.0, value, 0.2]))
        for value in (1.2, 1.4, 1.8, 2.0)
    )
    scenarios.extend(
        replace(BASE_SCENARIO, name=f"prop_{value:g}", prop=value)
        for value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    )
    scenarios.extend(
        replace(BASE_SCENARIO, name=f"alpha_{value:g}", rho=value)
        for value in (0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
    )
    scenarios.extend(
        replace(BASE_SCENARIO, name=f"n_{value:g}", n_subjects=value)
        for value in (25, 75, 100)
    )
    scenarios.append(
        replace(
            BASE_SCENARIO,
            name=f"confounded_shift_{_format_shift(confounding_shift)}",
            confounded=True,
            confounding_shift=confounding_shift,
        )
    )
    deduped = {scenario.name: scenario for scenario in scenarios}
    return tuple(deduped.values())


def smoke_scenarios(*, confounding_shift: float = 0.50) -> tuple[Scenario, ...]:
    """Return the full scenario grid with fewer replications for smoke testing."""
    return tuple(
        replace(scenario, n_replications=SMOKE_R)
        for scenario in full_scenarios(confounding_shift=confounding_shift)
    )


def _format_shift(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


SMOKE_SCENARIOS = smoke_scenarios()


def rep_seed(replication: int) -> int:
    """Return the published one-based replication seed."""
    if replication < 1:
        raise ValueError("Replication numbers are one-based to match published simulations.")
    return replication
