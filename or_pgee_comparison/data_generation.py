"""Dataset generation for the OR-PGEE comparison simulations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import REPO_ROOT, Scenario, rep_seed

SIM_CODE_DIR = REPO_ROOT / "Simulations" / "code"


def _historical_generators():
    """Load the historical simulation helpers only when data generation is requested."""
    import sys

    if str(SIM_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_CODE_DIR))
    from source_simdata import gen_dataPP, simulate_correlated_bernoulli, xch

    return gen_dataPP, simulate_correlated_bernoulli, xch


@dataclass(frozen=True)
class DatasetBundle:
    """Generated data and per-replication diagnostics."""

    data: pd.DataFrame
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class BatchedDatasetFrame:
    """Broadcast-generated standard-scenario data for many replications."""

    data: pd.DataFrame
    diagnostics: pd.DataFrame


def model_matrix(data: pd.DataFrame) -> np.ndarray:
    """Build the log-link design matrix used by RR-GEE/RR-PGEE."""
    x2_column = "X2" if "X2" in data.columns else "obstime"
    return np.column_stack(
        [
            np.ones(len(data)),
            data["X1i"].to_numpy(float),
            data[x2_column].to_numpy(float),
        ]
    )


def generate_dataset(scenario: Scenario, replication: int) -> DatasetBundle:
    """Generate one paired dataset using the published one-based replication seed."""
    rng = np.random.default_rng(rep_seed(replication))
    if scenario.confounded:
        return _generate_confounded_dataset(scenario, replication, rng)

    gen_dataPP, _simulate_correlated_bernoulli, _xch = _historical_generators()
    data = gen_dataPP(
        beta=scenario.beta,
        nc=scenario.n_subjects,
        cl_size=scenario.n_visits,
        p=scenario.prop,
        rho=scenario.rho,
        rng=rng,
    )
    data = data.assign(X2=data["obstime"].to_numpy(float))
    diagnostics = _diagnostics(scenario, replication, data, probability_clip_rate=0.0)
    return DatasetBundle(data=data, diagnostics=diagnostics)


def generate_standard_datasets_broadcast(
    scenario: Scenario,
    replications: Sequence[int],
    *,
    seed: int = 20260804,
) -> BatchedDatasetFrame:
    """Generate standard-scenario datasets for many replications in one array pass.

    This is intended for speed benchmarking. It does not reproduce the historical
    one-RNG-per-replication streams used by :func:`generate_dataset`.
    """
    if scenario.confounded:
        raise ValueError("Broadcast generation is only implemented for standard scenarios.")
    replication_arr = np.asarray(tuple(replications), dtype=int)
    if replication_arr.ndim != 1 or replication_arr.size == 0:
        raise ValueError("At least one replication number is required.")

    rng = np.random.default_rng(seed)
    n_rep = int(replication_arr.size)
    n_subjects = int(scenario.n_subjects)
    n_visits = int(scenario.n_visits)
    x2_values = np.asarray(scenario.x2_values, dtype=float)
    if x2_values.shape != (n_visits,):
        x2_values = 0.2 * np.arange(1, n_visits + 1, dtype=float)

    x1_cluster = rng.binomial(1, float(scenario.prop), size=(n_rep, n_subjects))
    x1 = np.repeat(x1_cluster[:, :, None], n_visits, axis=2)
    x2 = np.broadcast_to(x2_values, (n_rep, n_subjects, n_visits))
    design = np.stack([np.ones_like(x1, dtype=float), x1.astype(float), x2], axis=-1)
    raw_probs = np.exp(np.einsum("rsva,a->rsv", design, np.asarray(scenario.beta, dtype=float)))
    probs = np.clip(raw_probs, 1e-12, 1 - 1e-12)

    _gen_dataPP, _simulate_correlated_bernoulli, xch = _historical_generators()
    corr = xch(n_visits, scenario.rho)
    z = rng.multivariate_normal(np.zeros(n_visits), corr, size=(n_rep, n_subjects))
    y = (z <= norm.ppf(probs)).astype(int)

    flat_size = n_subjects * n_visits
    ids_one = np.repeat(np.arange(1, n_subjects + 1), n_visits)
    x2_one = np.tile(x2_values, n_subjects)
    data = pd.DataFrame(
        {
            "replication": np.repeat(replication_arr, flat_size),
            "id": np.tile(ids_one, n_rep),
            "yij": y.reshape(-1),
            "intercept": 1.0,
            "X1i": x1.reshape(-1),
            "obstime": np.tile(x2_one, n_rep),
            "X2": x2.reshape(-1),
        }
    )

    x1_flat = x1.reshape(n_rep, flat_size).astype(float)
    x2_flat = x2.reshape(n_rep, flat_size)
    x1_centered = x1_flat - x1_flat.mean(axis=1, keepdims=True)
    x2_centered = x2_flat - x2_flat.mean(axis=1, keepdims=True)
    denom = np.sqrt((x1_centered**2).mean(axis=1) * (x2_centered**2).mean(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_x1_x2 = (x1_centered * x2_centered).mean(axis=1) / denom
    corr_x1_x2[denom == 0.0] = np.nan

    diagnostics = pd.DataFrame(
        {
            "scenario": scenario.name,
            "scenario_type": scenario.scenario_type,
            "replication": replication_arr,
            "n_observations": flat_size,
            "x1_x2_corr": corr_x1_x2,
            "x2_clip_rate": 0.0,
            "probability_clip_rate": 0.0,
            "event_rate": y.reshape(n_rep, flat_size).mean(axis=1),
        }
    )
    return BatchedDatasetFrame(data=data, diagnostics=diagnostics)


def _generate_confounded_dataset(
    scenario: Scenario, replication: int, rng: np.random.Generator
) -> DatasetBundle:
    sizes = np.repeat(int(scenario.n_visits), int(scenario.n_subjects))
    ids = np.repeat(np.arange(1, int(scenario.n_subjects) + 1), sizes)
    x1_cluster = rng.binomial(1, float(scenario.prop), int(scenario.n_subjects))
    x1 = np.repeat(x1_cluster, sizes)
    base_x2 = np.tile(np.asarray(scenario.x2_values, dtype=float), int(scenario.n_subjects))
    shifted_x2 = base_x2 + float(scenario.confounding_shift) * (x1 - float(scenario.prop))
    x2 = np.maximum(shifted_x2, float(scenario.x2_floor))
    x2_clip_rate = float(np.mean(shifted_x2 < scenario.x2_floor))

    design = np.column_stack([np.ones(len(x1)), x1, x2])
    raw_probs = np.exp(design @ scenario.beta)
    probability_clip_rate = float(np.mean((raw_probs < 1e-12) | (raw_probs > 1 - 1e-12)))
    probs = np.clip(raw_probs, 1e-12, 1 - 1e-12)

    _gen_dataPP, simulate_correlated_bernoulli, xch = _historical_generators()
    y_parts: list[np.ndarray] = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        y_parts.append(simulate_correlated_bernoulli(probs[start:stop], xch(int(size), scenario.rho), rng))
        start = stop
    data = pd.DataFrame(
        {
            "id": ids,
            "yij": np.concatenate(y_parts),
            "intercept": 1.0,
            "X1i": x1,
            "obstime": base_x2,
            "X2": x2,
        }
    )
    diagnostics = _diagnostics(
        scenario,
        replication,
        data,
        probability_clip_rate=probability_clip_rate,
        x2_clip_rate=x2_clip_rate,
    )
    return DatasetBundle(data=data, diagnostics=diagnostics)


def _diagnostics(
    scenario: Scenario,
    replication: int,
    data: pd.DataFrame,
    *,
    probability_clip_rate: float,
    x2_clip_rate: float = 0.0,
) -> dict[str, float | int | str]:
    x1 = data["X1i"].to_numpy(float)
    x2 = data["X2"].to_numpy(float)
    corr = np.corrcoef(x1, x2)[0, 1] if np.std(x1) > 0 and np.std(x2) > 0 else np.nan
    return {
        "scenario": scenario.name,
        "scenario_type": scenario.scenario_type,
        "replication": replication,
        "n_observations": int(len(data)),
        "x1_x2_corr": float(corr),
        "x2_clip_rate": float(x2_clip_rate),
        "probability_clip_rate": float(probability_clip_rate),
        "event_rate": float(data["yij"].mean()),
    }
