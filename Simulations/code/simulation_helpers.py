"""Shared utilities for simulation driver scripts."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationArgs:
    true_beta: np.ndarray
    nc: int
    cl_size: int
    prop: float
    rho: float
    n_sim: int
    p: int
    output_path: Path


def parse_simulation_args(argv: list[str]) -> SimulationArgs:
    if len(argv) != 10:
        raise SystemExit("Wrong number of arguments.")
    return SimulationArgs(
        true_beta=np.asarray([float(argv[0]), float(argv[1]), float(argv[2])]),
        nc=int(float(argv[3])),
        cl_size=int(float(argv[4])),
        prop=float(argv[5]),
        rho=float(argv[6]),
        n_sim=int(float(argv[7])),
        p=int(float(argv[8])),
        output_path=Path(argv[9]),
    )


def model_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(df)), df["X1i"].to_numpy(float), df["obstime"].to_numpy(float)]
    )


def save_pickle(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def run_model(
    label: str, simulation_index: int, fn: Callable[..., dict[str, Any] | Any], **kwargs: Any
) -> dict[str, Any] | Any | None:
    try:
        return fn(**kwargs)
    except Exception as exc:
        print(f"{label} failed for simulation {simulation_index}: {exc!r}", file=sys.stderr)
        return None


class ModelResults:
    """Container that preserves the historical output variable names."""

    def __init__(self, suffix: str, n_sim: int, p: int, *, include_model_se: bool) -> None:
        self.suffix = suffix
        self.coefs = np.zeros((n_sim, p))
        self.ses = np.zeros((n_sim, p))
        self.model_ses = np.zeros((n_sim, p)) if include_model_se else None
        self.model_ratios = np.zeros((n_sim, p))
        self.alpha = np.zeros(n_sim)
        self.phi = np.zeros(n_sim)
        self.eta = np.zeros(n_sim)
        self.iterations = np.zeros(n_sim)
        self.converged = np.zeros(n_sim, dtype=bool)

    def mark_failed(self, idx: int) -> None:
        self.coefs[idx, :] = np.nan
        self.ses[idx, :] = np.nan
        if self.model_ses is not None:
            self.model_ses[idx, :] = np.nan
        self.model_ratios[idx, :] = np.nan
        self.alpha[idx] = np.nan
        self.phi[idx] = np.nan
        self.eta[idx] = np.nan
        self.iterations[idx] = np.nan
        self.converged[idx] = False

    def record(self, idx: int, fit: dict[str, Any] | None, X: np.ndarray) -> None:
        if not isinstance(fit, dict):
            self.mark_failed(idx)
            return
        self.coefs[idx, :] = fit["beta"]
        self.ses[idx, :] = fit["beta_se_sandwich"]
        model_se = fit["beta_se_model"]
        if self.model_ses is not None:
            self.model_ses[idx, :] = model_se
        self.model_ratios[idx, :] = model_se / fit["beta_se_model_trace"][0]
        self.alpha[idx] = fit["alpha"]
        self.phi[idx] = fit["phi"]
        self.eta[idx] = np.sum(X @ fit["beta"] > 0)
        self.iterations[idx] = fit["iterations"]
        self.converged[idx] = fit["conv"]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            f"coefs_{self.suffix}": self.coefs,
            f"SEs_{self.suffix}": self.ses,
            f"SEs_model_ratio_{self.suffix}": self.model_ratios,
            f"alpha_{self.suffix}": self.alpha,
            f"phi_{self.suffix}": self.phi,
            f"eta_{self.suffix}": self.eta,
            f"iter_{self.suffix}": self.iterations,
            f"conv_{self.suffix}": self.converged,
        }
        if self.model_ses is not None:
            payload[f"SEs_model_{self.suffix}"] = self.model_ses
        return payload
