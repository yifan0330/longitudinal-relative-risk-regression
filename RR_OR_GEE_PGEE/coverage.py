"""Confidence intervals and empirical coverage summaries."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .methods import METHOD_OR_GEE, METHOD_OR_PGEE, OR_METHODS, zhang_yu_rr

Z_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def add_interval_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Add RR-scale Wald interval and coverage columns to replication rows."""
    rows = rows.copy()
    rows["ci_lower"] = np.nan
    rows["ci_upper"] = np.nan
    usable = rows["converged"] & rows["finite"]

    rr_mask = usable & ~rows["method"].isin(OR_METHODS)
    rr_lower_log = rows.loc[rr_mask, "log_effect"] - Z_975 * rows.loc[rr_mask, "se_log_effect"]
    rr_upper_log = rows.loc[rr_mask, "log_effect"] + Z_975 * rows.loc[rr_mask, "se_log_effect"]
    with np.errstate(over="ignore", invalid="ignore"):
        rows.loc[rr_mask, "ci_lower"] = np.exp(rr_lower_log)
        rows.loc[rr_mask, "ci_upper"] = np.exp(rr_upper_log)

    or_mask = usable & rows["method"].isin(OR_METHODS)
    for idx, row in rows.loc[or_mask].iterrows():
        lower = zhang_yu_rr(row["log_effect"] - Z_975 * row["se_log_effect"], row["p0_hat"])
        upper = zhang_yu_rr(row["log_effect"] + Z_975 * row["se_log_effect"], row["p0_hat"])
        rows.loc[idx, "ci_lower"] = min(lower, upper)
        rows.loc[idx, "ci_upper"] = max(lower, upper)

    rows["covered"] = (
        usable
        & (rows["ci_lower"] <= rows["true_rr"])
        & (rows["ci_upper"] >= rows["true_rr"])
    )
    rows["coverage_eligible"] = usable
    return rows


def summarize_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize empirical coverage by scenario and method."""
    eligible = rows[rows["coverage_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=["scenario", "method", "coverage", "coverage_n", "true_rr"]
        )
    return (
        eligible.groupby(["scenario", "method"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            coverage_n=("covered", "size"),
            true_rr=("true_rr", "first"),
        )
        .sort_values(["scenario", "method"])
    )


def coverage_from_saved_payload(
    payload_path: str | Path,
    true_beta_b: float,
    *,
    method_key: str,
    coefficient_key: str,
    se_key: str,
    convergence_key: str,
) -> pd.DataFrame:
    """Compute RR-GEE/RR-PGEE coverage post hoc from saved per-replication arrays."""
    with Path(payload_path).open("rb") as fh:
        payload: dict[str, Any] = pickle.load(fh)
    coefficients = np.asarray(payload[coefficient_key], dtype=float)
    ses = np.asarray(payload[se_key], dtype=float)
    converged = np.asarray(payload[convergence_key], dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        rr_estimate = np.exp(coefficients[:, 1])
    rows = pd.DataFrame(
        {
            "scenario": Path(payload_path).stem,
            "method": method_key,
            "replication": np.arange(1, len(coefficients) + 1),
            "log_effect": coefficients[:, 1],
            "se_log_effect": ses[:, 1],
            "rr_estimate": rr_estimate,
            "p0_hat": np.nan,
            "converged": converged,
            "finite": np.isfinite(coefficients[:, 1])
            & np.isfinite(ses[:, 1])
            & np.isfinite(rr_estimate),
            "true_beta_b": true_beta_b,
            "true_rr": float(np.exp(true_beta_b)),
        }
    )
    return add_interval_columns(rows)
