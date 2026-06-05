#!/usr/bin/env python3
"""Python translation of Sept21_pgee_logPoisson_dispersion_fn.R.

Contains the custom penalized log-Poisson GEE used by the CVR voxel scripts,
plus small IO helpers shared by the converted entry points.  No R code is
embedded or executed.
"""

from __future__ import annotations

import gzip
import importlib.util
import math
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg

TEMPDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp")
GEEDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/CVRanalysis")
IMAGEDIR_VIS1 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis")
IMAGEDIR_VIS2 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis")

_COMPLETE_DF: pd.DataFrame | None = None
_IDS: np.ndarray | None = None
_LESIONS1: np.ndarray | None = None
_LESIONS2: np.ndarray | None = None
_SUBSET_IDX: np.ndarray | None = None
_MODEL_KIND = "pgee"
_INCLUDE_INTERACTIONS = True


def load_rdata(path: str | Path) -> dict[str, Any]:
    """Load an RData-like file.

    Converted Python scripts write pickle payloads to the original .RData
    filenames.  For original R inputs, install pyreadr (preferred) or rdata.
    """
    path = Path(path)
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        pass
    try:
        import pyreadr  # type: ignore

        return dict(pyreadr.read_r(str(path)))
    except Exception as exc_pyreadr:
        try:
            import rdata  # type: ignore

            return rdata.conversion.convert(rdata.parser.parse_file(str(path)))
        except Exception as exc_rdata:
            raise RuntimeError(
                f"Cannot read {path}. Install pyreadr or rdata for original RData inputs. "
                f"pyreadr error: {exc_pyreadr}; rdata error: {exc_rdata}"
            ) from exc_rdata


def save_rdata(path: str | Path, **objects: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(objects, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _as_array(obj: Any) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, pd.Series):
        return obj.to_numpy()
    return np.asarray(obj)


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cvr = load_rdata(TEMPDIR / "CVR_9June2021.Rdata")
    if "complete_df" not in cvr:
        raise KeyError("CVR_9June2021.Rdata must contain complete_df")
    complete_df = pd.DataFrame(cvr["complete_df"]).copy()
    complete_df = complete_df.iloc[:, 1:].copy()
    complete_df = complete_df.rename(columns={complete_df.columns[1]: "sexM"})
    complete_df["headsize"] = (
        complete_df["X25000.2.0"] + complete_df["X25000.3.0"]
    ) / 2.0
    complete_df = complete_df[
        ["eid_8107", "age_vis2", "age_vis3", "sexM", "headsize", "CVR_vis2", "CVR_vis3"]
    ]
    complete_df["age_diff"] = complete_df["age_vis3"] - complete_df["age_vis2"]
    complete_df["CVR_diff"] = complete_df["CVR_vis3"] - complete_df["CVR_vis2"]
    for col in ("age_vis2", "headsize", "CVR_vis2"):
        complete_df[col] = complete_df[col] - complete_df[col].mean()

    lesions = load_rdata(TEMPDIR / "lesions_atleast6_CVR.RData")
    lesions1 = _as_array(lesions["lesions_vis1"])
    lesions2 = _as_array(lesions["lesions_vis2"])
    return complete_df, complete_df["eid_8107"].to_numpy(), lesions1, lesions2


def subset_indices(n_rows: int, j: int, subset_size: int = 500) -> np.ndarray:
    n_subsets = int(math.ceil(n_rows / subset_size))
    if j < 1 or j > n_subsets:
        raise ValueError(f"subset j={j} outside 1..{n_subsets}")
    start = subset_size * (j - 1)
    stop = n_rows if j == n_subsets else subset_size * j
    return np.arange(start, stop, dtype=int)


def panel_design(
    local_i: int, include_interactions: bool = True, intercept: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert (
        _COMPLETE_DF is not None
        and _IDS is not None
        and _LESIONS1 is not None
        and _LESIONS2 is not None
        and _SUBSET_IDX is not None
    )
    n_subj = _LESIONS1.shape[1]
    row = _SUBSET_IDX[local_i]
    panel = pd.DataFrame(
        {
            "y": np.r_[_LESIONS1[row, :], _LESIONS2[row, :]],
            "vis": np.r_[np.ones(n_subj, dtype=int), np.full(n_subj, 2, dtype=int)],
            "eid_8107": np.r_[_IDS, _IDS],
        }
    )
    panel = panel.merge(_COMPLETE_DF, on="eid_8107", how="left").sort_values(
        ["eid_8107", "vis"]
    )
    second_visit = np.tile([0.0, 1.0], n_subj)
    panel["age_diff"] = panel["age_diff"].to_numpy() * second_visit
    panel["CVR_diff"] = panel["CVR_diff"].to_numpy() * second_visit
    cols = []
    if intercept:
        cols.append(np.ones(len(panel)))
    cols.extend(
        [
            panel["age_vis2"].to_numpy(float),
            panel["age_diff"].to_numpy(float),
            panel["CVR_vis2"].to_numpy(float),
            panel["CVR_diff"].to_numpy(float),
            panel["sexM"].to_numpy(float),
            panel["headsize"].to_numpy(float),
        ]
    )
    if include_interactions:
        cols.extend(
            [
                panel["age_vis2"].to_numpy(float) * panel["age_diff"].to_numpy(float),
                panel["age_vis2"].to_numpy(float) * panel["sexM"].to_numpy(float),
            ]
        )
    return (
        panel["y"].to_numpy(float),
        np.column_stack(cols),
        panel["eid_8107"].to_numpy(),
    )


def _safe_inv(a: np.ndarray) -> np.ndarray:
    try:
        return linalg.solve(a, np.eye(a.shape[0]), assume_a="sym")
    except Exception:
        return np.linalg.pinv(a)


def gee_penalty_run(
    y: np.ndarray,
    X: np.ndarray,
    n_subj: int,
    n_visits: int,
    covariance: str = "Independence",
    tol: float = 1e-4,
    max_iter: int = 10,
    phi_est: bool = True,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    n_obs = n_subj * n_visits
    beta_old = np.r_[math.log(max(float(np.mean(y)), 1e-8)), np.zeros(p - 1)]
    mu = np.exp(np.clip(X @ beta_old, -30, 30))
    I = X.T @ (mu[:, None] * X)
    U = X.T @ (y - mu)
    alpha = 0.0
    phi = 1.0
    se_model_trace: list[np.ndarray] = []
    H_diag = np.full(n_obs, np.nan)
    conv = False
    I_inv = _safe_inv(I)

    for iteration in range(1, max_iter + 1):
        beta_new = beta_old + I_inv @ U
        mu = np.exp(np.clip(X @ beta_new, -30, 30))
        res = (y - mu) / np.sqrt(np.maximum(mu, 1e-12))

        if covariance == "Independence":
            R = np.eye(n_visits)
            alpha = 0.0
        elif covariance == "Exchangeable":
            if phi_est:
                denom = max(n_obs - p, 1)
                phi = 1.0 / max(float(np.sum(res * res) / denom), 1e-12)
            sum_res = 0.0
            for s in range(n_subj):
                r = res[s * n_visits : (s + 1) * n_visits]
                rr = np.outer(r, r)
                sum_res += float(rr.sum() - np.trace(rr))
            alpha = phi * (sum_res / (n_visits * (n_visits - 1))) / n_subj
            alpha = float(np.clip(alpha, -0.95 / max(n_visits - 1, 1), 0.95))
            R = np.full((n_visits, n_visits), alpha)
            np.fill_diagonal(R, 1.0)
        else:
            raise ValueError(f"Unknown covariance structure: {covariance}")

        R_inv = _safe_inv(R)
        tmp2 = mu[:, None] * X
        tmp4 = np.zeros((n_obs, n_obs), dtype=float)
        for s in range(n_subj):
            idx = slice(s * n_visits, (s + 1) * n_visits)
            scale = np.diag(1.0 / np.sqrt(np.maximum(mu[idx], 1e-12)))
            tmp4[idx, idx] = scale @ R_inv @ scale * phi
        tmp5 = tmp2.T @ tmp4
        I = tmp5 @ tmp2
        I_inv = _safe_inv(I)
        H = np.diag(mu) @ tmp4 @ tmp2 @ I_inv @ X.T
        H_diag = np.diag(H)
        U = tmp5 @ (y - mu) + (H_diag[:, None] * X / 2.0).sum(axis=0)
        se_model = np.sqrt(np.maximum(np.diag(I_inv), 0.0))
        se_model_trace.append(se_model)

        if np.all(np.abs(beta_old - beta_new) < tol):
            conv = True
            beta_old = beta_new
            break
        beta_old = beta_new

    residual_outer = np.outer(y - mu, y - mu)
    cluster_mask = np.kron(np.eye(n_subj), np.ones((n_visits, n_visits)))
    meat = tmp5 @ (cluster_mask * residual_outer) @ tmp5.T
    sandwich = I_inv @ meat @ I_inv
    return {
        "beta": beta_old,
        "beta_se_model": se_model,
        "beta_se_model_trace": (
            np.vstack(se_model_trace) if se_model_trace else np.empty((0, p))
        ),
        "beta_se_sandwich": np.sqrt(np.maximum(np.diag(sandwich), 0.0)),
        "alpha": alpha,
        "phi": phi,
        "iterations": len(se_model_trace),
        "H": H_diag,
        "conv": conv,
    }


def gee_dispersion_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return gee_penalty_run(*args, **kwargs)


def statsmodels_gee_run(
    y: np.ndarray, X: np.ndarray, groups: np.ndarray, family: str = "poisson"
) -> dict[str, Any]:
    fam = (
        sm.families.Poisson(sm.families.links.Log())
        if family == "poisson"
        else sm.families.Binomial()
    )
    model = sm.GEE(
        y, X, groups=groups, family=fam, cov_struct=sm.cov_struct.Exchangeable()
    )
    result = model.fit(maxiter=25)
    alpha = getattr(result.cov_struct, "dep_params", np.nan)
    fit_history = getattr(result, "fit_history", {}) or {}
    iteration = fit_history.get("iteration", 0)
    if isinstance(iteration, (list, tuple, np.ndarray)):
        iteration = len(iteration)
    try:
        iteration = 0 if not np.isfinite(iteration) else int(iteration)
    except Exception:
        iteration = 0
    return {
        "beta": np.asarray(result.params),
        "beta_se_model": np.asarray(result.bse),
        "beta_se_model_trace": np.empty((0, X.shape[1])),
        "beta_se_sandwich": np.asarray(result.bse),
        "alpha": float(np.ravel(alpha)[0]) if np.size(alpha) else np.nan,
        "phi": float(result.scale),
        "iterations": iteration,
        "H": np.full(len(y), np.nan),
        "conv": bool(getattr(result, "converged", True)),
    }


def _init_worker(
    complete_df: pd.DataFrame,
    ids: np.ndarray,
    lesions1: np.ndarray,
    lesions2: np.ndarray,
    subset_idx: np.ndarray,
    model_kind: str,
    include_interactions: bool,
) -> None:
    global _COMPLETE_DF, _IDS, _LESIONS1, _LESIONS2, _SUBSET_IDX, _MODEL_KIND, _INCLUDE_INTERACTIONS
    _COMPLETE_DF, _IDS, _LESIONS1, _LESIONS2, _SUBSET_IDX = (
        complete_df,
        ids,
        lesions1,
        lesions2,
        subset_idx,
    )
    _MODEL_KIND, _INCLUDE_INTERACTIONS = model_kind, include_interactions


def fit_voxel(local_i: int) -> dict[str, Any]:
    try:
        intercept = _MODEL_KIND != "or_pgee"
        y, X, eid = panel_design(
            local_i, include_interactions=_INCLUDE_INTERACTIONS, intercept=intercept
        )
        n_visits = len(np.unique(np.tile([1, 2], len(y) // 2)))
        n_subj = len(y) // n_visits
        groups = np.repeat(np.arange(n_subj), n_visits)
        if _MODEL_KIND == "pgee":
            model = gee_penalty_run(
                y, X, n_subj, n_visits, "Exchangeable", tol=1e-3, max_iter=25
            )
        elif _MODEL_KIND == "gee":
            model = statsmodels_gee_run(y, X, groups, "poisson")
        elif _MODEL_KIND == "or_pgee":
            model = statsmodels_gee_run(y, X, groups, "binomial")
            model["beta_se_sandwich_modified"] = model["beta_se_sandwich"]
        else:
            raise ValueError(_MODEL_KIND)
        if (local_i + 1) % 100 == 0:
            print(local_i + 1, flush=True)
        model["voxel"] = local_i + 1
        return model
    except Exception as exc:
        return {"error": repr(exc), "voxel": local_i + 1}


def run_subset(
    n_cores: int,
    j: int,
    out_dir: str | Path,
    model_kind: str = "pgee",
    include_interactions: bool = True,
) -> None:
    print("Everything loaded")
    complete_df, ids, lesions1, lesions2 = load_inputs()
    subset_idx = subset_indices(lesions1.shape[0], j)
    print("Start running gee foreach fn")
    print(j)
    n_workers = max(int(n_cores), 1)
    with Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(
            complete_df,
            ids,
            lesions1,
            lesions2,
            subset_idx,
            model_kind,
            include_interactions,
        ),
    ) as pool:
        output = pool.map(fit_voxel, range(len(subset_idx)))
    save_rdata(Path(out_dir) / f"GEE_subset_{j}.RData", output=output)
