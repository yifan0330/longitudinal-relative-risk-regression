"""Thin wrappers around the existing validated fitting routines."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .config import MAX_ITER, REPO_ROOT, STRICT_ORIGINAL_IRLS, TOL, Scenario
from .data_generation import model_matrix

SIM_CODE_DIR = REPO_ROOT / "Simulations" / "code"
PGEE_DIR = REPO_ROOT / "PGEE_Mondol"
for path in (SIM_CODE_DIR, PGEE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gee_logPoisson_dispersion_fn import gee_dispersion_run  # noqa: E402
from Mar23_PGEE_source import geefirth  # noqa: E402
from Sept21_pgee_logPoisson_dispersion_fn import gee_penalty_run  # noqa: E402


METHOD_RR_GEE = "RR-GEE"
METHOD_RR_PGEE = "RR-PGEE"
METHOD_OR_GEE = "OR-GEE-ZY"
METHOD_OR_PGEE = "OR-PGEE-ZY"
METHOD_ORDER: tuple[str, ...] = (METHOD_RR_GEE, METHOD_RR_PGEE, METHOD_OR_GEE, METHOD_OR_PGEE)
OR_METHODS: frozenset[str] = frozenset({METHOD_OR_GEE, METHOD_OR_PGEE})

FIT_ENGINE_IRLS = "irls"
FIT_ENGINE_LBFGS_OR_GEE = "lbfgs-or-gee"
FIT_ENGINES = (FIT_ENGINE_IRLS, FIT_ENGINE_LBFGS_OR_GEE)


@dataclass(frozen=True)
class FitRecord:
    """One method's fit result for one replication."""

    method: str
    converged: bool
    finite: bool
    log_effect: float
    se_log_effect: float
    rr_estimate: float
    p0_hat: float
    iterations: float
    alpha: float
    phi: float
    bec_count: float
    failure_reason: str = ""


def fit_all_methods(
    data: pd.DataFrame,
    scenario: Scenario,
    *,
    max_iter: int = MAX_ITER,
    fit_engine: str = FIT_ENGINE_IRLS,
) -> list[FitRecord]:
    """Fit RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE to the same dataset."""
    if fit_engine not in FIT_ENGINES:
        raise ValueError(f"Unknown fit engine: {fit_engine}")
    or_gee_record = (
        fit_or_lbfgs_method(METHOD_OR_GEE, data, scenario, max_iter=max_iter)
        if fit_engine == FIT_ENGINE_LBFGS_OR_GEE
        else fit_or_method(METHOD_OR_GEE, data, scenario, penalized=False, max_iter=max_iter)
    )
    return [
        fit_rr_method(METHOD_RR_GEE, data, scenario, gee_dispersion_run, max_iter=max_iter),
        fit_rr_method(METHOD_RR_PGEE, data, scenario, gee_penalty_run, max_iter=max_iter),
        or_gee_record,
        fit_or_method(METHOD_OR_PGEE, data, scenario, penalized=True, max_iter=max_iter),
    ]


def fit_rr_method(
    method: str,
    data: pd.DataFrame,
    scenario: Scenario,
    fitter: Any,
    *,
    max_iter: int = MAX_ITER,
) -> FitRecord:
    """Fit an RR-scale log-link GEE method."""
    X = model_matrix(data)
    y = data["yij"].to_numpy(float)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fit = fitter(
                y=y,
                X=X,
                n_subj=scenario.n_subjects,
                n_visits=scenario.n_visits,
                covariance="Exchangeable",
                tol=TOL,
                max_iter=max_iter,
                phi_est=True,
                strict_original=STRICT_ORIGINAL_IRLS,
            )
        beta = np.asarray(fit["beta"], dtype=float)
        se = np.asarray(fit["beta_se_sandwich"], dtype=float)
        log_effect = float(beta[1])
        se_log_effect = float(se[1])
        rr_estimate = _safe_exp(log_effect)
        converged = bool(fit["conv"])
        finite = bool(
            np.isfinite(log_effect) and np.isfinite(se_log_effect) and np.isfinite(rr_estimate)
        )
        return FitRecord(
            method=method,
            converged=converged,
            finite=finite,
            log_effect=log_effect,
            se_log_effect=se_log_effect,
            rr_estimate=rr_estimate,
            p0_hat=np.nan,
            iterations=float(fit["iterations"]),
            alpha=float(fit["alpha"]),
            phi=float(fit["phi"]),
            bec_count=_bec_value_rr(fit),
        )
    except Exception as exc:
        return _failed_record(method, repr(exc))


def fit_or_method(
    method: str,
    data: pd.DataFrame,
    scenario: Scenario,
    *,
    penalized: bool,
    max_iter: int = MAX_ITER,
) -> FitRecord:
    """Fit logistic OR-GEE/OR-PGEE and transform the binary-covariate OR to an RR."""
    X = model_matrix(data)
    y = data["yij"].to_numpy(float)
    ids = data["id"].to_numpy(int)
    x_without_intercept = X[:, 1:3]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fit = geefirth(
                y=y,
                x=x_without_intercept,
                id=ids,
                ar=False,
                max_iter=max_iter,
                tol=TOL,
                strict_original=STRICT_ORIGINAL_IRLS,
                penalized=penalized,
            )
        coef_table = fit[0]
        beta = coef_table["coefficients"].to_numpy(dtype=float)
        se = coef_table["std.err"].to_numpy(dtype=float)
        if len(beta) != 3:
            raise ValueError(f"Expected three {method} coefficients, received {len(beta)}")
        log_or = float(beta[1])
        se_log_or = float(se[1])
        p0_hat = estimate_p0_observed(data)
        rr_estimate = zhang_yu_rr(log_or, p0_hat)
        finite = bool(np.isfinite(log_or) and np.isfinite(se_log_or) and np.isfinite(rr_estimate))
        iterations = float(fit[3])
        return FitRecord(
            method=method,
            converged=iterations < max_iter,
            finite=finite,
            log_effect=log_or,
            se_log_effect=se_log_or,
            rr_estimate=rr_estimate,
            p0_hat=p0_hat,
            iterations=iterations,
            alpha=float(fit[2]),
            phi=float(fit[4]),
            bec_count=_bec_value_or_pgee(fit),
        )
    except Exception as exc:
        return _failed_record(method, repr(exc))


def fit_or_lbfgs_method(
    method: str,
    data: pd.DataFrame,
    scenario: Scenario,
    *,
    max_iter: int = MAX_ITER,
) -> FitRecord:
    """Fit the unpenalized logistic OR-GEE coefficient with L-BFGS-B.

    This benchmark path uses an independence logistic likelihood for the
    coefficient optimization and cluster-robust sandwich SEs afterward. The
    penalized OR-PGEE estimator remains on the validated IRLS path.
    """
    if method != METHOD_OR_GEE:
        raise ValueError("L-BFGS is only implemented for OR-GEE-ZY.")
    X = model_matrix(data)
    y = data["yij"].to_numpy(float)
    ids = data["id"].to_numpy(int)
    init = np.r_[np.log(np.mean(y) + 0.0001), np.zeros(X.shape[1] - 1)]
    try:
        result = minimize(
            _logistic_negative_log_likelihood,
            init,
            args=(X, y),
            jac=_logistic_negative_log_likelihood_gradient,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": TOL, "gtol": TOL},
        )
        beta = np.asarray(result.x, dtype=float)
        se, alpha, phi = _cluster_robust_logistic_summary(X, y, ids, beta)
        log_or = float(beta[1])
        se_log_or = float(se[1])
        p0_hat = estimate_p0_observed(data)
        rr_estimate = zhang_yu_rr(log_or, p0_hat)
        finite = bool(np.isfinite(log_or) and np.isfinite(se_log_or) and np.isfinite(rr_estimate))
        return FitRecord(
            method=method,
            converged=bool(result.success),
            finite=finite,
            log_effect=log_or,
            se_log_effect=se_log_or,
            rr_estimate=rr_estimate,
            p0_hat=p0_hat,
            iterations=float(result.nit),
            alpha=alpha,
            phi=phi,
            bec_count=np.nan,
            failure_reason="" if result.success else str(result.message),
        )
    except Exception as exc:
        return _failed_record(method, repr(exc))


def estimate_p0_observed(data: pd.DataFrame) -> float:
    """Observed event proportion among unexposed observations."""
    unexposed = data.loc[data["X1i"] == 0, "yij"]
    if unexposed.empty:
        return np.nan
    return float(unexposed.mean())


def _logistic_negative_log_likelihood(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def _logistic_negative_log_likelihood_gradient(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    return X.T @ (expit(X @ beta) - y)


def _cluster_robust_logistic_summary(
    X: np.ndarray, y: np.ndarray, ids: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mu = expit(X @ beta)
    var = np.clip(mu * (1.0 - mu), 1e-12, None)
    information = X.T @ (X * var[:, None])
    information_inv = np.linalg.pinv(information)

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    residual = y - mu
    standardized = residual / np.sqrt(var)
    alpha_values: list[float] = []
    for cluster_id in pd.unique(ids):
        mask = ids == cluster_id
        score = X[mask].T @ residual[mask]
        meat += np.outer(score, score)
        cluster_residual = standardized[mask]
        if cluster_residual.size > 1:
            outer = np.outer(cluster_residual, cluster_residual)
            denom = cluster_residual.size * (cluster_residual.size - 1)
            alpha_values.append(float((outer.sum() - np.trace(outer)) / denom))

    covariance = information_inv @ meat @ information_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    alpha = float(np.clip(np.mean(alpha_values), -0.95, 0.95)) if alpha_values else 0.0
    phi = float(np.mean(standardized**2))
    return se, alpha, phi


def zhang_yu_rr(log_or: float, p0: float) -> float:
    """Transform a log odds-ratio to a relative risk using Zhang-Yu."""
    if not np.isfinite(log_or) or not np.isfinite(p0) or p0 < 0.0 or p0 > 1.0:
        return np.nan
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = p0 + (1.0 - p0) * np.exp(-log_or)
    if denominator == 0.0:
        return np.inf
    if not np.isfinite(denominator):
        return 0.0 if denominator > 0 else np.nan
    return float(1.0 / denominator)


def _safe_exp(value: float) -> float:
    """Exponentiate a scalar without emitting overflow warnings."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.exp(value))


def _bec_value_rr(fit: dict[str, Any]) -> float:
    """Return the historical BEC diagnostic: max model-SE ratio."""
    model_se = np.asarray(fit["beta_se_model"], dtype=float)
    trace = np.asarray(fit["beta_se_model_trace"], dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = model_se / trace[0]
    return float(np.nanmax(ratio))


def _bec_value_or_pgee(fit: Any) -> float:
    """Return the historical OR-PGEE BEC diagnostic: max model-SE ratio."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.asarray(fit[5], dtype=float) / np.asarray(fit[6], dtype=float)[0, :]
    return float(np.nanmax(ratio))


def _failed_record(method: str, reason: str) -> FitRecord:
    return FitRecord(
        method=method,
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
        failure_reason=reason,
    )
