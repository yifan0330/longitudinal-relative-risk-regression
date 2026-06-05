#!/usr/bin/env python3
"""Python port of the custom GEE routines from the matching R script."""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _as_col(y):
    return np.asarray(y, dtype=float).reshape(-1, 1)


def _as_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _spdinv(a):
    a = np.asarray(a, dtype=float)
    try:
        return linalg.pinvh((a + a.T) / 2.0)
    except Exception:
        return np.linalg.pinv(a)


def _safe_inv(a):
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)


def _ar1_cor(n, rho):
    idx = np.arange(n)
    return float(rho) ** np.abs(idx[:, None] - idx[None, :])


def _block_diag(mat, n_blocks):
    return linalg.block_diag(*([mat] * int(n_blocks)))


def _block_residual_outer(residual, n_subj, n_visits):
    residual = np.asarray(residual, dtype=float).reshape(-1)
    out = np.zeros((n_subj * n_visits, n_subj * n_visits), dtype=float)
    for s in range(n_subj):
        sl = slice(s * n_visits, (s + 1) * n_visits)
        r = residual[sl].reshape(-1, 1)
        out[sl, sl] = r @ r.T
    return out


def _poisson_alpha(y, mu, n_subj, n_visits, covariance):
    covariance = covariance.lower()
    if covariance == "independence":
        return 0.0, np.eye(n_visits)
    res = ((y - mu) / np.sqrt(np.clip(mu, 1e-12, None))).reshape(-1)
    if covariance == "ar1":
        res_temp = np.zeros((n_visits, n_visits), dtype=float)
        for s in range(n_subj):
            r = res[s * n_visits : (s + 1) * n_visits].reshape(-1, 1)
            res_temp += r @ r.T
        alpha = sum(res_temp[i, i + 1] for i in range(n_visits - 1)) / (
            (n_visits - 1) * n_subj
        )
        return float(alpha), _ar1_cor(n_visits, alpha)
    if covariance == "exchangeable":
        total = 0.0
        for s in range(n_subj):
            r = res[s * n_visits : (s + 1) * n_visits].reshape(-1, 1)
            rr = r @ r.T
            total += rr.sum() - np.trace(rr)
        alpha = total / (n_visits * (n_visits - 1) * n_subj)
        R = np.full((n_visits, n_visits), alpha, dtype=float)
        np.fill_diagonal(R, 1.0)
        return float(alpha), R
    raise ValueError("Unknown covariance structure!")


def _finish_iter(beta_old, beta_new, tol):
    return bool(np.all(np.abs(beta_old.reshape(-1, 1) - beta_new.reshape(-1, 1)) < tol))


def gee_run(
    y,
    X,
    n_subj,
    n_visits,
    covariance="Independence",
    tol=1e-3,
    max_iter=10,
    verbose=True,
):
    y = _as_col(y)
    X = _as_matrix(X)
    n_subj, n_visits, max_iter = int(n_subj), int(n_visits), int(max_iter)
    P = X.shape[1]
    beta_old = np.r_[np.log(np.mean(y)), np.zeros(P - 1)].reshape(-1, 1)
    mu = np.exp(X @ beta_old)
    I = X.T @ np.diagflat(mu) @ X
    U = X.T @ (y - mu)
    alpha = 0.0
    R = np.eye(n_visits)
    iterations = 0
    se_trace = np.full((max_iter, P), np.nan)
    se_model_trace = np.full((max_iter, P), np.nan)

    for it in range(max_iter):
        if verbose:
            print("Iteration:", it + 1)
        I_inv = _spdinv(I)
        beta_new = beta_old + I_inv @ U
        if verbose:
            print("-- beta --")
            print(beta_new)
        mu = np.exp(X @ beta_new)
        alpha, R = _poisson_alpha(y, mu, n_subj, n_visits, covariance)
        if verbose and covariance.lower() == "exchangeable":
            print("-- alpha --")
            print(alpha)

        tmp1 = np.diagflat(np.sqrt(1.0 / np.clip(mu, 1e-12, None)))
        tmp2 = np.diagflat(mu) @ X
        R_inv = _safe_inv(R)
        if verbose:
            print("-- R inverse --")
            print(R_inv)
        tmp4 = tmp1 @ _block_diag(R_inv, n_subj) @ tmp1
        tmp5 = tmp2.T @ tmp4
        I = tmp5 @ tmp2
        U = tmp5 @ (y - mu)
        I_inv = _spdinv(I)
        se_model = np.sqrt(np.clip(np.diag(I_inv), 0, None))
        se_model_trace[iterations, :] = se_model
        temp_v2 = _block_residual_outer(y - mu, n_subj, n_visits)
        temp_v3 = tmp5 @ temp_v2 @ tmp5.T
        cov = I_inv @ temp_v3 @ I_inv
        se_sandwich = np.sqrt(np.clip(np.diag(cov), 0, None))
        se_trace[iterations, :] = se_sandwich
        if verbose:
            print("-- SE sandwich --")
            print(se_sandwich)
        iterations += 1
        if _finish_iter(beta_old, beta_new, tol):
            break
        beta_old = beta_new

    return {
        "beta": beta_new.reshape(-1),
        "beta_se_model": se_model,
        "beta_se_model_trace": se_model_trace[:iterations, :],
        "beta_se_sandwich": se_sandwich,
        "beta_se_sandwich_trace": se_trace[:iterations, :],
        "alpha": alpha,
        "iterations": iterations,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import gee_run() from this module; it is not a command-line runner."
    )
