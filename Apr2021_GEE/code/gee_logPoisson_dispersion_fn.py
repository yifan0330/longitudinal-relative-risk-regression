#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _inv(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)


def _exchangeable(alpha: float, n_visits: int) -> np.ndarray:
    r = np.full((n_visits, n_visits), alpha, dtype=float)
    np.fill_diagonal(r, 1.0)
    return r


def _sandwich(tmp5_blocks, residual_blocks, i_inv):
    meat = np.zeros_like(i_inv)
    for tmp5_i, res_i in zip(tmp5_blocks, residual_blocks):
        meat += tmp5_i @ np.outer(res_i, res_i) @ tmp5_i.T
    return i_inv @ meat @ i_inv


def gee_dispersion_run(
    y,
    X,
    n_subj,
    n_visits,
    covariance="Independence",
    tol=1e-4,
    max_iter=10,
    phi_est=True,
):
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n_subj, n_visits = int(n_subj), int(n_visits)
    n_obs, p = n_subj * n_visits, X.shape[1]
    if np.all(y == 0):
        return np.nan
    beta_old = np.r_[np.log(np.mean(y)), np.zeros(p - 1)]
    mu = np.exp(X @ beta_old)
    iter_I = X.T @ (mu[:, None] * X)
    iter_U = X.T @ (y - mu)
    alpha = 0.0
    phi = 1.0
    se_model_trace = []
    beta_new = beta_old.copy()
    iter_I_inv = _inv(iter_I)
    tmp5_blocks = []
    residual_blocks = []
    iterations = 0
    for iterations in range(1, max_iter + 1):
        beta_new = beta_old + iter_I_inv @ iter_U
        mu = np.exp(X @ beta_new)
        if covariance == "Independence":
            alpha = 0.0
            R = np.eye(n_visits)
        elif covariance == "Exchangeable":
            pearson = (y - mu) / np.sqrt(mu)
            if phi_est:
                phi = 1.0 / (np.sum(pearson**2) / max(n_obs - p, 1))
            pair_sum = 0.0
            for s in range(n_subj):
                sl = slice(s * n_visits, (s + 1) * n_visits)
                outer = np.outer(pearson[sl], pearson[sl])
                pair_sum += outer.sum() - np.trace(outer)
            alpha = phi * (pair_sum / (n_visits * (n_visits - 1))) / n_subj
            R = _exchangeable(alpha, n_visits)
        else:
            raise ValueError(f"Unknown covariance structure: {covariance}")
        R_inv = _inv(R)
        iter_I = np.zeros((p, p), dtype=float)
        iter_U = np.zeros(p, dtype=float)
        tmp5_blocks = []
        residual_blocks = []
        for s in range(n_subj):
            sl = slice(s * n_visits, (s + 1) * n_visits)
            Xi = X[sl, :]
            mui = mu[sl]
            Di = mui[:, None] * Xi
            Vinv = (R_inv / np.sqrt(np.outer(mui, mui))) * phi
            tmp5_i = Di.T @ Vinv
            res_i = y[sl] - mui
            iter_I += tmp5_i @ Di
            iter_U += tmp5_i @ res_i
            tmp5_blocks.append(tmp5_i)
            residual_blocks.append(res_i)
        iter_I_inv = _inv(iter_I)
        se_model_trace.append(np.sqrt(np.diag(iter_I_inv)))
        if np.all(np.abs(beta_old - beta_new) < tol):
            break
        beta_old = beta_new
    sandwich = _sandwich(tmp5_blocks, residual_blocks, iter_I_inv)
    se_sandwich = np.sqrt(np.diag(sandwich))
    return {
        "beta": beta_new,
        "beta_se_model": se_model_trace[-1],
        "beta_se_model_trace": np.vstack(se_model_trace),
        "beta_se_sandwich": se_sandwich,
        "alpha": alpha,
        "phi": phi,
        "iterations": iterations,
    }
