"""Log-Poisson GEE with dispersion, ported from gee_logPoisson_dispersion_fn.R."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag


def ar1_cor(n: int, rho: float) -> np.ndarray:
    idx = np.arange(int(n))
    return float(rho) ** np.abs(idx[:, None] - idx[None, :])


def _inv(a: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)


def _block_diag_repeat(mat: np.ndarray, times: int) -> np.ndarray:
    return block_diag(*([mat] * int(times)))


def _cluster_mask(n_subj: int, n_visits: int) -> np.ndarray:
    blocks = [
        np.ones((int(n_visits), int(n_visits)), dtype=float) for _ in range(int(n_subj))
    ]
    return block_diag(*blocks)


def _gee_core(
    y,
    X,
    n_subj: int,
    n_visits: int,
    covariance="Independence",
    tol=1e-4,
    max_iter=10,
    phi_est=True,
    penalty="none",
):
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n_subj, n_visits = int(n_subj), int(n_visits)
    p = X.shape[1]
    n_obs = n_subj * n_visits
    if len(y) != n_obs or X.shape[0] != n_obs:
        raise ValueError("y and X dimensions must equal n_subj * n_visits")

    beta_old = np.r_[np.log(np.nanmean(y) + 0.0001), np.zeros(p - 1)]
    mu = np.clip(np.exp(X @ beta_old), 1e-12, 1e12)
    I = X.T @ np.diag(mu) @ X
    U = X.T @ (y - mu)
    I_inv = _inv(I)
    alpha = 0.0
    phi = 1.0
    R = np.eye(n_visits)
    se_model = np.full(p, np.nan)
    se_model_trace = []
    conv = False
    tmp2 = np.diag(mu) @ X
    tmp5 = X.T

    for iteration in range(1, int(max_iter) + 1):
        beta_new = beta_old + I_inv @ U
        mu = np.clip(np.exp(X @ beta_new), 1e-12, 1e12)

        cov_key = str(covariance).lower()
        if cov_key == "independence":
            alpha = 0.0
            R = np.eye(n_visits)
        elif cov_key == "exchangeable":
            res = (y - mu) / np.sqrt(mu)
            if phi_est:
                denom = max(n_obs - p, 1)
                phi = 1.0 / max(np.sum(res**2) / denom, 1e-12)
            sum_res = 0.0
            for subject in range(n_subj):
                idx = slice(subject * n_visits, (subject + 1) * n_visits)
                rr = np.outer(res[idx], res[idx])
                sum_res += rr.sum() - np.trace(rr)
            alpha = (
                phi * (sum_res / (n_visits * (n_visits - 1))) / n_subj
                if n_visits > 1
                else 0.0
            )
            alpha = float(np.clip(alpha, -0.95, 0.95))
            R = np.full((n_visits, n_visits), alpha)
            np.fill_diagonal(R, 1.0)
        elif cov_key == "ar1":
            res = (y - mu) / np.sqrt(mu)
            vals = []
            for subject in range(n_subj):
                r = res[subject * n_visits : (subject + 1) * n_visits]
                vals.extend(r[:-1] * r[1:])
            alpha = float(np.clip(phi * np.mean(vals), -0.95, 0.95)) if vals else 0.0
            R = ar1_cor(n_visits, alpha)
        else:
            raise ValueError("Unknown covariance structure")

        R_inv = _inv(R)
        inv_blocks = []
        for subject in range(n_subj):
            idx = slice(subject * n_visits, (subject + 1) * n_visits)
            D_inv = np.diag(1.0 / np.sqrt(mu[idx]))
            inv_blocks.append(D_inv @ R_inv @ D_inv * phi)
        V_inv = block_diag(*inv_blocks)
        tmp2 = np.diag(mu) @ X
        tmp5 = tmp2.T @ V_inv
        I = tmp5 @ tmp2
        I_inv = _inv(I)

        score = tmp5 @ (y - mu)
        H_diag = None
        if penalty == "qr":
            sqrt_blocks = []
            for subject in range(n_subj):
                idx = slice(subject * n_visits, (subject + 1) * n_visits)
                D = np.diag(np.sqrt(mu[idx]))
                vals, vecs = np.linalg.eigh((D @ R @ D + (D @ R @ D).T) / 2.0)
                sqrt_blocks.append((vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T)
            Q, _ = np.linalg.qr(block_diag(*sqrt_blocks) @ X, mode="reduced")
            H_diag = np.sum(Q * Q, axis=1)
            score = score + X.T @ (H_diag / 2.0)
        elif penalty == "ik":
            H = np.diag(mu) @ V_inv @ tmp2 @ I_inv @ X.T
            H_diag = np.diag(H)
            score = score + X.T @ (H_diag / 2.0)
        U = score

        se_model = np.sqrt(np.clip(np.diag(I_inv), 0, None))
        se_model_trace.append(se_model.copy())
        if np.all(np.abs(beta_old - beta_new) < tol):
            conv = True
            beta_old = beta_new
            break
        beta_old = beta_new

    beta = beta_old
    resid_outer = np.outer(y - mu, y - mu) * _cluster_mask(n_subj, n_visits)
    sandwich = I_inv @ (tmp5 @ resid_outer @ tmp5.T) @ I_inv
    se_sandwich = np.sqrt(np.clip(np.diag(sandwich), 0, None))
    out = {
        "beta": beta,
        "beta_se_model": se_model,
        "beta_se_model_trace": np.asarray(se_model_trace),
        "beta_se_sandwich": se_sandwich,
        "alpha": alpha,
        "phi": phi,
        "iterations": len(se_model_trace),
        "conv": conv,
    }
    if H_diag is not None:
        out["H"] = H_diag
    return out


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
    return _gee_core(
        y, X, n_subj, n_visits, covariance, tol, max_iter, phi_est, penalty="none"
    )
