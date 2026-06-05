"""Simulation data generation formerly implemented in source_simdata.R."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def xch(n: int, rho: float) -> np.ndarray:
    """Exchangeable correlation matrix."""
    mat = np.full((int(n), int(n)), float(rho), dtype=float)
    np.fill_diagonal(mat, 1.0)
    return mat


def cor2var(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix and Bernoulli means to a covariance matrix."""
    mu = np.asarray(mu, dtype=float)
    d = np.sqrt(np.clip(mu, 0.0, None))
    return np.diag(d) @ np.asarray(r, dtype=float) @ np.diag(d)


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-8, None)
    corr = (vecs * vals) @ vecs.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def simulate_correlated_bernoulli(
    probs: np.ndarray, corr: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Approximate binarySimCLF with a Gaussian-copula Bernoulli simulator.

    The R package binarySimCLF constructs Bernoulli vectors for compatible mean/
    covariance pairs.  This Python port uses a stable Gaussian-copula
    approximation and clips probabilities to valid Bernoulli ranges.
    """
    rng = np.random.default_rng() if rng is None else rng
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    corr = _nearest_correlation(corr)
    z = rng.multivariate_normal(np.zeros(len(probs)), corr)
    return (z <= norm.ppf(probs)).astype(int)


def gen_dataPP(
    beta,
    nc: int,
    cl_size=range(2, 7),
    p: float | None = None,
    rho: float = 0.0,
    prop: float | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate clustered binary data for the log-risk simulations.

    Parameters mirror the original R function.  ``p`` is the Bernoulli
    prevalence for the cluster-level covariate; ``prop`` is accepted as an alias
    because the calling scripts used that name.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta = np.asarray(beta, dtype=float)
    if p is None:
        if prop is None:
            raise ValueError("Either p or prop must be supplied")
        p = prop
    if np.isscalar(cl_size):
        sizes = np.repeat(int(cl_size), int(nc))
    else:
        vals = list(cl_size)
        lo, hi = float(vals[0]), float(vals[-1])
        sizes = np.rint(rng.uniform(lo, hi, int(nc))).astype(int)
    n_obs = int(sizes.sum())
    x1_cluster = rng.binomial(1, float(p), int(nc))
    x1 = np.repeat(x1_cluster, sizes)
    obstime = np.concatenate([0.2 * np.arange(1, s + 1) for s in sizes])
    intercept = np.ones(n_obs)
    design = np.column_stack([intercept, x1, obstime])
    probs = np.exp(design @ beta)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    ids = np.repeat(np.arange(1, int(nc) + 1), sizes)

    y_parts: list[np.ndarray] = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        corr = xch(int(size), rho)
        y_parts.append(simulate_correlated_bernoulli(probs[start:stop], corr, rng))
        start = stop
    y = np.concatenate(y_parts)
    return pd.DataFrame(
        {"id": ids, "yij": y, "intercept": intercept, "X1i": x1, "obstime": obstime}
    )


# R-compatible alias
GenDataPP = gen_dataPP
