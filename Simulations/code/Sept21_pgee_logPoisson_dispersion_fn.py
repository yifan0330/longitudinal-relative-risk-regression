"""September 2021 penalized log-Poisson GEE variant."""
from __future__ import annotations

try:
    from .gee_logPoisson_dispersion_fn import _gee_core
except ImportError:
    from gee_logPoisson_dispersion_fn import _gee_core


def gee_penalty_run(y, X, n_subj, n_visits, covariance="Independence", tol=1e-4, max_iter=10, phi_est=True):
    return _gee_core(y, X, n_subj, n_visits, covariance, tol, max_iter, phi_est, penalty="ik")
