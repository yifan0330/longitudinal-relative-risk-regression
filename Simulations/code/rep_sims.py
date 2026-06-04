#!/usr/bin/env python3
"""Run simulation repetitions, ported from rep_sims.R."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gee_logPoisson_dispersion_fn import gee_dispersion_run
from Sept21_pgee_logPoisson_dispersion_fn import gee_penalty_run
from source_simdata import gen_dataPP


def _model_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df["X1i"].to_numpy(float), df["obstime"].to_numpy(float)])


def _table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["X1i"], df["yij"])


def _save_exact(path: str, payload: dict) -> None:
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 10:
        raise SystemExit("Wrong number of arguments.")
    true_beta = np.asarray([float(argv[0]), float(argv[1]), float(argv[2])])
    nc = int(float(argv[3]))
    cl_size = int(float(argv[4]))
    prop = float(argv[5])
    rho = float(argv[6])
    n_sim = int(float(argv[7]))
    p = int(float(argv[8]))
    outputdir = argv[9]

    coefs_geePK = np.zeros((n_sim, p)); SEs_geePK = np.zeros((n_sim, p))
    SEs_model_geePK = np.zeros((n_sim, p)); SEs_model_ratio_geePK = np.zeros((n_sim, p))
    alpha_geePK = np.zeros(n_sim); phi_geePK = np.zeros(n_sim); eta_geePK = np.zeros(n_sim)
    iter_geePK = np.zeros(n_sim); conv_geePK = np.zeros(n_sim, dtype=bool)
    coefs_PgeePK = np.zeros((n_sim, p)); SEs_PgeePK = np.zeros((n_sim, p))
    SEs_model_ratio_PgeePK = np.zeros((n_sim, p))
    alpha_PgeePK = np.zeros(n_sim); phi_PgeePK = np.zeros(n_sim); eta_PgeePK = np.zeros(n_sim)
    iter_PgeePK = np.zeros(n_sim); conv_PgeePK = np.zeros(n_sim, dtype=bool)
    tabdat = []

    for i in range(1, n_sim + 1):
        rng = np.random.default_rng(i)
        datt = gen_dataPP(beta=true_beta, nc=nc, cl_size=cl_size, p=prop, rho=rho, rng=rng)
        tabdat.append(_table(datt))
        X = _model_matrix(datt)
        y = datt["yij"].to_numpy(float)

        try:
            gee_PK = gee_dispersion_run(y, X, nc, cl_size, covariance="Exchangeable", max_iter=1000, phi_est=True)
        except Exception:
            gee_PK = None
        if gee_PK is None:
            coefs_geePK[i - 1, :] = np.nan; SEs_geePK[i - 1, :] = np.nan; SEs_model_geePK[i - 1, :] = np.nan
            SEs_model_ratio_geePK[i - 1, :] = np.nan; alpha_geePK[i - 1] = np.nan; phi_geePK[i - 1] = np.nan
            eta_geePK[i - 1] = np.nan; iter_geePK[i - 1] = np.nan; conv_geePK[i - 1] = False
        else:
            coefs_geePK[i - 1, :] = gee_PK["beta"]; SEs_geePK[i - 1, :] = gee_PK["beta_se_sandwich"]
            SEs_model_geePK[i - 1, :] = gee_PK["beta_se_model"]
            first = gee_PK["beta_se_model_trace"][0]
            SEs_model_ratio_geePK[i - 1, :] = gee_PK["beta_se_model"] / first
            alpha_geePK[i - 1] = gee_PK["alpha"]; phi_geePK[i - 1] = gee_PK["phi"]
            eta_geePK[i - 1] = np.sum(X @ gee_PK["beta"] > 0)
            iter_geePK[i - 1] = gee_PK["iterations"]; conv_geePK[i - 1] = gee_PK["conv"]

        try:
            Pgee_PK = gee_penalty_run(y, X, nc, cl_size, covariance="Exchangeable", max_iter=1000, phi_est=True)
        except Exception:
            Pgee_PK = None
        if Pgee_PK is None:
            coefs_PgeePK[i - 1, :] = np.nan; SEs_PgeePK[i - 1, :] = np.nan; SEs_model_ratio_PgeePK[i - 1, :] = np.nan
            alpha_PgeePK[i - 1] = np.nan; phi_PgeePK[i - 1] = np.nan; eta_PgeePK[i - 1] = np.nan
            iter_PgeePK[i - 1] = np.nan; conv_PgeePK[i - 1] = False
        else:
            coefs_PgeePK[i - 1, :] = Pgee_PK["beta"]; SEs_PgeePK[i - 1, :] = Pgee_PK["beta_se_sandwich"]
            SEs_model_ratio_PgeePK[i - 1, :] = Pgee_PK["beta_se_model"] / Pgee_PK["beta_se_model_trace"][0]
            alpha_PgeePK[i - 1] = Pgee_PK["alpha"]; phi_PgeePK[i - 1] = Pgee_PK["phi"]
            eta_PgeePK[i - 1] = np.sum(X @ Pgee_PK["beta"] > 0)
            iter_PgeePK[i - 1] = Pgee_PK["iterations"]; conv_PgeePK[i - 1] = Pgee_PK["conv"]
        if i % 100 == 0:
            print(i)

    _save_exact(outputdir, locals() | {"true_beta": true_beta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
