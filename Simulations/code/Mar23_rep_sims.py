#!/usr/bin/env python3
"""Run March 2023 simulation repetitions, ported from Mar23_rep_sims.R."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PGEE_DIR = ROOT / "PGEE_Mondol"
for path in (SCRIPT_DIR, PGEE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gee_logPoisson_dispersion_fn import gee_dispersion_run
from Mar23_PGEE_source import geefirth
from Sept21_pgee_logPoisson_dispersion_fn import gee_penalty_run
from source_simdata import gen_dataPP

GEEDIR = str(PROJECT_ROOT / 'PGEE_Mondol')


def _model_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(df)), df["X1i"].to_numpy(float), df["obstime"].to_numpy(float)]
    )


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

    def mat():
        return np.zeros((n_sim, p))

    coefs_geePK = mat()
    SEs_geePK = mat()
    SEs_model_geePK = mat()
    SEs_model_ratio_geePK = mat()
    coefs_PgeePK = mat()
    SEs_PgeePK = mat()
    SEs_model_ratio_PgeePK = mat()
    coefs_ORpgee = mat()
    SEs_ORpgee = mat()
    SEs_model_ratio_ORpgee = mat()
    alpha_geePK = np.zeros(n_sim)
    phi_geePK = np.zeros(n_sim)
    eta_geePK = np.zeros(n_sim)
    iter_geePK = np.zeros(n_sim)
    conv_geePK = np.zeros(n_sim, dtype=bool)
    alpha_PgeePK = np.zeros(n_sim)
    phi_PgeePK = np.zeros(n_sim)
    eta_PgeePK = np.zeros(n_sim)
    iter_PgeePK = np.zeros(n_sim)
    conv_PgeePK = np.zeros(n_sim, dtype=bool)
    alpha_ORpgee = np.zeros(n_sim)
    phi_ORpgee = np.zeros(n_sim)
    eta_ORpgee = np.zeros(n_sim)
    iter_ORpgee = np.zeros(n_sim)
    conv_ORpgee = np.zeros(n_sim, dtype=bool)
    tabdat = []

    for i in range(1, n_sim + 1):
        datt = gen_dataPP(
            beta=true_beta,
            nc=nc,
            cl_size=cl_size,
            p=prop,
            rho=rho,
            rng=np.random.default_rng(i),
        )
        tabdat.append(pd.crosstab(datt["X1i"], datt["yij"]))
        X = _model_matrix(datt)
        y = datt["yij"].to_numpy(float)
        try:
            gee_PK = gee_dispersion_run(
                y, X, nc, cl_size, covariance="Exchangeable", max_iter=25, phi_est=True
            )
        except Exception:
            gee_PK = None
        if gee_PK is None:
            coefs_geePK[i - 1, :] = np.nan
            SEs_geePK[i - 1, :] = np.nan
            SEs_model_geePK[i - 1, :] = np.nan
            SEs_model_ratio_geePK[i - 1, :] = np.nan
            alpha_geePK[i - 1] = phi_geePK[i - 1] = eta_geePK[i - 1] = iter_geePK[
                i - 1
            ] = np.nan
        else:
            coefs_geePK[i - 1, :] = gee_PK["beta"]
            SEs_geePK[i - 1, :] = gee_PK["beta_se_sandwich"]
            SEs_model_geePK[i - 1, :] = gee_PK["beta_se_model"]
            SEs_model_ratio_geePK[i - 1, :] = (
                gee_PK["beta_se_model"] / gee_PK["beta_se_model_trace"][0]
            )
            alpha_geePK[i - 1] = gee_PK["alpha"]
            phi_geePK[i - 1] = gee_PK["phi"]
            eta_geePK[i - 1] = np.sum(X @ gee_PK["beta"] > 0)
            iter_geePK[i - 1] = gee_PK["iterations"]
            conv_geePK[i - 1] = gee_PK["conv"]
        try:
            Pgee_PK = gee_penalty_run(
                y, X, nc, cl_size, covariance="Exchangeable", max_iter=25, phi_est=True
            )
        except Exception:
            Pgee_PK = None
        if Pgee_PK is None:
            coefs_PgeePK[i - 1, :] = np.nan
            SEs_PgeePK[i - 1, :] = np.nan
            SEs_model_ratio_PgeePK[i - 1, :] = np.nan
            alpha_PgeePK[i - 1] = phi_PgeePK[i - 1] = eta_PgeePK[i - 1] = iter_PgeePK[
                i - 1
            ] = np.nan
        else:
            coefs_PgeePK[i - 1, :] = Pgee_PK["beta"]
            SEs_PgeePK[i - 1, :] = Pgee_PK["beta_se_sandwich"]
            SEs_model_ratio_PgeePK[i - 1, :] = (
                Pgee_PK["beta_se_model"] / Pgee_PK["beta_se_model_trace"][0]
            )
            alpha_PgeePK[i - 1] = Pgee_PK["alpha"]
            phi_PgeePK[i - 1] = Pgee_PK["phi"]
            eta_PgeePK[i - 1] = np.sum(X @ Pgee_PK["beta"] > 0)
            iter_PgeePK[i - 1] = Pgee_PK["iterations"]
            conv_PgeePK[i - 1] = Pgee_PK["conv"]
        try:
            ORpgee = geefirth(
                y=y, x=X[:, 1:3], id=np.repeat(np.arange(1, nc + 1), cl_size), ar=False
            )
        except Exception:
            ORpgee = None
        if ORpgee is None:
            coefs_ORpgee[i - 1, :] = np.nan
            SEs_ORpgee[i - 1, :] = np.nan
            SEs_model_ratio_ORpgee[i - 1, :] = np.nan
            alpha_ORpgee[i - 1] = phi_ORpgee[i - 1] = eta_ORpgee[i - 1] = iter_ORpgee[
                i - 1
            ] = np.nan
        else:
            coefs_ORpgee[i - 1, :] = ORpgee[0]["coefficients"].to_numpy()
            SEs_ORpgee[i - 1, :] = ORpgee[0]["std.err"].to_numpy()
            SEs_model_ratio_ORpgee[i - 1, :] = ORpgee[5] / ORpgee[6][0, :]
            alpha_ORpgee[i - 1] = ORpgee[2]
            iter_ORpgee[i - 1] = ORpgee[3]
            phi_ORpgee[i - 1] = ORpgee[4]
            eta_ORpgee[i - 1] = np.sum(X @ coefs_ORpgee[i - 1, :] > 0)
            conv_ORpgee[i - 1] = True
        if i % 2 == 0:
            print(i)

    _save_exact(outputdir, locals() | {"true_beta": true_beta})
    try:
        p0 = np.exp(coefs_ORpgee[:, 0]) / (1 + np.exp(coefs_ORpgee[:, 0]))
        or_b = np.exp(coefs_ORpgee[:, 1])
        _ = np.log(or_b / ((1 - p0) + p0 * or_b))
        pdf_path = Path(GEEDIR) / "test.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.scatter(coefs_ORpgee[:, 1], coefs_PgeePK[:, 1])
        plt.axline((0, 0), slope=1)
        plt.savefig(pdf_path)
        plt.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
