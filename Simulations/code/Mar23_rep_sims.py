#!/usr/bin/env python3
"""Run March 2023 simulation repetitions, ported from Mar23_rep_sims.R."""
from __future__ import annotations

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
from simulation_helpers import (
    ModelResults,
    model_matrix,
    parse_simulation_args,
    run_model,
    save_pickle,
)
from source_simdata import gen_dataPP

GEEDIR = str(PROJECT_ROOT / 'PGEE_Mondol')


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    config = parse_simulation_args(argv)

    def mat():
        return np.zeros((config.n_sim, config.p))

    gee = ModelResults("geePK", config.n_sim, config.p, include_model_se=True)
    pgee = ModelResults("PgeePK", config.n_sim, config.p, include_model_se=False)
    coefs_ORpgee = mat()
    SEs_ORpgee = mat()
    SEs_model_ratio_ORpgee = mat()
    alpha_ORpgee = np.zeros(config.n_sim)
    phi_ORpgee = np.zeros(config.n_sim)
    eta_ORpgee = np.zeros(config.n_sim)
    iter_ORpgee = np.zeros(config.n_sim)
    conv_ORpgee = np.zeros(config.n_sim, dtype=bool)
    tabdat = []

    for i in range(1, config.n_sim + 1):
        datt = gen_dataPP(
            beta=config.true_beta,
            nc=config.nc,
            cl_size=config.cl_size,
            p=config.prop,
            rho=config.rho,
            rng=np.random.default_rng(i),
        )
        tabdat.append(pd.crosstab(datt["X1i"], datt["yij"]))
        X = model_matrix(datt)
        y = datt["yij"].to_numpy(float)
        gee.record(
            i - 1,
            run_model(
                "GEE",
                i,
                gee_dispersion_run,
                y=y,
                X=X,
                n_subj=config.nc,
                n_visits=config.cl_size,
                covariance="Exchangeable",
                max_iter=25,
                phi_est=True,
            ),
            X,
        )
        pgee.record(
            i - 1,
            run_model(
                "PGEE",
                i,
                gee_penalty_run,
                y=y,
                X=X,
                n_subj=config.nc,
                n_visits=config.cl_size,
                covariance="Exchangeable",
                max_iter=25,
                phi_est=True,
            ),
            X,
        )
        ORpgee = run_model(
            "OR-PGEE",
            i,
            geefirth,
            y=y,
            x=X[:, 1:3],
            id=np.repeat(np.arange(1, config.nc + 1), config.cl_size),
            ar=False,
        )
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

    save_pickle(
        config.output_path,
        {
            **gee.as_dict(),
            **pgee.as_dict(),
            "coefs_ORpgee": coefs_ORpgee,
            "SEs_ORpgee": SEs_ORpgee,
            "SEs_model_ratio_ORpgee": SEs_model_ratio_ORpgee,
            "alpha_ORpgee": alpha_ORpgee,
            "phi_ORpgee": phi_ORpgee,
            "eta_ORpgee": eta_ORpgee,
            "iter_ORpgee": iter_ORpgee,
            "conv_ORpgee": conv_ORpgee,
            "tabdat": tabdat,
            "true_beta": config.true_beta,
        },
    )
    fig = None
    try:
        p0 = np.exp(coefs_ORpgee[:, 0]) / (1 + np.exp(coefs_ORpgee[:, 0]))
        or_b = np.exp(coefs_ORpgee[:, 1])
        _ = np.log(or_b / ((1 - p0) + p0 * or_b))
        pdf_path = Path(GEEDIR) / "test.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(coefs_ORpgee[:, 1], pgee.coefs[:, 1])
        ax.axline((0, 0), slope=1)
        fig.savefig(pdf_path)
    except Exception as exc:
        print(f"Diagnostic plot failed: {exc!r}", file=sys.stderr)
    finally:
        if fig is not None:
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
