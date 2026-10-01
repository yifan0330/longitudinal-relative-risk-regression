#!/usr/bin/env python3
"""Run simulation repetitions, ported from rep_sims.R."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gee_logPoisson_dispersion_fn import gee_dispersion_run
from Sept21_pgee_logPoisson_dispersion_fn import gee_penalty_run
from simulation_helpers import (
    ModelResults,
    model_matrix,
    parse_simulation_args,
    run_model,
    save_pickle,
)
from source_simdata import gen_dataPP


def _table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["X1i"], df["yij"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    config = parse_simulation_args(argv)
    gee = ModelResults("geePK", config.n_sim, config.p, include_model_se=True)
    pgee = ModelResults("PgeePK", config.n_sim, config.p, include_model_se=False)
    tabdat = []

    for i in range(1, config.n_sim + 1):
        rng = np.random.default_rng(i)
        datt = gen_dataPP(
            beta=config.true_beta,
            nc=config.nc,
            cl_size=config.cl_size,
            p=config.prop,
            rho=config.rho,
            rng=rng,
        )
        tabdat.append(_table(datt))
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
                max_iter=1000,
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
                max_iter=1000,
                phi_est=True,
            ),
            X,
        )
        if i % 100 == 0:
            print(i)

    save_pickle(
        config.output_path,
        {
            **gee.as_dict(),
            **pgee.as_dict(),
            "tabdat": tabdat,
            "true_beta": config.true_beta,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
