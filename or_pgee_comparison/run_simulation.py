#!/usr/bin/env python3
"""Run OR-PGEE comparison simulations.

Experiment settings
-------------------
The base simulation setting is beta=(beta_1, beta_b, beta_c)=(-4, 1.6, 0.2),
N=50 subjects, T=4 visits, c=P(X1=1)=0.2, alpha=0.4, and R=1000
replications. The configured full experiment changes one parameter at a time
from this base setting and runs RR-GEE, RR-PGEE, OR-GEE-ZY, and OR-PGEE-ZY
on the same seeded datasets.

Run all configured smoke scenarios with R=100:

    uv run python -m or_pgee_comparison.run_simulation --mode smoke

Run all configured full scenarios with R=1000:

    uv run python -m or_pgee_comparison.run_simulation --mode full

Run the base setting while specifying every simulation variable explicitly:

    uv run python -m or_pgee_comparison.run_simulation \
      --mode full \
      --custom-scenario-name base_explicit \
      --beta-1 -4 \
      --beta-b 1.6 \
      --beta-c 0.2 \
      --n-subjects 50 \
      --n-visits 4 \
      --prop 0.2 \
    --rho 0.4 \
      --replications 1000

Run a one-parameter alteration by changing only that value, for example
beta_1=-3 while all other variables remain at their base values:

    uv run python -m or_pgee_comparison.run_simulation \
      --mode full \
      --custom-scenario-name beta1_minus3_explicit \
      --beta-1 -3 \
      --beta-b 1.6 \
      --beta-c 0.2 \
      --n-subjects 50 \
      --n-visits 4 \
      --prop 0.2 \
    --rho 0.4 \
      --replications 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import pandas as pd

from .config import (
    BASE_SCENARIO,
    MAX_ITER,
    RESULTS_DIR,
    SMOKE_R,
    Scenario,
    full_scenarios,
    smoke_scenarios,
)
from .coverage import add_interval_columns
from .data_generation import generate_dataset
from .methods import (
    FIT_ENGINE_IRLS,
    fit_all_methods,
)
from .postprocess import build_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--scenario", help="Run only one scenario by name.")
    parser.add_argument(
        "--custom-scenario-name",
        default="custom_explicit",
        help="Scenario name used when explicit simulation parameters are supplied.",
    )
    parser.add_argument("--beta-1", type=float, help="Intercept beta_1.")
    parser.add_argument("--beta-b", type=float, help="Binary exposure coefficient beta_b.")
    parser.add_argument("--beta-c", type=float, help="Time/covariate coefficient beta_c.")
    parser.add_argument("--n-subjects", type=int, help="Number of subjects N.")
    parser.add_argument("--n-visits", type=int, help="Number of visits per subject T.")
    parser.add_argument("--prop", type=float, help="Exposure prevalence c = P(X1 = 1).")
    parser.add_argument(
        "--rho",
        type=float,
        help="Exchangeable within-subject correlation alpha used in data generation.",
    )
    parser.add_argument("--replications", type=int, help="Number of simulation replications R.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for replication-level parallelism. Use 0 for all CPUs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Replication jobs submitted to each worker at a time.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Write replication/diagnostic CSVs without postprocessing manuscript tables.",
    )
    parser.add_argument(
        "--confounded",
        action="store_true",
        help="Use the shifted-X2 confounded data-generating scenario for explicit runs.",
    )
    parser.add_argument(
        "--confounding-shift",
        type=float,
        default=0.50,
        help="Shift strength for the full-run confounded scenario.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = _selected_scenarios(args)
    if args.scenario:
        scenarios = tuple(scenario for scenario in scenarios if scenario.name == args.scenario)
        if not scenarios:
            raise SystemExit(f"Unknown scenario for {args.mode} mode: {args.scenario}")

    output_dir = args.output_dir or (RESULTS_DIR / args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    replications, diagnostics_frame = run_scenarios(
        scenarios,
        workers=_resolve_workers(args.workers),
        chunksize=args.chunksize,
        fit_engine=FIT_ENGINE_IRLS,
    )
    replications = add_interval_columns(replications)
    replications.to_csv(output_dir / "replications.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "diagnostics.csv", index=False)
    tables = {} if args.skip_tables else build_tables(replications, output_dir)

    print(f"Wrote replication rows: {replications.shape}")
    print(f"Wrote diagnostics rows: {diagnostics_frame.shape}")
    for name, table in tables.items():
        print(f"{name}: {table.shape}")
    print(
        diagnostics_frame.groupby("scenario", as_index=False)
        .agg(
            mean_x1_x2_corr=("x1_x2_corr", "mean"),
            max_probability_clip_rate=("probability_clip_rate", "max"),
            max_x2_clip_rate=("x2_clip_rate", "max"),
        )
        .to_string(index=False)
    )
    print(
        replications.groupby(["scenario", "method"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            coverage_n=("coverage_eligible", "sum"),
            converged=("converged", "sum"),
        )
        .to_string(index=False)
    )
    return 0


def run_scenarios(
    scenarios: Iterable[Scenario],
    *,
    workers: int = 1,
    chunksize: int = 1,
    fit_engine: str = FIT_ENGINE_IRLS,
    max_iter: int = MAX_ITER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation scenarios and return replication and diagnostic frames."""
    scenario_tuple = tuple(scenarios)
    jobs = [
        (scenario, replication, fit_engine, max_iter)
        for scenario in scenario_tuple
        for replication in range(1, scenario.n_replications + 1)
    ]
    if not jobs:
        return pd.DataFrame(), pd.DataFrame()

    if workers <= 1:
        results = [_run_replication_job(job) for job in jobs]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(executor.map(_run_replication_job, jobs, chunksize=max(1, chunksize)))

    diagnostics = [diagnostic for diagnostic, _ in results]
    rows = [row for _, replication_rows in results for row in replication_rows]
    replications = _replication_frame(rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    if not diagnostics_frame.empty:
        diagnostics_frame = diagnostics_frame.reset_index(drop=True)
    return replications, diagnostics_frame


def _run_replication_job(
    job: tuple[Scenario, int, str, int]
) -> tuple[dict[str, float | int | str], list[dict[str, object]]]:
    scenario, replication, fit_engine, max_iter = job
    bundle = generate_dataset(scenario, replication)
    rows = [
        {
            "scenario": scenario.name,
            "scenario_type": scenario.scenario_type,
            "replication": replication,
            "method": fit.method,
            "true_beta_b": scenario.true_beta_b,
            "true_rr": scenario.true_rr,
            "log_effect": fit.log_effect,
            "se_log_effect": fit.se_log_effect,
            "rr_estimate": fit.rr_estimate,
            "p0_hat": fit.p0_hat,
            "converged": fit.converged,
            "finite": fit.finite,
            "iterations": fit.iterations,
            "alpha": fit.alpha,
            "phi": fit.phi,
            "bec_count": fit.bec_count,
            "failure_reason": fit.failure_reason,
        }
        for fit in fit_all_methods(
            bundle.data,
            scenario,
            max_iter=max_iter,
            fit_engine=fit_engine,
        )
    ]
    return bundle.diagnostics, rows


def _replication_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    replications = pd.DataFrame(rows)
    if replications.empty:
        return replications
    return replications.reset_index(drop=True)


def _resolve_workers(workers: int) -> int:
    if workers == 0:
        return os.cpu_count() or 1
    if workers < 0:
        raise SystemExit("--workers must be non-negative.")
    return workers


def _selected_scenarios(args: argparse.Namespace) -> tuple[Scenario, ...]:
    if _uses_explicit_parameters(args):
        if args.scenario:
            raise SystemExit("--scenario cannot be combined with explicit simulation parameters.")
        return (_custom_scenario(args),)
    scenario_factory = smoke_scenarios if args.mode == "smoke" else full_scenarios
    return scenario_factory(confounding_shift=args.confounding_shift)


def _uses_explicit_parameters(args: argparse.Namespace) -> bool:
    return args.confounded or any(
        getattr(args, name) is not None
        for name in (
            "beta_1",
            "beta_b",
            "beta_c",
            "n_subjects",
            "n_visits",
            "prop",
            "rho",
            "replications",
        )
    )


def _custom_scenario(args: argparse.Namespace) -> Scenario:
    beta = BASE_SCENARIO.beta.copy()
    if args.beta_1 is not None:
        beta[0] = args.beta_1
    if args.beta_b is not None:
        beta[1] = args.beta_b
    if args.beta_c is not None:
        beta[2] = args.beta_c
    return replace(
        BASE_SCENARIO,
        name=args.custom_scenario_name,
        beta=beta,
        n_subjects=args.n_subjects
        if args.n_subjects is not None
        else BASE_SCENARIO.n_subjects,
        n_visits=args.n_visits if args.n_visits is not None else BASE_SCENARIO.n_visits,
        prop=args.prop if args.prop is not None else BASE_SCENARIO.prop,
        rho=args.rho if args.rho is not None else BASE_SCENARIO.rho,
        n_replications=args.replications
        if args.replications is not None
        else (SMOKE_R if args.mode == "smoke" else BASE_SCENARIO.n_replications),
        confounded=args.confounded,
        confounding_shift=args.confounding_shift if args.confounded else 0.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
