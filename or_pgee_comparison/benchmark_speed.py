#!/usr/bin/env python3
"""Benchmark validated parallel IRLS and exploratory non-production speed paths."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from .config import BASE_SCENARIO, RESULTS_DIR, Scenario
    from .data_generation import generate_dataset, generate_standard_datasets_broadcast
    from .methods import FIT_ENGINE_IRLS, FIT_ENGINE_LBFGS_OR_GEE
    from .run_simulation import run_scenarios
except ImportError:
    from config import BASE_SCENARIO, RESULTS_DIR, Scenario
    from data_generation import generate_dataset, generate_standard_datasets_broadcast
    from methods import FIT_ENGINE_IRLS, FIT_ENGINE_LBFGS_OR_GEE
    from run_simulation import run_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "benchmark")
    parser.add_argument(
        "--include-lbfgs",
        action="store_true",
        help="Also benchmark exploratory L-BFGS-B for OR-GEE only; not for manuscript results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replications < 1:
        raise SystemExit("--replications must be positive.")
    workers = max(1, int(args.workers))
    scenario = replace(BASE_SCENARIO, n_replications=args.replications)

    records: list[dict[str, object]] = []
    serial_seconds = _record_timing(
        records,
        label="serial_irls",
        scenario=scenario,
        workers=1,
        task=lambda: run_scenarios((scenario,), workers=1, fit_engine=FIT_ENGINE_IRLS),
    )
    if workers > 1:
        _record_timing(
            records,
            label="parallel_irls",
            scenario=scenario,
            workers=workers,
            task=lambda: run_scenarios(
                (scenario,),
                workers=workers,
                chunksize=max(1, args.replications // (workers * 4)),
                fit_engine=FIT_ENGINE_IRLS,
            ),
            baseline_seconds=serial_seconds,
        )
    if args.include_lbfgs:
        _record_timing(
            records,
            label="exploratory_parallel_lbfgs_or_gee",
            scenario=scenario,
            workers=workers,
            task=lambda: run_scenarios(
                (scenario,),
                workers=workers,
                chunksize=max(1, args.replications // (workers * 4)),
                fit_engine=FIT_ENGINE_LBFGS_OR_GEE,
            ),
            baseline_seconds=serial_seconds,
        )

    loop_generation_seconds = _record_timing(
        records,
        label="loop_data_generation_only",
        scenario=scenario,
        workers=1,
        task=lambda: [generate_dataset(scenario, rep) for rep in range(1, args.replications + 1)],
    )
    _record_timing(
        records,
        label="exploratory_broadcast_data_generation_only",
        scenario=scenario,
        workers=1,
        task=lambda: generate_standard_datasets_broadcast(
            scenario,
            range(1, args.replications + 1),
        ),
        baseline_seconds=loop_generation_seconds,
    )

    output = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "speed_comparison.csv"
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {output_path}")
    return 0


def _record_timing(
    records: list[dict[str, object]],
    *,
    label: str,
    scenario: Scenario,
    workers: int,
    task: Callable[[], object],
    baseline_seconds: float | None = None,
) -> float:
    start = time.perf_counter()
    result = task()
    elapsed = time.perf_counter() - start
    rows = _result_rows(result)
    records.append(
        {
            "task": label,
            "scenario": scenario.name,
            "replications": scenario.n_replications,
            "workers": workers,
            "elapsed_seconds": elapsed,
            "rows": rows,
            "validated_for_manuscript": label in {"serial_irls", "parallel_irls"},
            "speedup_vs_baseline": (
                baseline_seconds / elapsed if baseline_seconds is not None and elapsed > 0 else 1.0
            ),
        }
    )
    return elapsed


def _result_rows(result: object) -> int:
    if isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):
        return int(result[0].shape[0])
    if isinstance(result, list):
        return len(result)
    data = getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
