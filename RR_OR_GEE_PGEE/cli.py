"""Unified command-line interface for simulations and manuscript outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


COMMAND_MODULES = {
    "simulate": "RR_OR_GEE_PGEE.run_simulation",
    "figure1": "RR_OR_GEE_PGEE.plot_figure1",
    "figure-b1": "RR_OR_GEE_PGEE.plot_figureB1",
    "pp-plots": "RR_OR_GEE_PGEE.plot_pp",
    "benchmark": "RR_OR_GEE_PGEE.benchmark_speed",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="python -m RR_OR_GEE_PGEE",
        description="Run the OR-PGEE comparison study and reproduce its outputs.",
    )
    parser.add_argument(
        "command",
        choices=(*COMMAND_MODULES, "figures"),
        help="Study stage to run; remaining arguments are forwarded to that stage.",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _run_module(module: str, arguments: Sequence[str] = ()) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one study stage or regenerate every figure."""
    args = build_parser().parse_args(argv)
    if args.command != "figures":
        _run_module(COMMAND_MODULES[args.command], args.arguments)
        return 0

    if args.arguments:
        raise SystemExit("The 'figures' workflow does not accept additional arguments.")
    for pair in ("rr", "or"):
        _run_module(
            "RR_OR_GEE_PGEE.plot_figure1",
            ("--pair", pair, "--metric", "all"),
        )
    _run_module("RR_OR_GEE_PGEE.plot_figureB1")
    _run_module("RR_OR_GEE_PGEE.plot_pp")
    return 0
