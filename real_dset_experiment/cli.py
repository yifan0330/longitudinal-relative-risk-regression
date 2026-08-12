"""Command-line orchestration for all real-data manuscript outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleCommand:
    """A Python module and its command-line arguments."""

    module: str
    arguments: tuple[str, ...] = ()


FIGURE_COMMANDS = (
    ModuleCommand("real_dset_experiment.figure3.plot_ukb_empirical_maps"),
    *(
        ModuleCommand(
            "real_dset_experiment.figure4.plot_rr_pgee_significance_maps",
            ("--model", model, "--use-cache"),
        )
        for model in ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
    ),
    *(
        ModuleCommand(
            "real_dset_experiment.figure5.plot_rr_alpha_phi_maps",
            ("--model", model, "--use-cache"),
        )
        for model in ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
    ),
    ModuleCommand(
        "real_dset_experiment.figure6.plot_rr_pgee_relative_risk_maps",
        ("--models", "all", "--use-cache"),
    ),
    ModuleCommand(
        "real_dset_experiment.figure7.plot_age_relative_risk_comparison",
        ("--use-cache",),
    ),
    ModuleCommand("real_dset_experiment.figureB3.plot_figureB3"),
)

TABLE_COMMANDS = (
    ModuleCommand("real_dset_experiment.table5.calculate_ukb_characteristics"),
    ModuleCommand(
        "real_dset_experiment.table6.calculate_significant_coefficients",
        ("--use-cache",),
    ),
    ModuleCommand(
        "real_dset_experiment.table6.calculate_or_significant_coefficients",
        ("--use-cache",),
    ),
    ModuleCommand(
        "real_dset_experiment.table7.calculate_rr_relative_risk_by_incidence",
        ("--models", "all", "--use-cache"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m real_dset_experiment",
        description="Fit UKB models and reproduce every manuscript figure and table.",
    )
    parser.add_argument(
        "command",
        choices=("fit", "figures", "tables", "all"),
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "--rerun-models",
        action="store_true",
        help="Refit models instead of reusing maps in UKB/python_results.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Worker count used by the model-fitting stage.",
    )
    return parser


def _run_module(command_spec: ModuleCommand) -> None:
    command = [sys.executable, "-m", command_spec.module, *command_spec.arguments]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _fit_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    arguments = ["--models", "all"]
    if not args.rerun_models:
        arguments.append("--use-cache")
    if args.n_jobs is not None:
        if args.n_jobs <= 0:
            raise ValueError("--n-jobs must be positive")
        arguments.extend(("--n-jobs", str(args.n_jobs)))
    return tuple(arguments)


def _commands_with_cache_policy(
    commands: Sequence[ModuleCommand],
    *,
    rerun_models: bool,
) -> tuple[ModuleCommand, ...]:
    if not rerun_models:
        return tuple(commands)
    return tuple(
        ModuleCommand(
            command.module,
            tuple(argument for argument in command.arguments if argument != "--use-cache"),
        )
        for command in commands
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected reproducibility workflow."""
    args = build_parser().parse_args(argv)

    if args.command in {"fit", "all"}:
        _run_module(
            ModuleCommand(
                "real_dset_experiment.ukb_python_experiment",
                _fit_arguments(args),
            )
        )

    if args.command in {"figures", "all"}:
        for command in _commands_with_cache_policy(
            FIGURE_COMMANDS, rerun_models=args.rerun_models
        ):
            _run_module(command)

    if args.command in {"tables", "all"}:
        for command in _commands_with_cache_policy(
            TABLE_COMMANDS, rerun_models=args.rerun_models
        ):
            _run_module(command)

    return 0
