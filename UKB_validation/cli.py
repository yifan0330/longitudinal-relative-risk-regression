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
    ModuleCommand("UKB_validation.plot_figure3"),
    *(
        ModuleCommand(
            "UKB_validation.plot_figure4",
            ("--model", model, "--use-cache"),
        )
        for model in ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
    ),
    *(
        ModuleCommand(
            "UKB_validation.plot_figure5",
            ("--model", model, "--use-cache"),
        )
        for model in ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
    ),
    ModuleCommand(
        "UKB_validation.plot_figure6",
        ("--models", "all", "--use-cache"),
    ),
    ModuleCommand(
        "UKB_validation.plot_figure7",
        ("--use-cache",),
    ),
    ModuleCommand("UKB_validation.plot_figure_b3"),
)

TABLE_COMMANDS = (
    ModuleCommand("UKB_validation.table_5"),
    ModuleCommand(
        "UKB_validation.table6_rr",
        ("--use-cache",),
    ),
    ModuleCommand(
        "UKB_validation.table6_or",
        ("--use-cache",),
    ),
    ModuleCommand(
        "UKB_validation.table_7",
        ("--models", "all", "--use-cache"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m UKB_validation",
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
    """Run one reproduction module in a child Python process."""
    command = [sys.executable, "-m", command_spec.module, *command_spec.arguments]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _fit_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    """Translate workflow options into model-fitting command-line arguments."""
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
    """Apply the requested cache policy to figure and table commands."""
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
                "UKB_validation.ukb_python_experiment",
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
