"""Generate paired GEE vs PGEE scatter plots for Figure 1-style diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - density colouring is optional
    stats = None

from .config import RESULTS_DIR
from .methods import METHOD_OR_GEE, METHOD_OR_PGEE, METHOD_RR_GEE, METHOD_RR_PGEE


DEFAULT_REPLICATIONS = RESULTS_DIR / "full" / "replications.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "full" / "figure1"
TRUE_BETA_B = 1.6
AXIS_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 13


@dataclass(frozen=True)
class PairConfig:
    """Labels and method identifiers for one GEE/PGEE comparison pair."""

    gee_method: str
    pgee_method: str
    gee_label: str
    pgee_label: str
    stem_prefix: str
    boundary_method_label: str


PAIR_CONFIGS = {
    "rr": PairConfig(
        gee_method=METHOD_RR_GEE,
        pgee_method=METHOD_RR_PGEE,
        gee_label="RR-GEE",
        pgee_label="RR-PGEE",
        stem_prefix="figure1_rr",
        boundary_method_label="RR-GEE",
    ),
    "or": PairConfig(
        gee_method=METHOD_OR_GEE,
        pgee_method=METHOD_OR_PGEE,
        gee_label="OR-GEE",
        pgee_label="OR-PGEE",
        stem_prefix="figure1_or",
        boundary_method_label="OR-GEE",
    ),
}


@dataclass(frozen=True)
class MetricConfig:
    """Plot metadata for one paired diagnostic metric."""

    column_suffix: str
    label: str
    stem_suffix: str
    nonnegative: bool = False


METRIC_CONFIGS = {
    "z": MetricConfig(
        column_suffix="z",
        label=r"$\hat{z}_b$",
        stem_suffix="zscore_scatter",
    ),
    "beta": MetricConfig(
        column_suffix="beta",
        label=r"$\hat{\beta}_b$",
        stem_suffix="beta_scatter",
    ),
    "se": MetricConfig(
        column_suffix="se",
        label=r"$\mathrm{SE}(\hat{\beta}_b)$",
        stem_suffix="se_scatter",
        nonnegative=True,
    ),
}


def figure1_data(
    replications_path: str | Path,
    *,
    scenario: str,
    pair: str,
    boundary_warnings: bool,
) -> pd.DataFrame:
    """Return paired GEE/PGEE beta, SE, and z-statistics for one scenario."""
    pair_config = PAIR_CONFIGS[pair]
    replications_path = Path(replications_path)
    replications = pd.read_csv(replications_path)
    required = {
        "scenario",
        "replication",
        "method",
        "log_effect",
        "se_log_effect",
        "bec_count",
        "converged",
        "finite",
    }
    missing = required.difference(replications.columns)
    if missing:
        raise ValueError(f"Missing required columns in {replications_path}: {sorted(missing)}")

    pair_rows = replications[
        (replications["scenario"] == scenario)
        & replications["method"].isin([pair_config.gee_method, pair_config.pgee_method])
    ].copy()
    if pair_rows.empty:
        raise ValueError(
            f"No {pair_config.gee_label}/{pair_config.pgee_label} rows found "
            f"for scenario={scenario!r}"
        )

    pair_rows["log_effect"] = pd.to_numeric(pair_rows["log_effect"], errors="coerce")
    pair_rows["se_log_effect"] = pd.to_numeric(pair_rows["se_log_effect"], errors="coerce")
    pair_rows["z_statistic"] = pair_rows["log_effect"] / pair_rows["se_log_effect"]
    wide = pair_rows.pivot(
        index="replication",
        columns="method",
        values=[
            "log_effect",
            "se_log_effect",
            "z_statistic",
            "bec_count",
            "converged",
            "finite",
        ],
    )
    wide.columns = [f"{value}_{method}" for value, method in wide.columns]
    wide = wide.reset_index()

    paired = wide.rename(
        columns={
            f"log_effect_{pair_config.gee_method}": "gee_beta",
            f"log_effect_{pair_config.pgee_method}": "pgee_beta",
            f"se_log_effect_{pair_config.gee_method}": "gee_se",
            f"se_log_effect_{pair_config.pgee_method}": "pgee_se",
            f"z_statistic_{pair_config.gee_method}": "gee_z",
            f"z_statistic_{pair_config.pgee_method}": "pgee_z",
            f"bec_count_{pair_config.gee_method}": "gee_bec",
            f"converged_{pair_config.gee_method}": "gee_converged",
            f"converged_{pair_config.pgee_method}": "pgee_converged",
            f"finite_{pair_config.gee_method}": "gee_finite",
            f"finite_{pair_config.pgee_method}": "pgee_finite",
        }
    )
    value_columns = [
        "gee_beta",
        "pgee_beta",
        "gee_se",
        "pgee_se",
        "gee_z",
        "pgee_z",
    ]
    paired = paired[
        paired["gee_converged"].eq(True)
        & paired["pgee_converged"].eq(True)
        & paired["gee_finite"].eq(True)
        & paired["pgee_finite"].eq(True)
        & np.isfinite(paired[value_columns].astype(float)).all(axis=1)
    ]
    if boundary_warnings:
        paired = paired[pd.to_numeric(paired["gee_bec"], errors="coerce") > 10]
    if paired.empty:
        subset = (
            f"paired converged finite {pair_config.boundary_method_label} BEC > 10"
            if boundary_warnings
            else "paired converged finite"
        )
        raise ValueError(
            f"No {subset} {pair_config.gee_label}/{pair_config.pgee_label} "
            f"diagnostics found for scenario={scenario!r}"
        )

    return paired[["replication", *value_columns, "gee_bec"]].copy()


def _default_axis_limits(figure_data: pd.DataFrame, config: MetricConfig) -> tuple[float, float]:
    values = figure_data[
        [f"gee_{config.column_suffix}", f"pgee_{config.column_suffix}"]
    ].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values available for axis limits")
    lower = float(values.min())
    upper = float(values.max())
    if config.nonnegative:
        lower = max(0.0, lower)
    else:
        lower = min(0.0, lower)
        upper = max(0.0, upper)
    span = upper - lower
    padding = 0.05 * span if span > 0 else max(0.1, abs(upper) * 0.05)
    lower -= padding
    upper += padding
    if config.nonnegative:
        lower = max(0.0, lower)
    return lower, upper


def save_scatter(
    figure_data: pd.DataFrame,
    output_path: str | Path,
    *,
    pair: str,
    metric: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    boundary_warnings: bool,
) -> None:
    """Save one paired GEE-vs-PGEE diagnostic scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_config = PAIR_CONFIGS[pair]
    config = METRIC_CONFIGS[metric]
    default_limits = _default_axis_limits(figure_data, config)
    xlim = default_limits if xlim is None else xlim
    ylim = default_limits if ylim is None else ylim

    x = figure_data[f"gee_{config.column_suffix}"].to_numpy(dtype=float)
    y = figure_data[f"pgee_{config.column_suffix}"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    colours = None
    if stats is not None and len(x) > 2:
        try:
            colours = stats.gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        except Exception:
            colours = None
    if colours is None:
        ax.scatter(x, y, alpha=0.35, s=16)
    else:
        ax.scatter(x, y, c=colours, cmap="viridis", alpha=0.35, s=16)

    ax.axline(
        (0, 0),
        slope=1,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$y=x$",
        zorder=1,
    )
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.axvline(0, color="0.5", linewidth=0.5)
    if metric == "beta":
        ax.axhline(
            TRUE_BETA_B,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label=rf"$y={TRUE_BETA_B:g}$",
            zorder=1,
        )
        ax.axvline(
            TRUE_BETA_B,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label=rf"$x={TRUE_BETA_B:g}$",
            zorder=1,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        f"{pair_config.gee_label} {config.label}", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax.set_ylabel(
        f"{pair_config.pgee_label} {config.label}", fontsize=AXIS_LABEL_FONTSIZE
    )
    if boundary_warnings:
        ax.set_title(
            rf"{pair_config.boundary_method_label} boundary-warning replications "
            r"($\mathrm{BEC} > 10$)"
        )
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse Figure 1 plotting options."""
    parser = argparse.ArgumentParser(
        description="Generate paired GEE vs PGEE diagnostic scatter plots.",
    )
    parser.add_argument("--replications", type=Path, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default="base")
    parser.add_argument(
        "--pair",
        choices=tuple(PAIR_CONFIGS),
        default="rr",
        help="Method pair to plot. Default: rr.",
    )
    parser.add_argument(
        "--metric",
        choices=("z", "beta", "se", "all"),
        default="z",
        help="Diagnostic to plot. Default: z.",
    )
    parser.add_argument(
        "--boundary-warnings",
        action="store_true",
        help="Restrict to paired converged finite replications where the GEE method has BEC > 10.",
    )
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument(
        "--stem",
        help="Output filename stem. With --metric all, the metric name is appended.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "png"),
        choices=("pdf", "png", "svg"),
    )
    return parser.parse_args()


def main() -> int:
    """Generate the requested paired diagnostic scatter plots."""
    args = parse_args()
    figure_data = figure1_data(
        args.replications,
        scenario=args.scenario,
        pair=args.pair,
        boundary_warnings=args.boundary_warnings,
    )
    metrics = tuple(METRIC_CONFIGS) if args.metric == "all" else (args.metric,)
    for metric in metrics:
        if args.stem is None:
            stem = f"{PAIR_CONFIGS[args.pair].stem_prefix}_{METRIC_CONFIGS[metric].stem_suffix}"
        elif len(metrics) > 1:
            stem = f"{args.stem}_{metric}"
        else:
            stem = args.stem
        for file_format in args.formats:
            save_scatter(
                figure_data,
                args.output_dir / f"{stem}.{file_format}",
                pair=args.pair,
                metric=metric,
                xlim=None if args.xlim is None else tuple(args.xlim),
                ylim=None if args.ylim is None else tuple(args.ylim),
                boundary_warnings=args.boundary_warnings,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
