"""Generate Wald p-value PP-plots for all simulation scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import RESULTS_DIR
from .methods import METHOD_OR_GEE, METHOD_OR_PGEE, METHOD_ORDER, METHOD_RR_GEE, METHOD_RR_PGEE, OR_METHODS


DEFAULT_REPLICATIONS = RESULTS_DIR / "full" / "replications.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "full" / "pp_plots"
DEFAULT_TABLE4 = RESULTS_DIR / "full" / "table4" / "table4_variance_coverage.csv"
METHOD_LABELS = {
    "RR-GEE": "RR-GEE",
    "RR-PGEE": "RR-PGEE",
    "OR-GEE-ZY": "OR-GEE",
    "OR-PGEE-ZY": "OR-PGEE",
}
METHOD_MARKERS = {
    "RR-GEE": "o",
    "RR-PGEE": "s",
    "OR-GEE-ZY": "^",
    "OR-PGEE-ZY": "D",
}
REQUIRED_COLUMNS = {
    "scenario",
    "replication",
    "method",
    "true_beta_b",
    "true_rr",
    "log_effect",
    "se_log_effect",
    "p0_hat",
    "converged",
    "finite",
}


def _or_null_log_effect(true_rr: pd.Series, p0_hat: pd.Series) -> pd.Series:
    """Invert the Zhang-Yu transformation at the simulated true RR."""
    denominator = 1.0 - true_rr * p0_hat
    valid = denominator > 0.0
    null_log_effect = pd.Series(np.nan, index=true_rr.index, dtype=float)
    null_or = true_rr[valid] * (1.0 - p0_hat[valid]) / denominator[valid]
    positive = null_or > 0.0
    null_log_effect.loc[null_or.index[positive]] = np.log(null_or[positive])
    return null_log_effect


def wald_p_values(replications_path: str | Path) -> pd.DataFrame:
    """Return valid two-sided Wald p-values around each simulated true effect."""
    replications_path = Path(replications_path)
    replications = pd.read_csv(replications_path)
    missing = REQUIRED_COLUMNS.difference(replications.columns)
    if missing:
        raise ValueError(f"Missing required columns in {replications_path}: {sorted(missing)}")

    numeric_columns = [
        "true_beta_b",
        "true_rr",
        "log_effect",
        "se_log_effect",
        "p0_hat",
    ]
    replications[numeric_columns] = replications[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    replications["null_log_effect"] = replications["true_beta_b"]
    is_or_method = replications["method"].isin(OR_METHODS)
    replications.loc[is_or_method, "null_log_effect"] = _or_null_log_effect(
        replications.loc[is_or_method, "true_rr"],
        replications.loc[is_or_method, "p0_hat"],
    )

    valid = (
        replications["converged"].eq(True)
        & replications["finite"].eq(True)
        & replications["se_log_effect"].gt(0.0)
        & np.isfinite(
            replications[["log_effect", "se_log_effect", "null_log_effect"]]
        ).all(axis=1)
    )
    valid_methods = (
        replications.loc[
            valid & replications["method"].isin(METHOD_ORDER),
            ["scenario", "replication", "method"],
        ]
        .drop_duplicates()
        .groupby(["scenario", "replication"])["method"]
        .agg(frozenset)
    )
    expected_methods = frozenset(METHOD_ORDER)
    common_groups = valid_methods[valid_methods.eq(expected_methods)].index
    group_index = pd.MultiIndex.from_frame(replications[["scenario", "replication"]])
    common_valid = (
        valid
        & replications["method"].isin(METHOD_ORDER)
        & group_index.isin(common_groups)
    )
    p_values = replications.loc[
        common_valid,
        ["scenario", "method", "log_effect", "se_log_effect", "null_log_effect"],
    ].copy()
    z_scores = (
        p_values["log_effect"] - p_values["null_log_effect"]
    ) / p_values["se_log_effect"]
    p_values["p_value"] = 2.0 * stats.norm.sf(np.abs(z_scores))
    return p_values[["scenario", "method", "p_value"]]


def table4_statistics(table4_path: str | Path) -> pd.DataFrame:
    """Load coverage probability and mean CI width annotations from Table 4."""
    table4_path = Path(table4_path)
    table4 = pd.read_csv(table4_path)
    required = {"scenario", "method", "coverage", "mean_ci_width_rr"}
    missing = required.difference(table4.columns)
    if missing:
        raise ValueError(f"Missing required columns in {table4_path}: {sorted(missing)}")
    if table4.duplicated(["scenario", "method"]).any():
        raise ValueError(f"Duplicate scenario/method rows found in {table4_path}")
    table4[["coverage", "mean_ci_width_rr"]] = table4[
        ["coverage", "mean_ci_width_rr"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(table4[["coverage", "mean_ci_width_rr"]]).all(axis=None):
        raise ValueError(f"Non-finite coverage or mean CI width found in {table4_path}")
    return table4.set_index(["scenario", "method"])


def pp_coordinates(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return expected and observed ordered probabilities on a -log10 scale."""
    observed = np.sort(np.asarray(p_values, dtype=float))
    if observed.size == 0:
        raise ValueError("At least one p-value is required for a PP-plot")
    expected = (np.arange(1, observed.size + 1) - 0.5) / observed.size
    return -np.log10(expected), -np.log10(observed)


def nominal_pp_band(
    n_values: int,
    *,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pointwise beta order-statistic band under Uniform(0, 1)."""
    if n_values < 1:
        raise ValueError("At least one p-value is required for a PP-plot")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    order = np.arange(1, n_values + 1)
    expected_p = (order - 0.5) / n_values
    alpha = 1.0 - confidence_level
    lower_p = stats.beta.ppf(alpha / 2.0, order, n_values + 1 - order)
    upper_p = stats.beta.ppf(1.0 - alpha / 2.0, order, n_values + 1 - order)
    return (
        -np.log10(expected_p),
        -np.log10(upper_p),
        -np.log10(lower_p),
    )


def save_pp_plot(
    p_values: pd.DataFrame,
    table4: pd.DataFrame,
    scenario: str,
    output_path: str | Path,
    *,
    confidence_level: float,
) -> None:
    """Save one four-method PP-plot with a nominal beta confidence band."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_values = p_values[p_values["scenario"] == scenario]
    if scenario_values.empty:
        raise ValueError(f"No valid p-values found for scenario={scenario!r}")

    plot_lines: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for method in METHOD_ORDER:
        method_values = scenario_values.loc[
            scenario_values["method"] == method, "p_value"
        ].to_numpy()
        if method_values.size == 0:
            continue
        try:
            method_statistics = table4.loc[(scenario, method)]
        except KeyError as exc:
            raise ValueError(
                f"No Table 4 statistics found for scenario={scenario!r}, method={method!r}"
            ) from exc
        expected, observed = pp_coordinates(method_values)
        plot_lines.append(
            (
                expected,
                observed,
                METHOD_MARKERS[method],
                (
                    f"{METHOD_LABELS[method]} "
                    f"(n={method_values.size}, CP={method_statistics['coverage']:.3f}, "
                    f"mean CI width={method_statistics['mean_ci_width_rr']:.2f})"
                ),
            )
        )

    if not plot_lines:
        raise ValueError(f"No valid method p-values found for scenario={scenario!r}")
    n_values = len(plot_lines[0][0])
    expected_band, lower_band, upper_band = nominal_pp_band(
        n_values,
        confidence_level=confidence_level,
    )
    axis_max = 3.25
    raw_band_x = expected_band[::-1]
    raw_band_lower = lower_band[::-1]
    raw_band_upper = upper_band[::-1]
    upper_endpoint = axis_max
    upper_axis_crossings = np.flatnonzero(
        (raw_band_x <= axis_max) & (raw_band_upper >= axis_max)
    )
    if upper_axis_crossings.size:
        crossing_index = int(upper_axis_crossings[0])
        if crossing_index > 0:
            upper_endpoint = float(
                np.interp(
                    axis_max,
                    raw_band_upper[crossing_index - 1 : crossing_index + 1],
                    raw_band_x[crossing_index - 1 : crossing_index + 1],
                )
            )
    visible_band = raw_band_x < upper_endpoint
    band_x = np.append(raw_band_x[visible_band], [upper_endpoint, axis_max])
    band_lower = np.append(
        raw_band_lower[visible_band],
        [
            np.interp(upper_endpoint, raw_band_x, raw_band_lower),
            np.interp(axis_max, raw_band_x, raw_band_lower),
        ],
    )
    band_upper = np.append(raw_band_upper[visible_band], [axis_max, axis_max])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill_between(
        band_x,
        band_lower,
        band_upper,
        color="0.85",
        alpha=0.8,
        linewidth=0.0,
        label=f"Nominal {confidence_level:.0%} beta band",
    )
    ax.plot(
        [0, axis_max],
        [0, axis_max],
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    for expected, observed, marker, label in plot_lines:
        ax.scatter(
            expected[::-1],
            observed[::-1],
            s=14,
            alpha=0.7,
            marker=marker,
            linewidths=0.0,
            label=label,
        )

    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(scenario)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PP-plots of simulation Wald p-values.",
    )
    parser.add_argument("--replications", type=Path, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--table4", type=Path, default=DEFAULT_TABLE4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "png"),
        choices=("pdf", "png", "svg"),
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p_values = wald_p_values(args.replications)
    table4 = table4_statistics(args.table4)
    scenarios = pd.unique(p_values["scenario"])
    for scenario in scenarios:
        for file_format in args.formats:
            save_pp_plot(
                p_values,
                table4,
                str(scenario),
                args.output_dir / f"pp_{scenario}.{file_format}",
                confidence_level=args.confidence_level,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
