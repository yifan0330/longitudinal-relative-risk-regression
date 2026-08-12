#!/usr/bin/env python3
"""Plot Figure B1 BEC histograms for the base simulation scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .config import DEFAULT_BETA, DEFAULT_N_SUBJECTS, DEFAULT_PROP, DEFAULT_RHO, RESULTS_DIR


DEFAULT_REPLICATIONS = RESULTS_DIR / "full" / "replications.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "full" / "figureB1" / "figureB1_bec_threshold.png"
METHODS = ("RR-GEE", "RR-PGEE")
METHOD_LABELS = {"RR-GEE": "RR-GEE", "RR-PGEE": "RR-PGEE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure B1: BEC histograms for RR-GEE and RR-PGEE."
    )
    parser.add_argument("--replications", type=Path, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.replications.exists():
        raise FileNotFoundError(f"Replication file not found: {args.replications}")
    if args.x_max <= 0 or args.threshold <= 0 or args.bin_width <= 0 or args.dpi <= 0:
        raise ValueError("--x-max, --threshold, --bin-width, and --dpi must be positive")


def load_bec_values(path: Path, scenario: str) -> dict[str, np.ndarray]:
    replications = pd.read_csv(path)
    required = {"scenario", "method", "bec_count"}
    missing = required.difference(replications.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    values: dict[str, np.ndarray] = {}
    for method in METHODS:
        rows = replications[(replications["scenario"] == scenario) & (replications["method"] == method)]
        if rows.empty:
            raise ValueError(f"No {method} rows found for scenario={scenario!r}")
        bec = pd.to_numeric(rows["bec_count"], errors="coerce").to_numpy(dtype=float)
        values[method] = bec[np.isfinite(bec)]
    return values


def add_strip(axis: plt.Axes, label: str) -> None:
    strip = Rectangle(
        (0.0, 1.005),
        1.0,
        0.085,
        transform=axis.transAxes,
        facecolor="0.94",
        edgecolor="0.15",
        linewidth=0.8,
        clip_on=False,
    )
    axis.add_patch(strip)
    axis.text(
        0.5,
        1.04,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        clip_on=False,
    )


def plot_figure(values: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "0.15",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "0.1",
            "xtick.color": "0.1",
            "ytick.color": "0.1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    bins = np.arange(0.0, args.x_max + args.bin_width, args.bin_width)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)

    for index, (axis, method) in enumerate(zip(axes, METHODS)):
        method_values = values[method]
        shown_values = method_values[(method_values >= 0) & (method_values <= args.x_max)]
        above_threshold = np.count_nonzero(method_values > args.threshold)
        axis.hist(shown_values, bins=bins, color="0.28", edgecolor="white", linewidth=0.35)
        axis.axvline(args.threshold, color="0.05", linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
        axis.text(
            0.965,
            0.9,
            f"BEC > {args.threshold:g}: {above_threshold:,}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="0.15",
        )
        add_strip(axis, METHOD_LABELS[method])
        axis.set_xlim(0.0, args.x_max)
        axis.set_xticks(np.arange(0, args.x_max + 0.1, 2))
        if index == 1:
            axis.set_xticklabels(["", "2", "4", "6", "8", "10"])
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=9, length=3, width=0.8, pad=2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
        if index == 0:
            axis.set_ylabel("Frequency", fontsize=11)
        else:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False, labelleft=False)

    figure.supxlabel(r"Boundary estimates criterion, BEC($\beta_b$)", fontsize=11.5, y=0.045)
    figure.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.83, wspace=0.08)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {pdf_output}")


def print_summary(values: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    print(
        "Base parameters: "
        f"beta=({DEFAULT_BETA[0]:g}, {DEFAULT_BETA[1]:g}, {DEFAULT_BETA[2]:g}), "
        f"N={DEFAULT_N_SUBJECTS:g}, c={DEFAULT_PROP:g}, alpha={DEFAULT_RHO:g}"
    )
    for method in METHODS:
        method_values = values[method]
        in_axis = np.count_nonzero((method_values >= 0) & (method_values <= args.x_max))
        above = np.count_nonzero(method_values > args.threshold)
        print(
            f"{method}: finite BEC={method_values.size:,}; "
            f"shown within [0, {args.x_max:g}]={in_axis:,}; "
            f"> {args.threshold:g}={above:,}; "
            f"median={np.median(method_values):.3g}; max={np.max(method_values):.3g}"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    values = load_bec_values(args.replications, args.scenario)
    print_summary(values, args)
    plot_figure(values, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
