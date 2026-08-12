#!/usr/bin/env python3
"""Plot Figure 7 age relative-risk comparisons across UKB fitted methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent

from real_dset_experiment.ukb_python_experiment import (
    DEFAULT_ANATOMICAL,
    DEFAULT_PYTHON_RESULTS_DIR,
    DEFAULT_UKB_DIR,
    default_n_jobs,
    ensure_model_outputs,
    model_result_dir,
)
from real_dset_experiment.paths import DEFAULT_UKB_DIR


DEFAULT_VOXEL_IDS = DEFAULT_UKB_DIR / "voxel_IDs_CVR.dat"
DEFAULT_OUTPUT = SCRIPT_DIR / "figure7_age_relative_risk_comparison.png"
METHODS = ("rr-gee", "rr-pgee", "or-pgee")


def parse_args() -> argparse.Namespace:
    """Parse Figure 7 data, filtering, and plotting options."""
    parser = argparse.ArgumentParser(
        description="Create Figure 7 comparing age-at-visit-1 relative risks."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--voxel-ids", type=Path, default=DEFAULT_VOXEL_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF path; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument("--axis-min", type=float, default=-0.1)
    parser.add_argument("--axis-max", type=float, default=2.1)
    parser.add_argument("--colorbar-vmax", type=float, default=7000.0)
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=0.05,
        help="Radius in plot units for density coloring by local neighbor count.",
    )
    parser.add_argument(
        "--include-nonconverged",
        action="store_true",
        help="Include finite estimates from non-converged fits; default uses common converged voxels.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning model fits.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate Figure 7 paths and comparison parameters."""
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("anatomical image", args.anatomical),
        ("voxel ID file", args.voxel_ids),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.dpi <= 0 or args.n_jobs <= 0:
        raise ValueError("--dpi and --n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")
    if args.axis_max <= args.axis_min:
        raise ValueError("--axis-max must exceed --axis-min")
    if args.colorbar_vmax <= 0:
        raise ValueError("--colorbar-vmax must be positive")
    if args.neighbor_radius <= 0:
        raise ValueError("--neighbor-radius must be positive")


def load_voxel_ids(path: Path, max_voxels: int | None = None) -> np.ndarray:
    """Load unique one-based voxel indices for the comparison."""
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0 or np.any(voxel_ids < 1):
        raise ValueError("Voxel IDs must be nonempty positive one-based indices")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError("Voxel IDs must be unique")
    if max_voxels is not None:
        voxel_ids = voxel_ids[:max_voxels]
    return voxel_ids


def load_visit1_incidence(ukb_dir: Path, n_voxels: int) -> tuple[np.ndarray, int]:
    """Return visit-1 incidence values and participant count."""
    with np.load(ukb_dir / "lesions_atleast6_CVR.npz") as lesion_data:
        visit_1 = np.asarray(lesion_data["lesions_vis1"], dtype=float)
    if visit_1.shape[0] < n_voxels:
        raise ValueError("Lesion matrix contains fewer voxels than voxel_IDs_CVR.dat")
    visit_1 = visit_1[:n_voxels]
    return visit_1.mean(axis=1), visit_1.shape[1]


def load_beta(result_dir: Path, voxel_ids: np.ndarray, anatomical: Path) -> np.ndarray:
    """Load the age coefficient at the selected anatomical voxels."""
    template = nib.load(anatomical)
    path = result_dir / "estimate_baseAge_GEE.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Coefficient map not found: {path}")
    image = nib.load(path)
    if image.shape != template.shape or not np.allclose(image.affine, template.affine):
        raise ValueError(f"Coefficient map is not aligned with anatomical image: {path}")
    return np.asarray(image.get_fdata(), dtype=float).ravel(order="F")[voxel_ids - 1]


def load_fit_mask(result_dir: Path, n_voxels: int) -> tuple[np.ndarray, int, int]:
    """Return fit validity and counts of failed or non-converged fits."""
    summary_path = result_dir / "fit_summary.npz"
    if not summary_path.is_file():
        return np.ones(n_voxels, dtype=bool), 0, 0
    with np.load(summary_path) as summary:
        converged = np.asarray(summary["converged"], dtype=bool).reshape(-1)[:n_voxels]
        failed = np.asarray(summary["failed"], dtype=bool).reshape(-1)[:n_voxels]
    return converged & ~failed, int(np.count_nonzero(failed)), int(np.count_nonzero(~converged & ~failed))


def odds_ratio_to_relative_risk(odds_ratio: np.ndarray, baseline_risk: np.ndarray) -> np.ndarray:
    """Convert odds ratios to relative risks at a given baseline risk."""
    return odds_ratio / (1.0 - baseline_risk + baseline_risk * odds_ratio)


def load_method_values(args: argparse.Namespace, voxel_ids: np.ndarray, p1: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    """Load comparable age effects, fit masks, and fit-quality summaries."""
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    summaries: dict[str, tuple[int, int]] = {}
    for method in METHODS:
        result_dir = ensure_model_outputs(
            method,
            ukb_dir=args.ukb_dir,
            results_root=args.python_results_dir,
            result_dir=model_result_dir(args.python_results_dir, method),
            anatomical=args.anatomical,
            n_jobs=args.n_jobs,
            max_voxels=args.max_voxels,
            force_rerun=not args.use_cache,
        )
        beta = load_beta(result_dir, voxel_ids, args.anatomical)
        with np.errstate(over="ignore", invalid="ignore"):
            exponentiated = np.exp(beta)
        # OR coefficients need the baseline-risk correction before comparison
        # with the directly modelled RR coefficients.
        if method.startswith("or-"):
            values[method] = odds_ratio_to_relative_risk(exponentiated, p1)
        else:
            values[method] = exponentiated
        fit_mask, n_failed, n_not_converged = load_fit_mask(result_dir, voxel_ids.size)
        masks[method] = fit_mask
        summaries[method] = (n_failed, n_not_converged)
    return values, masks, summaries


def density_counts(x: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    """Count neighboring scatter points within a plotting-radius ball."""
    points = np.column_stack((x, y))
    tree = cKDTree(points)
    return np.asarray(tree.query_ball_point(points, r=radius, return_length=True), dtype=float)


def plot_panel(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ylabel: str | None,
    panel_index: int,
    args: argparse.Namespace,
) -> tuple[plt.Collection, int]:
    """Plot one method comparison and return its scatter and point count."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    counts = density_counts(x, y, args.neighbor_radius) if x.size else np.array([])
    order = np.argsort(counts) if counts.size else np.array([], dtype=int)
    scatter = axis.scatter(
        x[order],
        y[order],
        c=np.clip(counts[order], 0, args.colorbar_vmax),
        s=7,
        cmap="viridis",
        alpha=0.32,
        vmin=0,
        vmax=args.colorbar_vmax,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(
        [args.axis_min, args.axis_max],
        [args.axis_min, args.axis_max],
        color="black",
        linestyle=(0, (4, 4)),
        linewidth=0.8,
    )
    axis.set_xlim(args.axis_min, args.axis_max)
    axis.set_ylim(args.axis_min, args.axis_max)
    strip = Rectangle(
        (0.0, 1.01),
        1.0,
        0.065,
        transform=axis.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        clip_on=False,
    )
    axis.add_patch(strip)
    axis.text(
        0.5,
        1.042,
        title,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        clip_on=False,
    )
    axis.set_xlabel("")
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=11)
    axis.tick_params(axis="both", labelsize=9, length=3, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
    axis.spines["left"].set_visible(panel_index == 0)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="y", left=panel_index == 0, labelleft=panel_index == 0)
    return scatter, x.size


def plot_comparison(
    values: dict[str, np.ndarray],
    common_mask: np.ndarray,
    args: argparse.Namespace,
    n_subjects: int,
    n_voxels: int,
) -> None:
    """Render the three pairwise method comparisons for Figure 7."""
    rr_gee = values["rr-gee"][common_mask]
    rr_pgee = values["rr-pgee"][common_mask]
    rr_or_pgee = values["or-pgee"][common_mask]

    figure, axes = plt.subplots(1, 3, figsize=(8.6, 4.55), sharex=True, sharey=True)
    panels = (
        (rr_gee, rr_pgee, "RR-PGEE vs RR-GEE"),
        (rr_gee, rr_or_pgee, "RR (OR-PGEE) vs RR-GEE"),
        (rr_or_pgee, rr_pgee, "RR-PGEE vs RR (OR-PGEE)"),
    )

    scatters = []
    panel_counts = []
    for index, (axis, (x, y, title)) in enumerate(zip(axes, panels)):
        scatter, count = plot_panel(
            axis,
            x,
            y,
            title,
            "Relative risks for Age (visit 1)" if index == 0 else None,
            index,
            args,
        )
        scatters.append(scatter)
        panel_counts.append(count)

    figure.subplots_adjust(left=0.085, right=0.84, bottom=0.12, top=0.82, wspace=0.09)
    identity_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle=(0, (4, 4)),
        linewidth=0.8,
        label=r"$y=x$",
    )
    figure.legend(
        handles=[identity_handle],
        loc="upper left",
        bbox_to_anchor=(0.858, 0.75),
        frameon=False,
        fontsize=8,
        handlelength=2.2,
        borderpad=0.2,
        labelspacing=0.2,
    )
    colorbar_axis = figure.add_axes((0.865, 0.34, 0.018, 0.29))
    colorbar = figure.colorbar(scatters[-1], cax=colorbar_axis, ticks=(0, 2000, 4000, 6000))
    colorbar.set_label("Neighbors", fontsize=9, labelpad=6)
    colorbar.ax.tick_params(labelsize=8, length=3, pad=2)
    colorbar.outline.set_linewidth(0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)

    print(f"Subjects: {n_subjects:,}; analysis voxels explored: {n_voxels:,}")
    print(f"Common voxels plotted: {int(np.count_nonzero(common_mask)):,}")
    print("Panel finite point counts: " + ", ".join(f"{count:,}" for count in panel_counts))
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {pdf_output}")


def main() -> None:
    """Load comparable fitted values and write Figure 7."""
    args = parse_args()
    validate_args(args)
    voxel_ids = load_voxel_ids(args.voxel_ids, max_voxels=args.max_voxels)
    p1, n_subjects = load_visit1_incidence(args.ukb_dir, voxel_ids.size)
    values, masks, summaries = load_method_values(args, voxel_ids, p1)

    common_mask = np.isfinite(p1)
    for method in METHODS:
        common_mask &= np.isfinite(values[method])
        if not args.include_nonconverged:
            common_mask &= masks[method]
    for method in METHODS:
        n_failed, n_not_converged = summaries[method]
        print(f"{method.upper()}: {n_failed:,} failed; {n_not_converged:,} non-converged")

    plot_comparison(values, common_mask, args, n_subjects, voxel_ids.size)


if __name__ == "__main__":
    main()
