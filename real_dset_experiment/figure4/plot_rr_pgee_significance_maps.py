#!/usr/bin/env python3
"""Plot eight UKB coefficient z-score maps on one axial slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import nibabel as nib
import numpy as np
from scipy import stats

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


DEFAULT_RESULT_DIRS = {
    "rr-gee": model_result_dir(DEFAULT_PYTHON_RESULTS_DIR, "rr-gee"),
    "rr-pgee": model_result_dir(DEFAULT_PYTHON_RESULTS_DIR, "rr-pgee"),
    "or-gee": model_result_dir(DEFAULT_PYTHON_RESULTS_DIR, "or-gee"),
    "or-pgee": model_result_dir(DEFAULT_PYTHON_RESULTS_DIR, "or-pgee"),
}
DEFAULT_VOXEL_IDS = DEFAULT_UKB_DIR / "voxel_IDs_CVR.dat"

PREDICTORS = (
    ("baseAge", "Age (visit 1)"),
    ("ageDiff", "Time difference"),
    ("baseCVR", "CVR (visit 1)"),
    ("CVRdiff", "CVR diff."),
    ("sexM", "Sex"),
    ("headsize", "Head size"),
    ("ageBYageDiff", "Age (visit 1):Time diff."),
    ("ageBYsexM", "Age (visit 1):Sex"),
)


def parse_args(default_model: str = "rr-pgee") -> argparse.Namespace:
    """Parse options for a coefficient significance-map figure."""
    parser = argparse.ArgumentParser(
        description="Plot UKB coefficient significance maps in a 2-by-4 layout."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_RESULT_DIRS),
        default=default_model,
        help="UKB fitted model to plot.",
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--voxel-ids", type=Path, default=DEFAULT_VOXEL_IDS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF path; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument(
        "--significance",
        choices=("fixed", "fdr", "none"),
        default="fixed",
        help="Voxel display rule: fixed |z| threshold, BH FDR, or no filtering.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=45,
        help="One-based axial slice number, matching the original figure convention.",
    )
    parser.add_argument("--z-threshold", type=float, default=1.96)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--vmax", type=float, default=5.0)
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=1.0,
        help="Opacity of significant z-score voxels; 1 gives undiluted colors.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning the model fits.",
    )
    args = parser.parse_args()
    args.result_dir = args.result_dir or model_result_dir(args.python_results_dir, args.model)
    if args.output is None:
        model_slug = args.model.replace("-", "_")
        args.output = SCRIPT_DIR / (
            f"figure4_{model_slug}_significance_maps_"
            f"{args.significance}_z{args.slice}.png"
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and plotting parameters before loading model maps."""
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("anatomical image", args.anatomical),
        ("voxel ID file", args.voxel_ids),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.slice < 1:
        raise ValueError("--slice must be a positive one-based slice number")
    if args.z_threshold <= 0 or args.vmax <= 0 or args.dpi <= 0:
        raise ValueError("--z-threshold, --vmax, and --dpi must be positive")
    if not 0 < args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must lie in (0, 1]")
    if not 0 < args.fdr_alpha < 1:
        raise ValueError("--fdr-alpha must lie strictly between 0 and 1")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")


def load_voxel_ids(path: Path) -> np.ndarray:
    """Load unique one-based voxel indices used by the UKB analysis mask."""
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0 or np.any(voxel_ids < 1):
        raise ValueError("Voxel IDs must be nonempty positive one-based indices")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError("Voxel IDs must be unique")
    return voxel_ids


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values, retaining failed fits as p=1 tests."""
    finite_p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    order = np.argsort(finite_p_values)
    ranked = finite_p_values[order]
    ranks = np.arange(1, ranked.size + 1)
    adjusted_ranked = np.minimum.accumulate(
        (ranked * ranked.size / ranks)[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def filter_zscores(
    zscores: np.ndarray, significance: str, z_threshold: float, fdr_alpha: float
) -> np.ndarray:
    """Mask z-scores according to fixed, FDR, or unfiltered significance."""
    finite = np.isfinite(zscores)
    if significance == "fixed":
        keep = finite & (np.abs(zscores) > z_threshold)
    elif significance == "fdr":
        p_values = np.full(zscores.shape, np.nan)
        p_values[finite] = 2 * stats.norm.sf(np.abs(zscores[finite]))
        keep = benjamini_hochberg(p_values) <= fdr_alpha
    else:
        keep = finite
    return np.where(keep, zscores, np.nan)


def load_maps(
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, int]]]:
    """Load aligned coefficient maps and apply the selected display filter."""
    anatomical_image = nib.load(args.anatomical)
    anatomical = np.asarray(anatomical_image.get_fdata(), dtype=float)
    if anatomical.ndim != 3 or args.slice > anatomical.shape[2]:
        raise ValueError(
            f"Slice {args.slice} is outside anatomical shape {anatomical.shape}"
        )

    voxel_ids = load_voxel_ids(args.voxel_ids)
    if args.max_voxels is not None:
        voxel_ids = voxel_ids[: args.max_voxels]
    if voxel_ids.max() > anatomical.size:
        raise ValueError("Voxel IDs exceed the anatomical image grid")

    maps: list[tuple[str, np.ndarray, int]] = []
    for predictor, title in PREDICTORS:
        path = args.result_dir / f"zscore_{predictor}_GEE.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{args.model.upper()} z-statistic map not found: {path}"
            )
        image = nib.load(path)
        if image.shape != anatomical_image.shape or not np.allclose(
            image.affine, anatomical_image.affine
        ):
            raise ValueError(
                f"{args.model.upper()} map is not aligned with the anatomical: {path}"
            )
        data = np.asarray(image.get_fdata(), dtype=float)
        # Flatten in the same order as the UKB mask IDs, not NumPy's default C order.
        zscores = data.ravel(order="F")[voxel_ids - 1]
        filtered = filter_zscores(
            zscores, args.significance, args.z_threshold, args.fdr_alpha
        )
        map_flat = np.full(anatomical.size, np.nan)
        map_flat[voxel_ids - 1] = filtered
        maps.append(
            (
                title,
                map_flat.reshape(anatomical.shape, order="F"),
                int(np.count_nonzero(np.isfinite(filtered))),
            )
        )
    return anatomical, maps


def brain_crop(anatomical_slice: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return plot limits around the non-zero portion of an anatomical slice."""
    positive = np.argwhere(anatomical_slice > 0)
    if positive.size == 0:
        return (0, anatomical_slice.shape[1] - 1), (0, anatomical_slice.shape[0] - 1)
    margin = 2
    row_min, column_min = positive.min(axis=0)
    row_max, column_max = positive.max(axis=0)
    x_limits = (
        max(int(column_min) - margin, 0),
        min(int(column_max) + margin, anatomical_slice.shape[1] - 1),
    )
    y_limits = (
        max(int(row_min) - margin, 0),
        min(int(row_max) + margin, anatomical_slice.shape[0] - 1),
    )
    return x_limits, y_limits


def plot_maps(
    anatomical: np.ndarray,
    maps: list[tuple[str, np.ndarray, int]],
    args: argparse.Namespace,
) -> None:
    """Render the eight predictor maps in the Figure 4 layout."""
    slice_index = args.slice - 1
    anatomical_slice = anatomical[:, :, slice_index].T
    positive = anatomical_slice[anatomical_slice > 0]
    anatomical_vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    x_limits, y_limits = brain_crop(anatomical_slice)

    figure = plt.figure(figsize=(12.0, 5.75), facecolor="white")
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(1, 1, 1, 1, 0.045),
        wspace=0.035,
        hspace=0.075,
        left=0.025,
        right=0.965,
        bottom=0.035,
        top=0.965,
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    normalization = Normalize(vmin=-args.vmax, vmax=args.vmax)
    mappable = ScalarMappable(norm=normalization, cmap=cmap)

    for row in range(2):
        row_background = figure.add_subplot(grid[row, :4])
        row_background.set_facecolor("black")
        row_background.set_xticks([])
        row_background.set_yticks([])
        row_background.set_zorder(0)
        for spine in row_background.spines.values():
            spine.set_visible(False)

    for index, (title, values, _) in enumerate(maps):
        row, column = divmod(index, 4)
        axis = figure.add_subplot(grid[row, column])
        axis.set_facecolor("black")
        axis.set_zorder(1)
        axis.imshow(
            anatomical_slice,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=anatomical_vmax,
            interpolation="nearest",
        )
        overlay = np.ma.masked_invalid(values[:, :, slice_index].T)
        axis.imshow(
            overlay,
            cmap=cmap,
            norm=normalization,
            origin="lower",
            interpolation="nearest",
            alpha=args.overlay_alpha,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.text(
            0.015,
            0.985,
            title,
            transform=axis.transAxes,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    colorbar_ticks = np.arange(-4, 5, 2) if args.vmax >= 4 else None
    for row in range(2):
        colorbar_axis = figure.add_subplot(grid[row, 4])
        colorbar = figure.colorbar(mappable, cax=colorbar_axis, ticks=colorbar_ticks)
        colorbar.ax.tick_params(labelsize=9, length=3, pad=2)
        colorbar.outline.set_linewidth(0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {pdf_output}")


def main(default_model: str = "rr-pgee") -> None:
    """Prepare model outputs, load maps, and write a significance figure."""
    args = parse_args(default_model)
    validate_args(args)
    args.result_dir = ensure_model_outputs(
        args.model,
        ukb_dir=args.ukb_dir,
        results_root=args.python_results_dir,
        result_dir=args.result_dir,
        anatomical=args.anatomical,
        n_jobs=args.n_jobs,
        max_voxels=args.max_voxels,
        force_rerun=not args.use_cache,
    )
    anatomical, maps = load_maps(args)
    print(
        f"{args.model.upper()} significance mode: {args.significance}; "
        f"one-based axial slice: {args.slice}"
    )
    for (predictor, _), (_, _, count) in zip(PREDICTORS, maps):
        print(f"{predictor}: {count:,} displayed voxels across the 3D analysis mask")
    plot_maps(anatomical, maps, args)


if __name__ == "__main__":
    main()