#!/usr/bin/env python3
"""Plot separate Figure 6 relative-risk subfigures for UKB fitted methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

from UKB_validation.ukb_python_experiment import (
    DEFAULT_ANATOMICAL,
    DEFAULT_PYTHON_RESULTS_DIR,
    DEFAULT_UKB_DIR,
    default_n_jobs,
    ensure_model_outputs,
    model_result_dir,
)
from UKB_validation.paths import DEFAULT_UKB_DIR
from UKB_validation.io import load_empirical_visits, load_voxel_ids as _load_voxel_ids
from UKB_validation.mapping import values_to_map as _values_to_map


DEFAULT_VOXEL_IDS = DEFAULT_UKB_DIR / "voxel_IDs_CVR.dat"
DEFAULT_SLICES = (40, 45, 50)
FIGURE6_MODELS = ("rr-pgee", "or-gee", "or-pgee")
PREDICTORS = (
    ("c", "baseAge", "Age (visit 1)"),
    ("d", "ageDiff", "Time difference"),
    ("e", "baseCVR", "CVR (visit 1)"),
    ("f", "CVRdiff", "CVR diff."),
)


def parse_args() -> argparse.Namespace:
    """Parse model-selection, plotting, and output options for Figure 6."""
    parser = argparse.ArgumentParser(
        description="Plot separate unmasked Figure 6 relative-risk subfigures."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*FIGURE6_MODELS, "all"),
        default=("all",),
        help="Methods to plot. Use 'all' for rr-pgee, or-gee, and or-pgee.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Optional fitted-map directory; only valid when plotting one model.",
    )
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--voxel-ids", type=Path, default=DEFAULT_VOXEL_IDS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory where the six separate Figure 6 subfigures are saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Output formats saved for each subfigure.",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=DEFAULT_SLICES,
        help="Zero-based axial slice indices to display.",
    )
    parser.add_argument("--sqrt-vmax", type=float, default=0.7)
    parser.add_argument("--rr-vmin", type=float, default=0.0)
    parser.add_argument("--rr-vmax", type=float, default=2.0)
    parser.add_argument("--overlay-alpha", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing fitted maps instead of rerunning model fits.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths, model selection, and Figure 6 plotting parameters."""
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("anatomical image", args.anatomical),
        ("voxel ID file", args.voxel_ids),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.slices:
        raise ValueError("At least one axial slice is required")
    if args.dpi <= 0 or args.sqrt_vmax <= 0 or args.rr_vmax <= args.rr_vmin:
        raise ValueError("--dpi and color-scale ranges must be positive")
    if not 0 < args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must lie in (0, 1]")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")
    if args.result_dir is not None and len(resolve_models(args.models)) != 1:
        raise ValueError("--result-dir can only be used with a single model")


def resolve_models(models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Expand the ``all`` option to the methods used in Figure 6."""
    return FIGURE6_MODELS if "all" in models else tuple(models)


def load_voxel_ids(path: Path, max_voxels: int | None = None) -> np.ndarray:
    """Load unique one-based voxel indices, optionally truncating the mask."""
    return _load_voxel_ids(path, max_voxels=max_voxels)


def load_empirical_maps(
    ukb_dir: Path,
    voxel_ids: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build empirical incidence and relative-risk maps from two visits."""
    visit_1, visit_2 = load_empirical_visits(ukb_dir, voxel_ids.size)

    p1 = visit_1.mean(axis=1)
    p2 = visit_2.mean(axis=1)
    relative_risk = np.divide(p2, p1, out=np.full_like(p1, np.nan), where=p1 > 0)
    return values_to_map(np.sqrt(p1), voxel_ids, shape), values_to_map(relative_risk, voxel_ids, shape)


def values_to_map(values: np.ndarray, voxel_ids: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Place mask values into a 3D array using Fortran-order voxel IDs."""
    return _values_to_map(values, voxel_ids, shape)


def load_relative_risk_map(
    result_dir: Path,
    predictor: str,
    anatomical: nib.Nifti1Image,
    voxel_ids: np.ndarray,
) -> np.ndarray:
    """Load a coefficient map and exponentiate it into relative risks."""
    path = result_dir / f"estimate_{predictor}_GEE.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Coefficient map not found: {path}")
    image = nib.load(path)
    if image.shape != anatomical.shape or not np.allclose(image.affine, anatomical.affine):
        raise ValueError(f"Coefficient map is not aligned with anatomical image: {path}")
    beta_map = np.asarray(image.get_fdata(), dtype=float)
    beta_values = beta_map.ravel(order="F")[voxel_ids - 1]
    with np.errstate(over="ignore", invalid="ignore"):
        rr_values = np.exp(beta_values)
    return values_to_map(rr_values, voxel_ids, anatomical.shape)


def brain_crop(anatomical: np.ndarray, slices: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return shared plot limits covering all requested slices."""
    masks = [anatomical[:, :, slice_index].T > 0 for slice_index in slices]
    positive = np.argwhere(np.any(np.stack(masks, axis=0), axis=0))
    if positive.size == 0:
        return (0, anatomical.shape[0] - 1), (0, anatomical.shape[1] - 1)
    margin = 2
    row_min, column_min = positive.min(axis=0)
    row_max, column_max = positive.max(axis=0)
    x_limits = (max(int(column_min) - margin, 0), min(int(column_max) + margin, anatomical.shape[0] - 1))
    y_limits = (max(int(row_min) - margin, 0), min(int(row_max) + margin, anatomical.shape[1] - 1))
    return x_limits, y_limits


def rr_scale(vmin: float, vmax: float):
    """Build the relative-risk color scale and its labeled tick values."""
    ticks = (vmin, 0.5, 1.0, 1.5, vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap, norm, ticks


def save_panel(
    output_dir: Path,
    formats: tuple[str, ...],
    stem: str,
    anatomical: np.ndarray,
    values: np.ndarray,
    slices: tuple[int, ...],
    title: str,
    cmap,
    norm,
    ticks: tuple[float, ...],
    *,
    x_limits: tuple[int, int],
    y_limits: tuple[int, int],
    anatomical_vmax: float,
    overlay_alpha: float,
    dpi: int,
) -> None:
    """Save one multi-slice panel in each requested output format."""
    figure = plt.figure(figsize=(6.4, 3.0), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1, 1, 1, 0.06),
        wspace=0.04,
        left=0.035,
        right=0.9,
        bottom=0.035,
        top=0.98,
    )

    background_axis = figure.add_subplot(grid[0, :3])
    background_axis.set_facecolor("black")
    background_axis.set_xticks([])
    background_axis.set_yticks([])
    background_axis.set_zorder(0)
    for spine in background_axis.spines.values():
        spine.set_visible(False)
    background_axis.text(
        0.0,
        1.0,
        title,
        transform=background_axis.transAxes,
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )

    for local_column, slice_index in enumerate(slices):
        axis = figure.add_subplot(grid[0, local_column])
        axis.set_facecolor("black")
        axis.set_zorder(1)
        axis.imshow(
            anatomical[:, :, slice_index].T,
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
            norm=norm,
            origin="lower",
            interpolation="nearest",
            alpha=overlay_alpha,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    colorbar_axis = figure.add_subplot(grid[0, 3])
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, ticks=ticks)
    colorbar.ax.tick_params(labelsize=8, length=3, pad=2)
    colorbar.outline.set_linewidth(0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output = output_dir / f"{stem}.{file_format}"
        save_kwargs = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}
        if file_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output, **save_kwargs)
        print(f"Saved {file_format.upper()}: {output}")
    plt.close(figure)


def plot_maps(
    anatomical_img: nib.Nifti1Image,
    panels: list[tuple[str, str, np.ndarray, str]],
    args: argparse.Namespace,
) -> None:
    """Render all empirical and model panels comprising Figure 6."""
    anatomical = np.asarray(anatomical_img.get_fdata(), dtype=float)
    slices = tuple(int(index) for index in args.slices)
    if len(slices) != 3:
        raise ValueError("This layout expects exactly three axial slices")
    if any(index < 0 or index >= anatomical.shape[2] for index in slices):
        raise ValueError(f"Slices must be between 0 and {anatomical.shape[2] - 1}")

    positive = anatomical[anatomical > 0]
    anatomical_vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    x_limits, y_limits = brain_crop(anatomical, slices)

    sqrt_cmap = plt.get_cmap("viridis").copy()
    sqrt_cmap.set_bad((0, 0, 0, 0))
    sqrt_norm = Normalize(vmin=0.0, vmax=args.sqrt_vmax)
    rr_cmap, rr_norm, rr_boundaries = rr_scale(args.rr_vmin, args.rr_vmax)

    for stem, title, values, scale in panels:
        if scale == "sqrt":
            cmap = sqrt_cmap
            norm = sqrt_norm
            ticks = tuple(np.arange(0.0, args.sqrt_vmax + 0.001, 0.1))
        else:
            cmap = rr_cmap
            norm = rr_norm
            ticks = rr_boundaries
        save_panel(
            args.output_dir,
            tuple(args.formats),
            stem,
            anatomical,
            values,
            slices,
            title,
            cmap,
            norm,
            ticks,
            x_limits=x_limits,
            y_limits=y_limits,
            anatomical_vmax=anatomical_vmax,
            overlay_alpha=args.overlay_alpha,
            dpi=args.dpi,
        )


def main() -> None:
    """Load model results and write the Figure 6 subfigures."""
    args = parse_args()
    validate_args(args)
    models = resolve_models(args.models)
    anatomical_img = nib.load(args.anatomical)
    voxel_ids = load_voxel_ids(args.voxel_ids, max_voxels=args.max_voxels)
    sqrt_p1, empirical_rr = load_empirical_maps(args.ukb_dir, voxel_ids, anatomical_img.shape)
    for model in models:
        result_dir = args.result_dir or model_result_dir(args.python_results_dir, model)
        result_dir = ensure_model_outputs(
            model,
            ukb_dir=args.ukb_dir,
            results_root=args.python_results_dir,
            result_dir=result_dir,
            anatomical=args.anatomical,
            n_jobs=args.n_jobs,
            max_voxels=args.max_voxels,
            force_rerun=not args.use_cache,
        )
        model_stem = model.replace("-", "_")
        panels: list[tuple[str, str, np.ndarray, str]] = [
            (f"figure6_{model_stem}_a_sqrt_p1", r"$\sqrt{p_1}$", sqrt_p1, "sqrt"),
            (f"figure6_{model_stem}_b_p2_over_p1", r"$p_2/p_1$", empirical_rr, "rr"),
        ]
        for letter, predictor, title in PREDICTORS:
            stem = f"figure6_{model_stem}_{letter}_{predictor}_relative_risk"
            panels.append(
                (
                    stem,
                    title,
                    load_relative_risk_map(result_dir, predictor, anatomical_img, voxel_ids),
                    "rr",
                )
            )

        print(f"{model.upper()} subfigures:")
        for _, title, values, _ in panels:
            print(f"{title}: {np.count_nonzero(np.isfinite(values)):,} displayed voxels")
        plot_maps(anatomical_img, panels, args)


if __name__ == "__main__":
    main()
