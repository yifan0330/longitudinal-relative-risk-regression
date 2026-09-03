#!/usr/bin/env python3
"""Plot UKB Figure B4 method-specific boundary-estimate voxels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.affines import apply_affine
from nilearn.plotting import plot_stat_map
import numpy as np
from matplotlib.colors import ListedColormap

from UKB_validation.mapping import values_to_map
from UKB_validation.paths import DEFAULT_ANATOMICAL, DEFAULT_PYTHON_RESULTS_DIR


HERE = Path(__file__).resolve().parent
DEFAULT_RR_BEC_CACHE = HERE.parent / "figureB3" / "figureB3_bec_values.npz"
DEFAULT_OR_BEC_CACHE = (
    HERE.parents[1]
    / "RR_OR_GEE_PGEE"
    / "results"
    / "full"
    / "figureB1"
    / "figureB1_or_bec_values.npz"
)
METHOD_LABELS = {
    "rr_gee": "RR-GEE",
    "rr_pgee": "RR-PGEE",
    "or_gee": "OR-GEE",
    "or_pgee": "OR-PGEE",
}
DEFAULT_SLICES = (40, 45, 50)
DEFAULT_THRESHOLD = 10.0
DEFAULT_FIGURE_SIZE = (4.0, 2.15)
DEFAULT_BOTTOM_MARGIN = 0.20


def default_output(method: str) -> Path:
    """Return the default Figure B4 output for one fitted method."""
    return HERE / f"figureB4_{method}_boundary_voxels.png"


def parse_args() -> argparse.Namespace:
    """Parse Figure B4 input and output options."""
    parser = argparse.ArgumentParser(
        description="Plot UKB voxels with BEC > 10 or failed fitting on axial slices."
    )
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_LABELS),
        default="rr_gee",
        help="Fitted method to plot.",
    )
    parser.add_argument(
        "--bec-cache",
        type=Path,
        default=None,
        help="Optional BEC cache. By default the script chooses the RR or OR cache by method.",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=None,
        help="Optional fit summary. By default this is selected from UKB/python_results/<method>.",
    )
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument("--slices", type=int, nargs="+", default=DEFAULT_SLICES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def resolve_bec_cache(method: str, requested_cache: Path | None) -> Path:
    """Return a BEC cache containing the requested method."""
    if requested_cache is not None:
        return requested_cache
    preferred = DEFAULT_RR_BEC_CACHE if method.startswith("rr_") else DEFAULT_OR_BEC_CACHE
    fallback = DEFAULT_OR_BEC_CACHE if method.startswith("rr_") else DEFAULT_RR_BEC_CACHE
    for cache in (preferred, fallback):
        if cache.exists():
            with np.load(cache) as cache_data:
                if method in cache_data.files:
                    return cache
    raise FileNotFoundError(f"No BEC cache containing {method!r} was found")


def resolve_fit_summary(method: str, requested_summary: Path | None) -> Path:
    """Return the fit-summary path for the requested method."""
    return requested_summary or DEFAULT_PYTHON_RESULTS_DIR / method / "fit_summary.npz"


def slice_indices_to_cut_coords(
    slice_indices: tuple[int, ...], anatomical_img: nib.Nifti1Image
) -> list[float]:
    """Convert axial voxel slice indices to z coordinates for nilearn."""
    if any(index < 0 or index >= anatomical_img.shape[2] for index in slice_indices):
        raise ValueError(
            f"Slice indices must lie between 0 and {anatomical_img.shape[2] - 1}"
        )
    voxel_coords = np.column_stack(
        (
            np.zeros(len(slice_indices)),
            np.zeros(len(slice_indices)),
            np.asarray(slice_indices),
        )
    )
    return apply_affine(anatomical_img.affine, voxel_coords)[:, 2].astype(float).tolist()


def map_to_nifti(values: np.ndarray, anatomical_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Create a NIfTI image with the anatomical geometry."""
    header = anatomical_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(
        np.asarray(values, dtype=np.float32),
        affine=anatomical_img.affine,
        header=header,
    )


def load_flagged_voxels(
    bec_cache: Path,
    fit_summary: Path,
    method: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load BEC and fit flags, returning one boolean flag per voxel."""
    with np.load(bec_cache) as bec_data:
        if method not in bec_data.files:
            raise KeyError(f"{bec_cache} does not contain BEC values for {method!r}")
        bec = np.asarray(bec_data[method], dtype=float)
    with np.load(fit_summary, allow_pickle=True) as summary_data:
        voxel_ids = np.asarray(summary_data["voxel_ids"], dtype=int)
        converged = np.asarray(summary_data["converged"], dtype=bool)
        failed = np.asarray(summary_data["failed"], dtype=bool)

    if bec.shape[0] != voxel_ids.size:
        raise ValueError("BEC rows do not match fit-summary voxel IDs")

    finite_all = np.all(np.isfinite(bec), axis=1)
    bec_above = np.any(bec > threshold, axis=1)
    irls_failed = failed | ~converged | ~finite_all
    flagged = bec_above | irls_failed
    counts = {
        "total_voxels": int(voxel_ids.size),
        "flagged_voxels": int(np.count_nonzero(flagged)),
        "bec_gt_threshold": int(np.count_nonzero(bec_above)),
        "irls_failed_or_missing": int(np.count_nonzero(irls_failed)),
        "failed": int(np.count_nonzero(failed)),
        "not_converged": int(np.count_nonzero(~converged)),
        "missing_bec": int(np.count_nonzero(~finite_all)),
    }
    return voxel_ids, flagged, counts


def plot_boundary_voxels(
    flagged_map: np.ndarray,
    anatomical_img: nib.Nifti1Image,
    cut_coords: list[float],
    slice_indices: tuple[int, ...],
    output: Path,
    dpi: int,
) -> None:
    """Plot flagged voxels as black markers on the anatomical background."""
    stat_img = map_to_nifti(flagged_map, anatomical_img)
    figure = plt.figure(figsize=DEFAULT_FIGURE_SIZE, facecolor="black")
    display = plot_stat_map(
        stat_img,
        bg_img=anatomical_img,
        cut_coords=cut_coords,
        display_mode="z",
        annotate=False,
        draw_cross=False,
        cmap=ListedColormap(["black"]),
        threshold=0.5,
        colorbar=False,
        black_bg=True,
        dim=0,
        figure=figure,
    )
    slice_axes = [cut_axis.ax for cut_axis in display.axes.values()]
    slice_axes.sort(key=lambda axis: axis.get_position().x0)
    for axis, slice_index in zip(slice_axes, slice_indices):
        bounds = axis.get_position()
        axis.set_position(
            [
                bounds.x0,
                bounds.y0 + DEFAULT_BOTTOM_MARGIN,
                bounds.width,
                max(bounds.height - DEFAULT_BOTTOM_MARGIN, 0.1),
            ]
        )
        bounds = axis.get_position()
        figure.text(
            bounds.x0 + bounds.width / 2.0,
            DEFAULT_BOTTOM_MARGIN / 3.0,
            f"z = {slice_index}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    display.savefig(output, dpi=dpi)
    display.close()


def main() -> None:
    """Create Figure B4 outputs."""
    args = parse_args()
    bec_cache = resolve_bec_cache(args.method, args.bec_cache)
    fit_summary = resolve_fit_summary(args.method, args.fit_summary)
    output = args.output or default_output(args.method)
    anatomical_img = nib.load(args.anatomical)
    if len(anatomical_img.shape) != 3:
        raise ValueError("The anatomical reference must be a three-dimensional image")

    voxel_ids, flagged, counts = load_flagged_voxels(
        bec_cache,
        fit_summary,
        args.method,
        args.threshold,
    )
    flagged_values = np.where(flagged, 1.0, np.nan)
    flagged_map = values_to_map(flagged_values, voxel_ids, anatomical_img.shape)
    cut_coords = slice_indices_to_cut_coords(tuple(args.slices), anatomical_img)

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = args.pdf_output or output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    for current_output in (output, pdf_output):
        plot_boundary_voxels(
            flagged_map,
            anatomical_img,
            cut_coords,
            tuple(args.slices),
            current_output,
            args.dpi,
        )

    print(
        "{method}: flagged {flagged_voxels:,}/{total_voxels:,} voxels "
        "({percent:.2f}%). BEC > {threshold:g}: {bec_gt_threshold:,}; "
        "IRLS failed/not converged/missing BEC: {irls_failed_or_missing:,}.".format(
            method=METHOD_LABELS[args.method],
            percent=100.0 * counts["flagged_voxels"] / counts["total_voxels"],
            threshold=args.threshold,
            **counts,
        )
    )
    print(f"BEC cache: {bec_cache}")
    print(f"Fit summary: {fit_summary}")
    print(f"Saved PNG: {output}")
    print(f"Saved PDF: {pdf_output}")


if __name__ == "__main__":
    main()