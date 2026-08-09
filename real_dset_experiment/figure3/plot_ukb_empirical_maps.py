#!/usr/bin/env python3
"""Plot UKB empirical lesion incidence and relative risk across two visits."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "figure3" else SCRIPT_DIR
WORKSPACE_ROOT = PROJECT_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "UKB"
DEFAULT_ANATOMICAL = WORKSPACE_ROOT / "CBMR" / "ALE" / "template" / "MNI152_T1_2mm_brain.nii.gz"
DEFAULT_OUTPUT = SCRIPT_DIR / "figure3_ukb_empirical_maps.png"
DEFAULT_SLICES = (40, 45, 50)
DEFAULT_THRESHOLD = 5e-4
DEFAULT_FIGURE_SIZE = (8.76, 3.2)
DEFAULT_SLICE_GAP = 0.01


def load_inputs(
    data_dir: Path, anatomical_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load lesion matrices, one-based voxel IDs, and the anatomical image."""
    lesion_path = data_dir / "lesions_atleast6_CVR.npz"
    voxel_path = data_dir / "voxel_IDs_CVR.dat"

    with np.load(lesion_path) as lesion_data:
        visit_1 = np.asarray(lesion_data["lesions_vis1"], dtype=float)
        visit_2 = np.asarray(lesion_data["lesions_vis2"], dtype=float)
    voxel_ids = np.loadtxt(voxel_path, dtype=int).reshape(-1)

    if visit_1.shape != visit_2.shape:
        raise ValueError("Visit lesion matrices must have identical dimensions")
    if visit_1.shape[0] != voxel_ids.size:
        raise ValueError("The number of lesion rows must equal the number of voxel IDs")
    if not np.all(np.isin(np.unique(np.concatenate((visit_1, visit_2))), (0, 1))):
        raise ValueError("Lesion matrices must be binary")
    combined_counts = visit_1.sum(axis=1) + visit_2.sum(axis=1)
    if np.any(combined_counts < 6):
        raise ValueError("Every included voxel must contain at least six lesions")

    anatomical_img = nib.load(anatomical_path)
    if len(anatomical_img.shape) != 3:
        raise ValueError("The anatomical reference must be a three-dimensional image")
    if voxel_ids.min() < 1 or voxel_ids.max() > np.prod(anatomical_img.shape):
        raise ValueError("Voxel IDs fall outside the anatomical image grid")
    return visit_1, visit_2, voxel_ids, anatomical_img


def reconstruct_maps(
    visit_1: np.ndarray,
    visit_2: np.ndarray,
    voxel_ids: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 3D maps from one-based R/Fortran-order voxel indices."""
    p1 = visit_1.mean(axis=1)
    p2 = visit_2.mean(axis=1)
    relative_risk = np.divide(
        p2,
        p1,
        out=np.full_like(p1, np.nan),
        where=p1 > 0,
    )

    coordinates = np.unravel_index(voxel_ids - 1, shape, order="F")
    sqrt_incidence_map = np.full(shape, np.nan)
    relative_risk_map = np.full(shape, np.nan)
    sqrt_incidence_map[coordinates] = np.sqrt(p1)
    relative_risk_map[coordinates] = relative_risk
    return sqrt_incidence_map, relative_risk_map


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


def map_to_nifti(
    values: np.ndarray,
    anatomical_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Create a NIfTI map with NaNs outside the reconstructed voxel set."""
    header = anatomical_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(
        np.asarray(values, dtype=np.float32),
        affine=anatomical_img.affine,
        header=header,
    )


def plot_brain_map(
    values: np.ndarray,
    anatomical_img: nib.Nifti1Image,
    cut_coords: list[float],
    output_filename: Path,
    label: str,
    cmap: str,
    threshold: float | None = DEFAULT_THRESHOLD,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    slice_gap: float = DEFAULT_SLICE_GAP,
) -> None:
    """Plot one empirical map using nilearn.plotting.plot_stat_map."""
    finite_values = np.asarray(values, dtype=float).ravel()
    finite_values = finite_values[np.isfinite(finite_values)]
    has_negative = finite_values.size > 0 and np.nanmin(finite_values) < 0
    has_positive = finite_values.size > 0 and np.nanmax(finite_values) > 0
    signed_map = has_negative and has_positive
    if vmax is None:
        if finite_values.size == 0:
            vmax = 1.0
        elif signed_map:
            vmax = float(np.nanpercentile(np.abs(finite_values), 99.0))
        else:
            vmax = float(np.nanpercentile(finite_values, 99.0))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    if vmin is None:
        vmin = -vmax if signed_map else 0.0

    plot_threshold = float(threshold) if threshold is not None else None
    if plot_threshold is not None and plot_threshold <= 0:
        plot_threshold = 1e-6

    stat_img = map_to_nifti(values, anatomical_img)
    figure = plt.figure(figsize=DEFAULT_FIGURE_SIZE, facecolor="white")
    display = plot_stat_map(
        stat_img,
        bg_img=anatomical_img,
        cut_coords=cut_coords,
        display_mode="z",
        draw_cross=False,
        cmap=cmap,
        threshold=plot_threshold,
        colorbar=colorbar,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=signed_map,
        dim=0,
        title=label,
        figure=figure,
    )
    repack_slice_axes(display, slice_gap)
    display.savefig(output_filename)
    display.close()


def repack_slice_axes(display, gap: float) -> None:
    """Repack nilearn z-slice axes left-to-right with a fixed gap."""
    slice_axes = [cut_axis.ax for cut_axis in display.axes.values()]
    slice_axes.sort(key=lambda axis: axis.get_position().x0)
    if not slice_axes:
        return

    x_position = slice_axes[0].get_position().x0
    for axis in slice_axes:
        bounds = axis.get_position()
        axis.set_position([x_position, bounds.y0, bounds.width, bounds.height])
        x_position += bounds.width + gap


def output_path(base_output: Path, suffix: str) -> Path:
    extension = base_output.suffix or ".png"
    return base_output.with_name(f"{base_output.stem}_{suffix}{extension}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate empirical UKB lesion incidence and visit-2/visit-1 risk maps."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Base output path; suffixes are added for sqrt_p1 and p2_over_p1 maps.",
    )
    parser.add_argument("--slices", type=int, nargs="+", default=DEFAULT_SLICES)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Nilearn plot_stat_map threshold; use 0 or a negative value for 1e-6.",
    )
    parser.add_argument(
        "--slice-gap",
        type=float,
        default=DEFAULT_SLICE_GAP,
        help="Figure-fraction gap between nilearn z-slice axes.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.rcParams["savefig.dpi"] = args.dpi
    visit_1, visit_2, voxel_ids, anatomical_img = load_inputs(
        args.data_dir, args.anatomical
    )
    sqrt_incidence, relative_risk = reconstruct_maps(
        visit_1, visit_2, voxel_ids, anatomical_img.shape
    )
    cut_coords = slice_indices_to_cut_coords(tuple(args.slices), anatomical_img)
    figure_specs = (
        (
            "sqrt_p1",
            sqrt_incidence,
            r"$\sqrt{p_1}$",
            "viridis",
            0.0,
            0.7,
        ),
        (
            "p2_over_p1",
            relative_risk,
            r"$p_2/p_1$",
            "RdBu_r",
            0.0,
            2.0,
        ),
    )

    written_outputs = []
    for suffix, overlay, label, cmap, vmin, vmax in figure_specs:
        output = output_path(args.output, suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs = [output]
        pdf_output = output.with_suffix(".pdf")
        if pdf_output != output:
            outputs.append(pdf_output)
        for current_output in outputs:
            plot_brain_map(
                overlay,
                anatomical_img,
                cut_coords,
                current_output,
                label,
                cmap,
                threshold=args.threshold,
                vmin=vmin,
                vmax=vmax,
                slice_gap=args.slice_gap,
            )
        written_outputs.extend(outputs)

    outputs = ", ".join(str(path) for path in written_outputs)
    print(
        f"Wrote {outputs} using "
        f"{visit_1.shape[1]:,} participants and {visit_1.shape[0]:,} voxels"
    )


if __name__ == "__main__":
    main()