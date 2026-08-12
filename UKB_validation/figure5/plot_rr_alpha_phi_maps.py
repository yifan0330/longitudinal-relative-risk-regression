#!/usr/bin/env python3
"""Plot voxel-wise exchangeable alpha and dispersion phi for GEE fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
import nibabel as nib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

from UKB_validation.ukb_python_experiment import (
    COEFFICIENT_NAMES,
    DEFAULT_ANATOMICAL,
    DEFAULT_PYTHON_RESULTS_DIR,
    DEFAULT_UKB_DIR,
    MODEL_NAMES,
    default_n_jobs,
    ensure_model_outputs,
    load_ukb_design,
    model_is_poisson,
    model_result_dir,
)
from UKB_validation.mapping import values_to_map as _values_to_map


DEFAULT_OUTPUT = SCRIPT_DIR / "figure5_rr_pgee_alpha_map.png"
DEFAULT_SLICES = (40, 45, 50)
DEFAULT_CHUNK_SIZE = 512


def parse_args() -> argparse.Namespace:
    """Parse model, map, and output options for Figure 5."""
    parser = argparse.ArgumentParser(
        description="Plot voxel-wise GEE exchangeable alpha and dispersion phi maps."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument("--model", choices=MODEL_NAMES, default="rr-pgee")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF path; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=DEFAULT_SLICES,
        help="Zero-based axial slice indices to display.",
    )
    parser.add_argument("--alpha-vmin", type=float, default=-1.0)
    parser.add_argument("--alpha-vmax", type=float, default=1.0)
    parser.add_argument("--phi-vmin", type=float, default=0.0)
    parser.add_argument("--phi-vmax", type=float, default=2.0)
    parser.add_argument("--overlay-alpha", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning model fits.",
    )
    parser.add_argument(
        "--write-nifti",
        action="store_true",
        help="Also write alpha_GEE.nii.gz and phi_GEE.nii.gz to the model result directory.",
    )
    args = parser.parse_args()
    args.result_dir = args.result_dir or model_result_dir(args.python_results_dir, args.model)
    if args.output == DEFAULT_OUTPUT and args.model != "rr-pgee":
        args.output = SCRIPT_DIR / f"figure5_{args.model.replace('-', '_')}_alpha_map.png"
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate Figure 5 paths, scales, and worker settings."""
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("anatomical image", args.anatomical),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.slices:
        raise ValueError("At least one axial slice is required")
    if args.dpi <= 0 or args.chunk_size <= 0 or args.n_jobs <= 0:
        raise ValueError("--dpi, --chunk-size, and --n-jobs must be positive")
    if not 0 < args.overlay_alpha <= 1:
        raise ValueError("--overlay-alpha must lie in (0, 1]")
    if args.alpha_vmin >= args.alpha_vmax or args.phi_vmin >= args.phi_vmax:
        raise ValueError("Color scale minima must be smaller than maxima")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")


def load_beta_matrix(result_dir: Path, anatomical: nib.Nifti1Image, voxel_ids: np.ndarray) -> np.ndarray:
    """Load aligned coefficient maps at the analysis-mask voxels."""
    beta = np.full((voxel_ids.size, len(COEFFICIENT_NAMES)), np.nan, dtype=float)
    for index, coefficient_name in enumerate(COEFFICIENT_NAMES):
        path = result_dir / f"estimate_{coefficient_name}_GEE.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Coefficient map not found: {path}")
        image = nib.load(path)
        if image.shape != anatomical.shape or not np.allclose(image.affine, anatomical.affine):
            raise ValueError(f"Coefficient map is not aligned with anatomical image: {path}")
        data = np.asarray(image.get_fdata(), dtype=float)
        beta[:, index] = data.ravel(order="F")[voxel_ids - 1]
    return beta


def load_convergence_mask(result_dir: Path, n_voxels: int) -> np.ndarray:
    """Return the converged, non-failed voxel mask from fit metadata."""
    path = result_dir / "fit_summary.npz"
    if not path.is_file():
        return np.ones(n_voxels, dtype=bool)
    with np.load(path) as summary:
        converged = np.asarray(summary["converged"], dtype=bool)
        failed = np.asarray(summary["failed"], dtype=bool)
    if converged.size != n_voxels or failed.size != n_voxels:
        raise ValueError(f"fit_summary.npz does not match voxel count in {result_dir}")
    return converged & ~failed


def estimate_alpha_phi(
    beta: np.ndarray,
    design,
    valid_mask: np.ndarray,
    *,
    model: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate exchangeable correlation alpha and dispersion phi by voxel."""
    n_voxels = beta.shape[0]
    n_obs = design.n_subjects * 2
    df = max(n_obs - design.n_coefficients, 1)
    alpha = np.full(n_voxels, np.nan, dtype=float)
    phi = np.full(n_voxels, np.nan, dtype=float)
    X = design.X
    poisson = model_is_poisson(model)

    for start in range(0, n_voxels, chunk_size):
        stop = min(start + chunk_size, n_voxels)
        chunk_valid = valid_mask[start:stop] & np.all(np.isfinite(beta[start:stop]), axis=1)
        if not np.any(chunk_valid):
            continue

        beta_chunk = beta[start:stop, :]
        eta = np.clip(X @ beta_chunk.T, -30.0, 30.0)
        if poisson:
            mu = np.exp(eta)
            variance = np.clip(mu, 1e-10, None)
        else:
            mu = 1.0 / (1.0 + np.exp(-eta))
            variance = np.clip(mu * (1.0 - mu), 1e-10, None)
        y = np.stack(
            (design.lesions1[start:stop].T, design.lesions2[start:stop].T),
            axis=1,
        ).reshape(n_obs, stop - start)
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = (y - mu) / np.sqrt(variance)
            pearson = np.sum(residuals**2, axis=0) / df
            phi_chunk = 1.0 / np.maximum(pearson, 1e-12)
            residual_pairs = residuals.reshape(design.n_subjects, 2, stop - start)
            alpha_chunk = phi_chunk * np.mean(
                residual_pairs[:, 0, :] * residual_pairs[:, 1, :], axis=0
            )
        alpha_chunk = np.clip(alpha_chunk, -0.95, 0.95)
        finite = chunk_valid & np.isfinite(alpha_chunk) & np.isfinite(phi_chunk)
        alpha[start:stop][finite] = alpha_chunk[finite]
        phi[start:stop][finite] = phi_chunk[finite]
    return alpha, phi


def values_to_map(values: np.ndarray, voxel_ids: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Place mask values into a 3D map using the UKB voxel convention."""
    return _values_to_map(values, voxel_ids, shape)


def write_nifti(values: np.ndarray, voxel_ids: np.ndarray, template: nib.Nifti1Image, output: Path) -> None:
    """Write mask values as a NIfTI image aligned to a template."""
    data = values_to_map(values, voxel_ids, template.shape).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, template.affine, template.header), str(output))


def brain_crop(anatomical: np.ndarray, slices: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return common plot limits covering the requested anatomical slices."""
    masks = []
    for slice_index in slices:
        masks.append(anatomical[:, :, slice_index].T > 0)
    positive = np.argwhere(np.any(np.stack(masks, axis=0), axis=0))
    if positive.size == 0:
        return (0, anatomical.shape[0] - 1), (0, anatomical.shape[1] - 1)
    margin = 2
    row_min, column_min = positive.min(axis=0)
    row_max, column_max = positive.max(axis=0)
    x_limits = (max(int(column_min) - margin, 0), min(int(column_max) + margin, anatomical.shape[0] - 1))
    y_limits = (max(int(row_min) - margin, 0), min(int(row_max) + margin, anatomical.shape[1] - 1))
    return x_limits, y_limits


def paired_output_paths(output: Path, pdf_output: Path | None) -> tuple[Path, Path, Path, Path]:
    """Return paired PNG and PDF paths for alpha and phi maps."""
    def phi_path(path: Path) -> Path:
        """Derive the matching phi filename from an alpha filename."""
        if "alpha_map" in path.stem:
            return path.with_name(path.stem.replace("alpha_map", "phi_map") + path.suffix)
        if "alpha_phi_maps" in path.stem:
            return path.with_name(path.stem.replace("alpha_phi_maps", "phi_map") + path.suffix)
        return path.with_name(f"{path.stem}_phi{path.suffix}")

    alpha_png = output
    phi_png = phi_path(output)
    alpha_pdf = pdf_output or output.with_suffix(".pdf")
    phi_pdf = phi_path(alpha_pdf)
    return alpha_png, phi_png, alpha_pdf, phi_pdf


def stratified_scale(boundaries: tuple[float, ...]):
    """Create the discrete color scale used for an alpha or phi map."""
    cmap = plt.get_cmap("RdBu_r", len(boundaries) - 1).copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def plot_single_map(
    anatomical_img: nib.Nifti1Image,
    values: np.ndarray,
    args: argparse.Namespace,
    *,
    label: str,
    boundaries: tuple[float, ...],
    ticks: tuple[float, ...],
    output: Path,
    pdf_output: Path,
) -> None:
    """Render one parameter map across the requested axial slices."""
    anatomical = np.asarray(anatomical_img.get_fdata(), dtype=float)
    slices = tuple(int(index) for index in args.slices)
    if any(index < 0 or index >= anatomical.shape[2] for index in slices):
        raise ValueError(f"Slices must be between 0 and {anatomical.shape[2] - 1}")

    positive = anatomical[anatomical > 0]
    anatomical_vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    x_limits, y_limits = brain_crop(anatomical, slices)

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

    cmap, normalization = stratified_scale(boundaries)
    background_axis = figure.add_subplot(grid[0, :3])
    background_axis.set_facecolor("black")
    background_axis.set_xticks([])
    background_axis.set_yticks([])
    background_axis.set_zorder(0)
    for spine in background_axis.spines.values():
        spine.set_visible(False)

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
            norm=normalization,
            origin="lower",
            interpolation="nearest",
            alpha=args.overlay_alpha,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if local_column == 0:
            axis.text(
                0.015,
                0.985,
                label,
                transform=axis.transAxes,
                color="white",
                fontsize=13,
                ha="left",
                va="top",
            )

    colorbar_axis = figure.add_subplot(grid[0, 3])
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=cmap),
        cax=colorbar_axis,
        boundaries=boundaries,
        ticks=ticks,
        spacing="proportional",
    )
    colorbar.ax.tick_params(labelsize=8, length=3, pad=2)
    colorbar.outline.set_linewidth(0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Saved PNG: {output}")
    print(f"Saved PDF: {pdf_output}")


def main() -> None:
    """Estimate or load model diagnostics and write Figure 5 maps."""
    args = parse_args()
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
    anatomical_img = nib.load(args.anatomical)
    design = load_ukb_design(args.ukb_dir, max_voxels=args.max_voxels)
    beta = load_beta_matrix(args.result_dir, anatomical_img, design.voxel_ids)
    valid_mask = load_convergence_mask(args.result_dir, design.n_voxels)
    alpha, phi = estimate_alpha_phi(
        beta,
        design,
        valid_mask,
        model=args.model,
        chunk_size=args.chunk_size,
    )
    if args.write_nifti:
        write_nifti(alpha, design.voxel_ids, anatomical_img, args.result_dir / "alpha_GEE.nii.gz")
        write_nifti(phi, design.voxel_ids, anatomical_img, args.result_dir / "phi_GEE.nii.gz")
    alpha_map = values_to_map(alpha, design.voxel_ids, anatomical_img.shape)
    phi_map = values_to_map(phi, design.voxel_ids, anatomical_img.shape)
    print(
        f"{args.model.upper()} finite alpha/phi voxels: "
        f"{np.count_nonzero(np.isfinite(alpha)):,}/{design.n_voxels:,}; "
        f"alpha range=({np.nanmin(alpha):.3f}, {np.nanmax(alpha):.3f}), "
        f"phi range=({np.nanmin(phi):.3f}, {np.nanmax(phi):.3f})"
    )
    alpha_png, phi_png, alpha_pdf, phi_pdf = paired_output_paths(args.output, args.pdf_output)
    plot_single_map(
        anatomical_img,
        alpha_map,
        args,
        label=r"$\alpha$",
        boundaries=(args.alpha_vmin, -0.5, 0.0, 0.5, args.alpha_vmax),
        ticks=(args.alpha_vmin, -0.5, 0.0, 0.5, args.alpha_vmax),
        output=alpha_png,
        pdf_output=alpha_pdf,
    )
    plot_single_map(
        anatomical_img,
        phi_map,
        args,
        label=r"$\phi$",
        boundaries=(args.phi_vmin, 0.5, 1.0, 1.5, args.phi_vmax),
        ticks=(args.phi_vmin, 0.5, 1.0, 1.5, args.phi_vmax),
        output=phi_png,
        pdf_output=phi_pdf,
    )


if __name__ == "__main__":
    main()
