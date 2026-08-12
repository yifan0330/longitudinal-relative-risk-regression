#!/usr/bin/env python3
"""Plot UKB Figure B3 BEC histograms for RR-GEE and RR-PGEE."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent

from real_dset_experiment.ukb_python_experiment import COEFFICIENT_NAMES, load_ukb_design
from real_dset_experiment.paths import DEFAULT_PYTHON_RESULTS_DIR


DEFAULT_RESULTS_ROOT = DEFAULT_PYTHON_RESULTS_DIR
DEFAULT_OUTPUT = HERE / "figureB3_ukb_bec_threshold.png"
DEFAULT_CACHE = HERE / "figureB3_bec_values.npz"
DEFAULT_SUMMARY = HERE / "figureB3_ukb_bec_threshold_summary.csv"
METHODS = ("rr_gee", "rr_pgee")
METHOD_LABELS = {"rr_gee": "RR-GEE", "rr_pgee": "RR-PGEE"}
SEX_COEFFICIENT = "sexM"
CHUNK_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure B3: UKB BEC histograms for RR-GEE and RR-PGEE."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output; by default a PDF is saved beside the PNG.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--bin-width", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.results_root.exists():
        raise FileNotFoundError(f"Results root not found: {args.results_root}")
    if args.x_max <= 0 or args.threshold <= 0 or args.bin_width <= 0 or args.dpi <= 0:
        raise ValueError("--x-max, --threshold, --bin-width, and --dpi must be positive")


def load_nifti_values(path: Path, voxel_ids: np.ndarray) -> np.ndarray:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    coordinates = np.unravel_index(voxel_ids - 1, image.shape, order="F")
    return np.asarray(data[coordinates], dtype=float)


def load_method_arrays(
    results_root: Path,
    method: str,
    voxel_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method_dir = results_root / method
    beta = np.column_stack(
        [
            load_nifti_values(method_dir / f"estimate_{name}_GEE.nii.gz", voxel_ids)
            for name in COEFFICIENT_NAMES
        ]
    )
    alpha = load_nifti_values(method_dir / "alpha_GEE.nii.gz", voxel_ids)
    phi = load_nifti_values(method_dir / "phi_GEE.nii.gz", voxel_ids)
    return beta, alpha, phi


def safe_inverse_stack(matrices: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.pinv(matrix, hermitian=True) for matrix in matrices])


def model_se_from_information(
    x_clusters: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    n_voxels, n_coefficients = beta.shape
    output = np.full((n_voxels, n_coefficients), np.nan, dtype=float)
    finite_rows = np.all(np.isfinite(beta), axis=1)
    if not np.any(finite_rows):
        return output

    beta_finite = beta[finite_rows]
    alpha_finite = np.nan_to_num(alpha[finite_rows], nan=0.0, posinf=0.95, neginf=-0.95)
    alpha_finite = np.clip(alpha_finite, -0.95, 0.95)
    phi_finite = np.nan_to_num(phi[finite_rows], nan=1.0, posinf=1.0, neginf=1.0)
    phi_finite = np.where(phi_finite > 0.0, phi_finite, 1.0)

    n_subjects, n_visits, _ = x_clusters.shape
    x = x_clusters.reshape(n_subjects * n_visits, n_coefficients)
    eta = np.clip(x @ beta_finite.T, -30.0, 30.0)
    mu = np.exp(eta).T.reshape(beta_finite.shape[0], n_subjects, n_visits)

    determinant = 1.0 - alpha_finite**2
    r00 = 1.0 / determinant
    r01 = -alpha_finite / determinant
    r11 = r00
    information = np.zeros((beta_finite.shape[0], n_coefficients, n_coefficients), dtype=float)
    for subject_index in range(n_subjects):
        subject_mu = mu[:, subject_index, :]
        a00 = phi_finite * subject_mu[:, 0] * r00
        a01 = phi_finite * np.sqrt(subject_mu[:, 0] * subject_mu[:, 1]) * r01
        a11 = phi_finite * subject_mu[:, 1] * r11
        x0 = x_clusters[subject_index, 0]
        x1 = x_clusters[subject_index, 1]
        information += a00[:, None, None] * np.outer(x0, x0)
        information += a01[:, None, None] * (np.outer(x0, x1) + np.outer(x1, x0))
        information += a11[:, None, None] * np.outer(x1, x1)

    scale = np.trace(information, axis1=1, axis2=2) / n_coefficients
    information += np.maximum(scale, 1.0)[:, None, None] * 1e-8 * np.eye(n_coefficients)
    information_inverse = safe_inverse_stack(information)
    output[finite_rows] = np.sqrt(
        np.clip(np.diagonal(information_inverse, axis1=1, axis2=2), 0.0, None)
    )
    return output


def first_iteration_model_se(x_clusters: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    n_subjects, n_visits, n_coefficients = x_clusters.shape
    n_voxels = outcomes.shape[1]
    x = x_clusters.reshape(n_subjects * n_visits, n_coefficients)
    output = np.full((n_voxels, n_coefficients), np.nan, dtype=float)

    for start in range(0, n_voxels, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, n_voxels)
        y = outcomes[:, start:stop]
        beta = np.zeros((stop - start, n_coefficients), dtype=float)
        beta[:, 0] = np.log(np.maximum(y.mean(axis=0), 1e-6))
        eta = np.clip(x @ beta.T, -30.0, 30.0)
        mu = np.exp(eta)
        information = np.einsum("nv,ni,nj->vij", mu, x, x, optimize=True)
        scale = np.trace(information, axis1=1, axis2=2) / n_coefficients
        information += np.maximum(scale, 1.0)[:, None, None] * 1e-8 * np.eye(n_coefficients)
        score = x.T @ (y - mu)
        try:
            step = np.linalg.solve(information, score.T[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            information_inverse = safe_inverse_stack(information)
            step = np.einsum("vij,vj->vi", information_inverse, score.T, optimize=True)
        beta += np.clip(step, -2.0, 2.0)

        eta = np.clip(x @ beta.T, -30.0, 30.0)
        mu = np.exp(eta).T.reshape(stop - start, n_subjects, n_visits)
        residual = (y.T.reshape(stop - start, n_subjects, n_visits) - mu) / np.sqrt(
            np.clip(mu, 1e-12, None)
        )
        phi = (n_subjects * n_visits - n_coefficients) / np.maximum(
            np.sum(residual**2, axis=(1, 2)),
            1e-12,
        )
        alpha = np.clip(phi * np.sum(residual[:, :, 0] * residual[:, :, 1], axis=1) / n_subjects, -0.95, 0.95)
        output[start:stop] = model_se_from_information(x_clusters, beta, alpha, phi)

    return output


def compute_bec_values(args: argparse.Namespace) -> dict[str, np.ndarray]:
    design = load_ukb_design()
    outcomes = np.stack((design.lesions1.T, design.lesions2.T), axis=1).reshape(
        design.n_subjects * 2,
        design.n_voxels,
    )
    initial_se = first_iteration_model_se(design.X_clusters, outcomes)
    bec_values: dict[str, np.ndarray] = {}

    for method in METHODS:
        beta, alpha, phi = load_method_arrays(args.results_root, method, design.voxel_ids)
        final_se = np.full_like(initial_se, np.nan)
        for start in range(0, design.n_voxels, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, design.n_voxels)
            final_se[start:stop] = model_se_from_information(
                design.X_clusters,
                beta[start:stop],
                alpha[start:stop],
                phi[start:stop],
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            bec_values[method] = final_se / initial_se

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        coefficient_names=np.asarray(COEFFICIENT_NAMES, dtype=str),
        **{method: bec_values[method] for method in METHODS},
    )
    return bec_values


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
        1.042,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        clip_on=False,
    )


def plot_figure(bec_values: dict[str, np.ndarray], args: argparse.Namespace) -> pd.DataFrame:
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

    sex_index = COEFFICIENT_NAMES.index(SEX_COEFFICIENT)
    bins = np.arange(0.0, args.x_max + args.bin_width, args.bin_width)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    summary_rows: list[dict[str, object]] = []

    for index, (axis, method) in enumerate(zip(axes, METHODS)):
        bec = bec_values[method]
        sex_bec = bec[:, sex_index]
        finite_all = np.all(np.isfinite(bec), axis=1)
        any_above = np.any(bec > args.threshold, axis=1)
        shown = sex_bec[finite_all & ~any_above & (sex_bec >= 0.0) & (sex_bec <= args.x_max)]
        axis.hist(shown, bins=bins, color="0.28", edgecolor="white", linewidth=0.35)
        axis.axvline(args.threshold, color="0.05", linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
        axis.text(
            0.965,
            0.9,
            f"BEC > {args.threshold:g}: {np.count_nonzero(any_above):,}",
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

        summary_rows.append(
            {
                "method": METHOD_LABELS[method],
                "n_voxels": int(bec.shape[0]),
                "finite_all_coefficients": int(np.count_nonzero(finite_all)),
                "shown_in_histogram": int(shown.size),
                "bec_gt_10_any_coefficient": int(np.count_nonzero(any_above)),
                "missing_any_coefficient": int(np.count_nonzero(~finite_all)),
                "sex_bec_gt_10": int(np.count_nonzero(sex_bec > args.threshold)),
                "sex_bec_median": float(np.nanmedian(sex_bec)),
                "sex_bec_max": float(np.nanmax(sex_bec)),
            }
        )

    figure.supxlabel(r"Boundary estimates criterion, BEC($\beta_6$)", fontsize=11.5, y=0.045)
    figure.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.83, wspace=0.08)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)

    summary = pd.DataFrame(summary_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {pdf_output}")
    print(f"Saved BEC cache: {args.cache}")
    print(f"Saved summary: {args.summary}")
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    bec_values = compute_bec_values(args)
    plot_figure(bec_values, args)


if __name__ == "__main__":
    main()