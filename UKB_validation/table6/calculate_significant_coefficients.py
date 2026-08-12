#!/usr/bin/env python3
"""Count significant RR-GEE and RR-PGEE coefficients across CVR voxels."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent

from UKB_validation.ukb_python_experiment import (
    DEFAULT_ANATOMICAL,
    DEFAULT_PYTHON_RESULTS_DIR,
    DEFAULT_UKB_DIR,
    RR_GEE_CHUNK_SIZE,
    default_n_jobs,
    ensure_model_outputs,
)
from UKB_validation.paths import DEFAULT_UKB_DIR


DEFAULT_VOXEL_IDS = DEFAULT_UKB_DIR / "voxel_IDs_CVR.dat"
DEFAULT_OUTPUT = SCRIPT_DIR / "significant_coefficient_counts.csv"
DEFAULT_LATEX_OUTPUT = SCRIPT_DIR / "table6_RR_significant_voxels.tex"

PREDICTORS = (
    ("baseAge", "Age (visit 1)"),
    ("ageDiff", "Time difference"),
    ("baseCVR", "CVR score (visit 1)"),
    ("CVRdiff", "CVR score difference"),
    ("sexM", "Sex"),
    ("headsize", "Head size"),
    ("ageBYageDiff", "Age (visit 1):Time difference"),
    ("ageBYsexM", "Age (visit 1):Sex"),
)


def parse_args() -> argparse.Namespace:
    """Parse RR significance-counting and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Count significant coefficient z statistics for RR-GEE and RR-PGEE "
            "within the CVR voxel set."
        )
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument("--gee-dir", type=Path, default=None)
    parser.add_argument("--pgee-dir", type=Path, default=None)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--voxel-ids", type=Path, default=DEFAULT_VOXEL_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--z-threshold", type=float, default=1.96)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=RR_GEE_CHUNK_SIZE,
        help="RR-GEE voxel chunk size for the vectorized IRLS/Newton solver.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning the model fits.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate RR table paths and multiple-testing parameters."""
    if not 0 < args.fdr_alpha < 1:
        raise ValueError("--fdr-alpha must lie strictly between 0 and 1")
    if args.z_threshold <= 0:
        raise ValueError("--z-threshold must be positive")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("voxel ID file", args.voxel_ids),
        ("anatomical image", args.anatomical),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def load_voxel_ids(path: Path) -> np.ndarray:
    """Load unique positive one-based analysis-mask voxel IDs."""
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0:
        raise ValueError(f"No voxel IDs found in {path}")
    if np.any(voxel_ids < 1):
        raise ValueError("Voxel IDs must be positive one-based indices")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError("Voxel IDs must be unique")
    return voxel_ids


def analysis_voxel_ids(args: argparse.Namespace) -> np.ndarray:
    """Apply the optional voxel limit to the analysis mask."""
    voxel_ids = load_voxel_ids(args.voxel_ids)
    return voxel_ids[: args.max_voxels] if args.max_voxels is not None else voxel_ids


def load_zscores(result_dir: Path, predictor: str, voxel_ids: np.ndarray) -> np.ndarray:
    """Load one predictor’s z-scores at the analysis-mask voxels."""
    path = result_dir / f"zscore_{predictor}_GEE.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Z-statistic map not found: {path}")

    data = np.asarray(nib.load(path).get_fdata(), dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D z-statistic map, got {data.shape}: {path}")
    if voxel_ids.max() > data.size:
        raise ValueError(f"Voxel IDs exceed the image grid in {path}")
    return data.ravel(order="F")[voxel_ids - 1]


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values, treating failed voxel fits as p=1."""
    p_values = np.asarray(p_values, dtype=float)
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


def count_significant(
    zscores: np.ndarray, fdr_alpha: float, z_threshold: float
) -> tuple[int, int, int]:
    """Count FDR-significant, threshold-significant, and finite z-scores."""
    finite = np.isfinite(zscores)
    p_values = np.full(zscores.shape, np.nan, dtype=float)
    p_values[finite] = 2 * stats.norm.sf(np.abs(zscores[finite]))
    fdr_count = int(np.count_nonzero(benjamini_hochberg(p_values) <= fdr_alpha))
    threshold_count = int(np.count_nonzero(finite & (np.abs(zscores) > z_threshold)))
    return fdr_count, threshold_count, int(np.count_nonzero(finite))


def calculate_counts(args: argparse.Namespace) -> pd.DataFrame:
    """Fit or load both RR methods and assemble their count table."""
    voxel_ids = analysis_voxel_ids(args)
    gee_dir = ensure_model_outputs(
        "rr-gee",
        ukb_dir=args.ukb_dir,
        results_root=args.python_results_dir,
        result_dir=args.gee_dir,
        anatomical=args.anatomical,
        n_jobs=args.n_jobs,
        max_voxels=args.max_voxels,
        chunk_size=args.chunk_size,
        force_rerun=not args.use_cache,
    )
    pgee_dir = ensure_model_outputs(
        "rr-pgee",
        ukb_dir=args.ukb_dir,
        results_root=args.python_results_dir,
        result_dir=args.pgee_dir,
        anatomical=args.anatomical,
        n_jobs=args.n_jobs,
        max_voxels=args.max_voxels,
        force_rerun=not args.use_cache,
    )
    methods = (("RR-GEE", gee_dir), ("RR-PGEE", pgee_dir))
    rows: list[dict[str, int | str]] = []

    for predictor, label in PREDICTORS:
        row: dict[str, int | str] = {"Predictor": label}
        for method, result_dir in methods:
            zscores = load_zscores(result_dir, predictor, voxel_ids)
            fdr_count, threshold_count, finite_count = count_significant(
                zscores, args.fdr_alpha, args.z_threshold
            )
            row[f"{method} FDR-corrected"] = fdr_count
            row[f"{method} |z| > {args.z_threshold:g}"] = threshold_count
            row[f"{method} finite z"] = finite_count
        rows.append(row)

    columns = [
        "Predictor",
        "RR-GEE FDR-corrected",
        "RR-PGEE FDR-corrected",
        f"RR-GEE |z| > {args.z_threshold:g}",
        f"RR-PGEE |z| > {args.z_threshold:g}",
        "RR-GEE finite z",
        "RR-PGEE finite z",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_latex_table(
    counts: pd.DataFrame, output: Path, n_voxels: int, z_threshold: float
) -> None:
    """Write the publication-format RR significant-voxels table."""
    threshold_label = f"{z_threshold:g}"
    rows = []
    for row in counts.itertuples(index=False, name=None):
        rows.append(
            f"{row[0]} & {row[1]:,} & {row[2]:,} & {row[3]:,} & {row[4]:,} \\\\"
        )
    latex = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            (
                "\\caption{Number of significant voxels across predictors for RR-GEE "
                f"and RR-PGEE estimates. Voxels with at least six individuals having a lesion "
                f"across two visits were analyzed ({n_voxels:,} voxels in the brain mask). "
                f"A 5\\% FDR correction and a fixed threshold of $|z| > {threshold_label}$ "
                "were applied.}"
            ),
            "\\label{tab:significant-voxels}",
            "\\setlength{\\tabcolsep}{7pt}",
            "\\renewcommand{\\arraystretch}{1.15}",
            "\\begin{tabular}{lrrrr}",
            "\\hline",
            (
                "& \\multicolumn{2}{c}{FDR-corrected voxels} "
                f"& \\multicolumn{{2}}{{c}}{{$|z| > {threshold_label}$}} \\\\"
            ),
            "\\cline{2-3}\\cline{4-5}",
            "Predictor & RR-GEE & RR-PGEE & RR-GEE & RR-PGEE \\\\",
            "\\hline",
            *rows,
            "\\hline",
            "\\end{tabular}",
            "",
            "\\vspace{0.4em}",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            (
                "FDR: false discovery rate; GEE: generalized estimating equations; "
                "PGEE: penalized generalized estimating equations; RR: relative risk."
            ),
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="ascii")


def main() -> None:
    """Calculate and write Table 6 RR significance outputs."""
    args = parse_args()
    validate_args(args)
    voxel_ids = analysis_voxel_ids(args)
    counts = calculate_counts(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output, index=False)
    write_latex_table(counts, args.latex_output, voxel_ids.size, args.z_threshold)

    print(f"Analysis voxels: {voxel_ids.size:,}")
    print(f"FDR: Benjamini-Hochberg at {args.fdr_alpha:g} (two-sided Wald tests)")
    print(counts.drop(columns=["RR-GEE finite z", "RR-PGEE finite z"]).to_string(index=False))
    print(f"\nSaved: {args.output}")
    print(f"Saved LaTeX: {args.latex_output}")


if __name__ == "__main__":
    main()