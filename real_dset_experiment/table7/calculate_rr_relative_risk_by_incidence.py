#!/usr/bin/env python3
"""Summarise RR-PGEE relative risks by empirical visit-1 lesion incidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "table7" else SCRIPT_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ukb_python_experiment import (  # noqa: E402
    DEFAULT_ANATOMICAL,
    DEFAULT_PYTHON_RESULTS_DIR,
    DEFAULT_UKB_DIR,
    default_n_jobs,
    ensure_model_outputs,
    model_result_dir,
)


DEFAULT_VOXEL_IDS = PROJECT_DIR / "UKB" / "voxel_IDs_CVR.dat"
DEFAULT_MODELS = ("rr-pgee", "rr-gee", "or-gee", "or-pgee")
MODEL_SPECS = {
    "rr-pgee": {
        "label": "RR-PGEE",
        "quantity": "Relative risk",
        "caption_quantity": "relative risks",
        "csv_name": "rr_pgee_relative_risk_by_incidence.csv",
        "latex_name": "table7_RR_PGEE_relative_risk_by_incidence.tex",
        "table_label": "tab:rr-pgee-relative-risk-by-incidence",
    },
    "rr-gee": {
        "label": "RR-GEE",
        "quantity": "Relative risk",
        "caption_quantity": "relative risks",
        "csv_name": "rr_gee_relative_risk_by_incidence.csv",
        "latex_name": "table7_RR_GEE_relative_risk_by_incidence.tex",
        "table_label": "tab:rr-gee-relative-risk-by-incidence",
    },
    "or-gee": {
        "label": "OR-GEE",
        "quantity": "Odds ratio",
        "caption_quantity": "odds ratios",
        "csv_name": "or_gee_odds_ratio_by_incidence.csv",
        "latex_name": "table7_OR_GEE_odds_ratio_by_incidence.tex",
        "table_label": "tab:or-gee-odds-ratio-by-incidence",
    },
    "or-pgee": {
        "label": "OR-PGEE",
        "quantity": "Odds ratio",
        "caption_quantity": "odds ratios",
        "csv_name": "or_pgee_odds_ratio_by_incidence.csv",
        "latex_name": "table7_OR_PGEE_odds_ratio_by_incidence.tex",
        "table_label": "tab:or-pgee-odds-ratio-by-incidence",
    },
}
P1_BINS = (
    (0.0, 0.0025, "[0; 0.0025)"),
    (0.0025, 0.005, "[0.0025; 0.005)"),
    (0.005, 0.01, "[0.005; 0.01)"),
    (0.01, 1.0, "[0.01; 1)"),
)
PREDICTORS = (
    ("baseAge", "Age (visit 1)"),
    ("ageDiff", "Time difference"),
    ("baseCVR", "CVR (visit 1)"),
    ("CVRdiff", "CVR difference"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Table 7: RR-PGEE relative risks by empirical lesion incidence."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*DEFAULT_MODELS, "all"),
        default=("all",),
        help="Models to summarize. Use 'all' for RR-PGEE, RR-GEE, OR-GEE, and OR-PGEE.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Optional fitted-map directory; only valid when summarizing one model.",
    )
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument("--voxel-ids", type=Path, default=DEFAULT_VOXEL_IDS)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path; only valid when summarizing one model.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Optional LaTeX output path; only valid when summarizing one model.",
    )
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning RR-PGEE fits.",
    )
    return parser.parse_args()


def resolve_models(models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return DEFAULT_MODELS if "all" in models else tuple(models)


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("voxel ID file", args.voxel_ids),
        ("anatomical image", args.anatomical),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    models = resolve_models(args.models)
    if args.result_dir is not None and len(models) != 1:
        raise ValueError("--result-dir can only be used with a single model")
    if args.output is not None and len(models) != 1:
        raise ValueError("--output can only be used with a single model")
    if args.latex_output is not None and len(models) != 1:
        raise ValueError("--latex-output can only be used with a single model")
    if args.result_dir is not None and not args.result_dir.exists():
        raise FileNotFoundError(f"Fitted-map directory not found: {args.result_dir}")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")


def load_voxel_ids(path: Path, max_voxels: int | None = None) -> np.ndarray:
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0 or np.any(voxel_ids < 1):
        raise ValueError("Voxel IDs must be nonempty positive one-based indices")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError("Voxel IDs must be unique")
    if max_voxels is not None:
        voxel_ids = voxel_ids[:max_voxels]
    return voxel_ids


def load_empirical_values(
    ukb_dir: Path, voxel_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(ukb_dir / "lesions_atleast6_CVR.npz") as lesion_data:
        visit_1 = np.asarray(lesion_data["lesions_vis1"], dtype=float)
        visit_2 = np.asarray(lesion_data["lesions_vis2"], dtype=float)
    if visit_1.shape != visit_2.shape:
        raise ValueError("Visit lesion matrices must have matching shapes")
    if visit_1.shape[0] < voxel_ids.size:
        raise ValueError("Lesion matrices contain fewer voxels than voxel IDs")
    visit_1 = visit_1[: voxel_ids.size]
    visit_2 = visit_2[: voxel_ids.size]
    return visit_1.mean(axis=1), visit_2.mean(axis=1) - visit_1.mean(axis=1), visit_1.shape[1]


def load_exponentiated_estimates(
    result_dir: Path, voxel_ids: np.ndarray, anatomical: Path
) -> dict[str, np.ndarray]:
    template = nib.load(anatomical)
    estimates: dict[str, np.ndarray] = {}
    for predictor, _ in PREDICTORS:
        path = result_dir / f"estimate_{predictor}_GEE.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Coefficient map not found: {path}")
        image = nib.load(path)
        if image.shape != template.shape or not np.allclose(image.affine, template.affine):
            raise ValueError(f"Coefficient map is not aligned with anatomical image: {path}")
        beta_values = np.asarray(image.get_fdata(), dtype=float).ravel(order="F")[voxel_ids - 1]
        with np.errstate(over="ignore", invalid="ignore"):
            estimates[predictor] = np.exp(beta_values)
    return estimates


def load_fit_mask(result_dir: Path, n_voxels: int) -> tuple[np.ndarray, int, int]:
    summary_path = result_dir / "fit_summary.npz"
    if not summary_path.is_file():
        return np.ones(n_voxels, dtype=bool), 0, 0
    with np.load(summary_path) as summary:
        converged = np.asarray(summary["converged"], dtype=bool).reshape(-1)
        failed = np.asarray(summary["failed"], dtype=bool).reshape(-1)
    if converged.size < n_voxels or failed.size < n_voxels:
        raise ValueError(f"fit_summary.npz has fewer voxels than expected: {summary_path}")
    converged = converged[:n_voxels]
    failed = failed[:n_voxels]
    return converged & ~failed, int(np.count_nonzero(failed)), int(np.count_nonzero(~converged & ~failed))


def mean_sd(values: np.ndarray, decimals: int) -> str:
    values = np.asarray(values, dtype=float)
    return f"{np.mean(values):.{decimals}f} ({np.std(values, ddof=1):.{decimals}f})"


def summarise_table(
    p1: np.ndarray,
    p2_minus_p1: np.ndarray,
    exponentiated_estimates: dict[str, np.ndarray],
    fit_mask: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    finite = fit_mask & np.isfinite(p1) & np.isfinite(p2_minus_p1)
    for values in exponentiated_estimates.values():
        finite &= np.isfinite(values)

    rows: list[dict[str, str | int]] = []
    for lower, upper, label in (*P1_BINS, (0.0, 1.0, "[0; 1)")):
        in_bin = finite & (p1 >= lower) & (p1 < upper)
        row: dict[str, str | int] = {
            "p1": label,
            "voxels": int(np.count_nonzero(in_bin)),
            "p2_minus_p1": mean_sd(p2_minus_p1[in_bin], 4),
        }
        for predictor, _ in PREDICTORS:
            row[predictor] = mean_sd(exponentiated_estimates[predictor][in_bin], 2)
        rows.append(row)
    return pd.DataFrame(rows), int(np.count_nonzero(~finite))


def write_latex_table(
    table: pd.DataFrame,
    output: Path,
    model_label: str,
    quantity: str,
    table_label: str,
    n_subjects: int,
    n_regressions: int,
    n_excluded: int,
) -> None:
    body_rows = []
    for index, row in enumerate(table.itertuples(index=False)):
        if index == len(table) - 1:
            body_rows.append(r"\hline")
        body_rows.append(
            " & ".join(
                (
                    row.p1,
                    f"{row.voxels:,}",
                    row.p2_minus_p1,
                    row.baseAge,
                    row.ageDiff,
                    row.baseCVR,
                    row.CVRdiff,
                )
            )
            + r" \\" 
        )
    body = "\n".join(body_rows)
    excluded_phrase = "voxel" if n_excluded == 1 else "voxels"
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{quantity}s across empirical lesion incidence $p_1$ at visit 1 based on {model_label} estimates $\\tilde{{\\beta}}$. Mean and standard deviation (SD) are taken across voxels within each bin as defined in column 1. Data on {n_subjects:,} subjects across two visits and {n_regressions:,} regressions performed, excluding {n_excluded:,} non-converged, failed, or non-finite {excluded_phrase}.}}
\\label{{{table_label}}}
\\setlength{{\\tabcolsep}}{{5pt}}
\\renewcommand{{\\arraystretch}}{{1.12}}
\\begin{{tabular}}{{|l|r|r|rrrr|}}
\\hline
$p_1$ & \\# voxels & \\multicolumn{{5}}{{c|}}{{Mean (SD)}} \\\\
\\cline{{3-7}}
& & $p_2 - p_1$ & \\multicolumn{{4}}{{c|}}{{{quantity} $\\exp(\\tilde{{\\beta}})$}} \\\\
\\cline{{4-7}}
& & & Age (visit 1) & Time difference & CVR (visit 1) & CVR difference \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="ascii")


def output_paths(args: argparse.Namespace, model: str) -> tuple[Path, Path]:
    spec = MODEL_SPECS[model]
    csv_output = args.output or args.output_dir / str(spec["csv_name"])
    latex_output = args.latex_output or args.output_dir / str(spec["latex_name"])
    return csv_output, latex_output


def summarise_model(
    args: argparse.Namespace,
    model: str,
    voxel_ids: np.ndarray,
    p1: np.ndarray,
    p2_minus_p1: np.ndarray,
    n_subjects: int,
) -> None:
    result_dir = ensure_model_outputs(
        model,
        ukb_dir=args.ukb_dir,
        results_root=args.python_results_dir,
        result_dir=args.result_dir or model_result_dir(args.python_results_dir, model),
        anatomical=args.anatomical,
        n_jobs=args.n_jobs,
        max_voxels=args.max_voxels,
        force_rerun=not args.use_cache,
    )
    exponentiated_estimates = load_exponentiated_estimates(result_dir, voxel_ids, args.anatomical)
    fit_mask, n_failed, n_not_converged = load_fit_mask(result_dir, voxel_ids.size)
    table, n_excluded = summarise_table(p1, p2_minus_p1, exponentiated_estimates, fit_mask)
    csv_output, latex_output = output_paths(args, model)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_output, index=False)

    spec = MODEL_SPECS[model]
    write_latex_table(
        table,
        latex_output,
        str(spec["label"]),
        str(spec["quantity"]),
        str(spec["table_label"]),
        n_subjects,
        voxel_ids.size,
        n_excluded,
    )

    print(f"\n{spec['label']}:")
    print(table.to_string(index=False))
    print(f"Analysis regressions: {voxel_ids.size:,}")
    print(
        f"Excluded {spec['label']} voxels: {n_excluded:,} "
        f"({n_failed:,} failed; {n_not_converged:,} non-converged; "
        "remaining excluded were non-finite)"
    )
    print(f"Saved: {csv_output}")
    print(f"Saved LaTeX: {latex_output}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    models = resolve_models(args.models)
    voxel_ids = load_voxel_ids(args.voxel_ids, max_voxels=args.max_voxels)
    p1, p2_minus_p1, n_subjects = load_empirical_values(args.ukb_dir, voxel_ids)
    for model in models:
        summarise_model(args, model, voxel_ids, p1, p2_minus_p1, n_subjects)


if __name__ == "__main__":
    main()
