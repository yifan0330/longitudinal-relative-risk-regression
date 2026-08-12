#!/usr/bin/env python3
"""Calculate and typeset descriptive statistics for the UK Biobank cohort."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
from real_dset_experiment.paths import DEFAULT_UKB_DIR

DEFAULT_INPUT = DEFAULT_UKB_DIR / "CVR_9June2021.pkl"
DEFAULT_OUTPUT = SCRIPT_DIR / "table5_ukb_characteristics.tex"

REQUIRED_COLUMNS = {
    "age_vis2",
    "age_vis3",
    "X31.0.0",
    "X25000.2.0",
    "X25000.3.0",
    "CVR_vis2",
    "CVR_vis3",
    "X25781.2.0",
    "X25781.3.0",
}


def load_ukb_data(path: Path) -> pd.DataFrame:
    """Load the trusted project pickle and return its complete cohort."""
    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "complete_df" not in data:
        raise ValueError(f"{path} must contain a 'complete_df' object")

    frame = data["complete_df"]
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values")
    return frame


def mean_sd(series: pd.Series, decimals: int = 1, comma: bool = False) -> str:
    """Format a mean and sample standard deviation."""
    if comma:
        return f"{series.mean():,.0f} ({series.std(ddof=1):,.0f})"
    return f"{series.mean():.{decimals}f} ({series.std(ddof=1):.{decimals}f})"


def median_range(series: pd.Series, decimals: int = 1, comma: bool = False) -> str:
    """Format a median and observed range."""
    if comma:
        return f"{series.median():,.0f} ({series.min():,.0f}; {series.max():,.0f})"
    return (
        f"{series.median():.{decimals}f} "
        f"({series.min():.{decimals}f}; {series.max():.{decimals}f})"
    )


def build_table(frame: pd.DataFrame) -> str:
    """Build the LaTeX table using the layout of Table 7."""
    age = frame["age_vis2"].astype(float)
    time_between_visits = (
        frame["age_vis3"].astype(float) - frame["age_vis2"].astype(float)
    )
    head_size = (
        frame["X25000.2.0"].astype(float)
        + frame["X25000.3.0"].astype(float)
    ) / 2
    percent_men = 100 * frame["X31.0.0"].astype(float).mean()
    cvr_visit_1 = frame["CVR_vis2"].astype(float)
    cvr_visit_2 = frame["CVR_vis3"].astype(float)
    lesion_visit_1 = frame["X25781.2.0"].astype(float)
    lesion_visit_2 = frame["X25781.3.0"].astype(float)

    rows = [
        ("Age, visit 1 (years)", mean_sd(age), median_range(age)),
        (
            "Time between visits (years)",
            mean_sd(time_between_visits),
            median_range(time_between_visits),
        ),
        ("Sex (baseline female)", f"{percent_men:.1f}\\% Men", "--"),
        ("Head size scaling*", mean_sd(head_size), median_range(head_size)),
        ("CVR score, visit 1", mean_sd(cvr_visit_1), median_range(cvr_visit_1, 0)),
        ("CVR score, visit 2", mean_sd(cvr_visit_2), median_range(cvr_visit_2, 0)),
        (
            "Lesion volume, visit 1 ($\\mathrm{mm}^3$)",
            mean_sd(lesion_visit_1, comma=True),
            median_range(lesion_visit_1, comma=True),
        ),
        (
            "Lesion volume, visit 2 ($\\mathrm{mm}^3$)",
            mean_sd(lesion_visit_2, comma=True),
            median_range(lesion_visit_2, comma=True),
        ),
    ]

    table_rows = []
    for index, row in enumerate(rows):
        if index in {4, 6}:
            table_rows.append(r"\hline")
        table_rows.append(" & ".join(row) + r" \\")

    body = "\n".join(table_rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Characteristics of UK Biobank dataset of {len(frame):,} participants.}}
\\label{{tab:ukb-characteristics}}
\\setlength{{\\tabcolsep}}{{8pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{lrr}}
\\hline
Characteristics & Mean (SD) & Median (range) \\\\
\\hline
{body}
\\hline
\\end{{tabular}}

\\vspace{{0.4em}}
\\begin{{minipage}}{{0.94\\linewidth}}
\\footnotesize
*Average of head size scaling for visits 1 and 2.\\\\
N: number of participants; SD: standard deviation; CVR: cerebrovascular risk.
\\end{{minipage}}
\\end{{table}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate UK Biobank cohort characteristics and write LaTeX."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_ukb_data(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_table(frame), encoding="utf-8")
    print(f"Wrote {args.output} from {len(frame):,} participants")


if __name__ == "__main__":
    main()