"""Build smoke and full-run output tables for the OR-PGEE comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import Scenario, full_scenarios
    from .coverage import add_interval_columns
    from .data_generation import generate_dataset
    from .methods import fit_all_methods
except ImportError:
    from config import Scenario, full_scenarios
    from coverage import add_interval_columns
    from data_generation import generate_dataset
    from methods import fit_all_methods

METHOD_RR_GEE = "RR-GEE"
METHOD_RR_PGEE = "RR-PGEE"
METHOD_OR_GEE = "OR-GEE-ZY"
METHOD_OR_PGEE = "OR-PGEE-ZY"
METHOD_ORDER = (METHOD_RR_GEE, METHOD_RR_PGEE, METHOD_OR_GEE, METHOD_OR_PGEE)
BOUNDARY_CASE_RANDOM_STATE = 20260709
TABLE1_ITERATION_CAPS = (5, 15, 25)
FULL_TABLE_DIRS = {
    "table1_boundary_case": "table1",
    "table2_convergence": "table2",
    "table3_bias_mse": "table3",
    "table3_beta_bias_mse": "table3",
    "table4_variance_coverage": "table4",
    "confounded_summary": "confounded_summary",
}


def build_tables(replications: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Build CSV, Markdown, and LaTeX tables from replication-level output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table1 = _table1_boundary_case(replications)
    table2 = _table2_convergence(replications)
    table3 = _table3_rr_bias_mse(replications)
    table3_beta = _table3_beta_bias_mse(replications)
    table4 = _table4_variance_coverage(replications)
    confounded = table3.merge(
        table3_beta[["scenario", "method", "mean_beta", "bias_beta", "mse_beta"]],
        on=["scenario", "method"],
        how="left",
    ).merge(
        table4[["scenario", "method", "coverage", "coverage_n"]],
        on=["scenario", "method"],
        how="left",
    )
    confounded = confounded[confounded["scenario_type"] == "confounded"]

    tables = {
        "table1_boundary_case": table1,
        "table2_convergence": table2,
        "table3_bias_mse": table3,
        "table3_beta_bias_mse": table3_beta,
        "table4_variance_coverage": table4,
        "confounded_summary": confounded,
    }
    for name, table in tables.items():
        latex = None
        if name == "table1_boundary_case":
            latex = _to_table1_boundary_case_latex(table)
        elif name == "table2_convergence":
            latex = _to_table2_convergence_latex(table)
        elif name == "table3_bias_mse":
            latex = _to_table3_bias_mse_latex(table, scale="rr")
        elif name == "table3_beta_bias_mse":
            latex = _to_table3_bias_mse_latex(table, scale="beta")
        elif name == "table4_variance_coverage":
            latex = _to_table4_variance_coverage_latex(table)
        _write_table(table, _table_stem(output_dir, name), latex=latex)
    return tables


def _table_stem(output_dir: Path, name: str) -> Path:
    if output_dir.name == "full":
        return output_dir / FULL_TABLE_DIRS[name] / name
    return output_dir / name


def _table1_boundary_case(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one reproducibly sampled replication where BEC exceeded 10."""
    method_rows = rows[rows["method"].isin(METHOD_ORDER)].copy()
    candidates = (
        method_rows.assign(bec_gt_10=method_rows["bec_count"] > 10)
        .groupby(["scenario", "replication"], as_index=False)
        .agg(
            n_methods=("method", "nunique"),
            any_bec_gt_10=("bec_gt_10", "any"),
            all_finite=("finite", "all"),
            all_rr_finite=("rr_estimate", lambda x: np.isfinite(x).all()),
        )
    )
    candidates = candidates[
        candidates["any_bec_gt_10"] & (candidates["n_methods"] == len(METHOD_ORDER))
    ]
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "replication",
                "method",
                "max_iter",
                "beta_hat_b",
                "z_score_beta_hat_b",
                "bec_count",
                "status",
            ]
        )

    complete_candidates = candidates[candidates["all_finite"] & candidates["all_rr_finite"]]
    if not complete_candidates.empty:
        candidates = complete_candidates

    selected = candidates.sample(
        n=1,
        random_state=BOUNDARY_CASE_RANDOM_STATE,
    ).iloc[0]
    scenario = selected["scenario"]
    replication = int(selected["replication"])
    case_rows = _table1_refit_rows(str(scenario), replication)
    if case_rows.empty:
        case_rows = method_rows[
            (method_rows["scenario"] == scenario)
            & (method_rows["replication"] == replication)
        ].copy()
        case_rows["max_iter"] = 25
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    case_rows["method_order"] = case_rows["method"].map(order)
    case_rows["z_score_beta_hat_b"] = case_rows["log_effect"] / case_rows["se_log_effect"]
    case_rows["status"] = [
        _table1_status(converged, finite) if int(max_iter) == 25 else ""
        for converged, finite, max_iter in zip(
            case_rows["converged"],
            case_rows["finite"],
            case_rows["max_iter"],
        )
    ]
    case_rows["method"] = case_rows["method"].map(_display_method)
    table = case_rows.sort_values(["method_order", "max_iter"])[
        [
            "scenario",
            "replication",
            "method",
            "max_iter",
            "log_effect",
            "z_score_beta_hat_b",
            "bec_count",
            "status",
        ]
    ].rename(
        columns={
            "log_effect": "beta_hat_b",
        }
    )
    return table.reset_index(drop=True)


def _table1_refit_rows(scenario_name: str, replication: int) -> pd.DataFrame:
    """Refit the selected boundary dataset at the Table 1 iteration caps."""
    scenario = _scenario_by_name(scenario_name)
    if scenario is None:
        return pd.DataFrame()
    data = generate_dataset(scenario, replication).data
    rows: list[dict[str, object]] = []
    for max_iter in TABLE1_ITERATION_CAPS:
        for fit in fit_all_methods(data, scenario, max_iter=max_iter):
            rows.append(
                {
                    "scenario": scenario.name,
                    "scenario_type": scenario.scenario_type,
                    "replication": replication,
                    "max_iter": max_iter,
                    "method": fit.method,
                    "true_beta_b": scenario.true_beta_b,
                    "true_rr": scenario.true_rr,
                    "log_effect": fit.log_effect,
                    "se_log_effect": fit.se_log_effect,
                    "rr_estimate": fit.rr_estimate,
                    "p0_hat": fit.p0_hat,
                    "converged": fit.converged,
                    "finite": fit.finite,
                    "iterations": fit.iterations,
                    "alpha": fit.alpha,
                    "phi": fit.phi,
                    "bec_count": fit.bec_count,
                    "failure_reason": fit.failure_reason,
                }
            )
    return add_interval_columns(pd.DataFrame(rows))


def _scenario_by_name(scenario_name: str) -> Scenario | None:
    for scenario in full_scenarios():
        if scenario.name == scenario_name:
            return scenario
    return None


def _display_method(method: object) -> str:
    return str(method).removesuffix("-ZY")


def _table2_convergence(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["bec_gt_10_flag"] = rows["bec_count"] > 10
    rows["finite_flag"] = rows["finite"].astype(bool)
    rows["converged_flag"] = rows["converged"].astype(bool)
    rows["successful_convergence_flag"] = rows["finite_flag"] & rows["converged_flag"]
    rows["converged_bec_le_10_flag"] = (
        rows["successful_convergence_flag"] & ~rows["bec_gt_10_flag"]
    )
    rows["converged_bec_gt_10_flag"] = (
        rows["successful_convergence_flag"] & rows["bec_gt_10_flag"]
    )
    rows["non_converged_bec_le_10_flag"] = (
        ~rows["successful_convergence_flag"] & ~rows["bec_gt_10_flag"]
    )
    rows["non_converged_bec_gt_10_flag"] = (
        ~rows["successful_convergence_flag"] & rows["bec_gt_10_flag"]
    )
    return (
        rows.groupby(["scenario", "scenario_type", "method"], as_index=False)
        .agg(
            n_replications=("replication", "size"),
            converged_bec_le_10=("converged_bec_le_10_flag", "sum"),
            converged_bec_gt_10=("converged_bec_gt_10_flag", "sum"),
            non_converged_bec_le_10=("non_converged_bec_le_10_flag", "sum"),
            non_converged_bec_gt_10=("non_converged_bec_gt_10_flag", "sum"),
        )
        .sort_values(["scenario", "method"])
    )


def _table3_rr_bias_mse(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["bias_rr"] = rows["rr_estimate"] - rows["true_rr"]
    rows["squared_error_rr"] = rows["bias_rr"] ** 2

    summaries: list[dict[str, object]] = []
    for (scenario, scenario_type), group in rows.groupby(["scenario", "scenario_type"], sort=False):
        common_replications = _common_table3_replications(group)
        for method in METHOD_ORDER:
            method_rows = group[
                (group["method"] == method)
                & group["replication"].isin(common_replications)
            ]
            unconditional_rows = _eligible_no_boundary(group[group["method"] == method])
            summaries.append(
                _table3_rr_summary(
                    scenario,
                    scenario_type,
                    method,
                    method_rows,
                    true_rr=float(group["true_rr"].iloc[0]),
                    unconditional_rows=unconditional_rows,
                )
            )

    return pd.DataFrame(summaries).sort_values(["scenario", "method"])


def _table3_rr_summary(
    scenario: str,
    scenario_type: str,
    method: str,
    rows: pd.DataFrame,
    *,
    true_rr: float,
    unconditional_rows: pd.DataFrame | None = None,
) -> dict[str, object]:
    summary = {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "method": method,
        "n_used": len(rows),
        "true_rr": true_rr,
        "mean_rr": rows["rr_estimate"].mean(),
        "bias_rr": rows["bias_rr"].mean(),
        "mse_rr": rows["squared_error_rr"].mean(),
    }
    if unconditional_rows is not None:
        summary.update(
            {
                "n_unconditional": len(unconditional_rows),
                "bias_rr_unconditional": unconditional_rows["bias_rr"].mean(),
                "mse_rr_unconditional": unconditional_rows["squared_error_rr"].mean(),
            }
        )
    else:
        summary.update(
            {
                "n_unconditional": float("nan"),
                "bias_rr_unconditional": float("nan"),
                "mse_rr_unconditional": float("nan"),
            }
        )
    return summary


def _table3_beta_bias_mse(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["beta_estimate"] = rows["rr_estimate"].apply(_safe_log)
    rows["bias_beta"] = rows["beta_estimate"] - rows["true_beta_b"]
    rows["squared_error_beta"] = rows["bias_beta"] ** 2

    summaries: list[dict[str, object]] = []
    for (scenario, scenario_type), group in rows.groupby(["scenario", "scenario_type"], sort=False):
        common_replications = _common_table3_beta_replications(group)
        for method in METHOD_ORDER:
            method_rows = group[
                (group["method"] == method)
                & group["replication"].isin(common_replications)
            ]
            unconditional_rows = _eligible_no_boundary(group[group["method"] == method])
            summaries.append(
                _table3_beta_summary(
                    scenario,
                    scenario_type,
                    method,
                    method_rows,
                    true_beta_b=float(group["true_beta_b"].iloc[0]),
                    unconditional_rows=unconditional_rows,
                )
            )

    return pd.DataFrame(summaries).sort_values(["scenario", "method"])


def _common_table3_beta_replications(group: pd.DataFrame) -> pd.Index:
    return _common_table3_replications(group)


def _common_table3_replications(group: pd.DataFrame) -> pd.Index:
    method_rows = group[group["method"].isin(METHOD_ORDER)]
    if method_rows.empty:
        return pd.Index([])
    paired = method_rows.pivot(index="replication", columns="method", values=["coverage_eligible", "bec_count"])
    if any(method not in paired["coverage_eligible"] for method in METHOD_ORDER):
        return pd.Index([])
    usable = pd.Series(True, index=paired.index)
    for method in METHOD_ORDER:
        usable &= paired[("coverage_eligible", method)].astype(bool)
        usable &= paired[("bec_count", method)] <= 10
    return paired.index[usable]


def _eligible_no_boundary(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["coverage_eligible"] & (rows["bec_count"] <= 10)]


def _table3_beta_summary(
    scenario: str,
    scenario_type: str,
    method: str,
    rows: pd.DataFrame,
    *,
    true_beta_b: float,
    unconditional_rows: pd.DataFrame | None = None,
) -> dict[str, object]:
    summary = {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "method": method,
        "n_used": len(rows),
        "true_beta_b": true_beta_b,
        "mean_beta": rows["beta_estimate"].mean(),
        "bias_beta": rows["bias_beta"].mean(),
        "mse_beta": rows["squared_error_beta"].mean(),
    }
    if unconditional_rows is not None:
        summary.update(
            {
                "n_unconditional": len(unconditional_rows),
                "bias_beta_unconditional": unconditional_rows["bias_beta"].mean(),
                "mse_beta_unconditional": unconditional_rows["squared_error_beta"].mean(),
            }
        )
    else:
        summary.update(
            {
                "n_unconditional": float("nan"),
                "bias_beta_unconditional": float("nan"),
                "mse_beta_unconditional": float("nan"),
            }
        )
    return summary


def _safe_log(value: object) -> float:
    value = float(value)
    if value <= 0.0:
        return float("nan")
    return float(np.log(value))


def _table4_variance_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["log_variance"] = rows["se_log_effect"] ** 2

    summaries: list[dict[str, object]] = []
    for (scenario, scenario_type), group in rows.groupby(["scenario", "scenario_type"], sort=False):
        common_replications = _common_table3_replications(group)
        for method in METHOD_ORDER:
            method_rows = group[
                (group["method"] == method)
                & group["replication"].isin(common_replications)
            ]
            ci_width = method_rows["ci_upper"] - method_rows["ci_lower"]
            summaries.append(
                {
                    "scenario": scenario,
                    "scenario_type": scenario_type,
                    "method": method,
                    "n_used": len(method_rows),
                    "mean_log_variance": method_rows["log_variance"].mean(),
                    "empirical_log_variance": method_rows["log_effect"].var(),
                    "mean_ci_width_rr": ci_width.mean(),
                    "variance_ratio": method_rows["log_variance"].mean()
                    / method_rows["log_effect"].var(),
                    "coverage": method_rows["covered"].mean(),
                    "coverage_n": int(method_rows["covered"].size),
                }
            )
    return pd.DataFrame(summaries).sort_values(["scenario", "method"])


def _write_table(table: pd.DataFrame, stem: Path, *, latex: str | None = None) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_to_markdown(table), encoding="utf-8")
    stem.with_suffix(".tex").write_text(latex or _to_latex(table), encoding="utf-8")


def _to_table1_boundary_case_latex(table: pd.DataFrame) -> str:
    """Render the selected boundary-estimate example table."""
    if table.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"

    scenario = str(table["scenario"].iloc[0]).replace("_", "\\_")
    replication = int(table["replication"].iloc[0])
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Four-method comparison for one randomly selected replication with a",
        "         boundary-estimate warning.}",
        "\\label{tab:four-method}",
        "% Consistent number formatting, scoped to this table:",
        "\\sisetup{group-separator={,}, group-minimum-digits=4, table-number-alignment=center}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{",
        "    l",
        "    S[table-format=2.0]",
        "    S[table-format=2.2]",
        "    S[table-format=2.2]",
        "    S[table-format=6.2]",
        "    l",
        "  }",
        "\\toprule",
        "{Method} & {$K$} & {$\\hat{\\beta}_b$} & {$\\hat{z}_b$} & {BEC} & {Status} \\\\",
        "\\midrule",
    ]
    for method_index, (method, method_rows) in enumerate(table.groupby("method", sort=False)):
        if method_index:
            lines.append("\\midrule")
        method_rows = method_rows.reset_index(drop=True)
        for row_index, row in enumerate(method_rows.itertuples(index=False)):
            status = "" if pd.isna(row.status) else str(row.status)
            method_cell = f"\\multirow{{{len(method_rows)}}}{{*}}{{{method}}}" if row_index == 0 else ""
            lines.append(
                "  "
                + " & ".join(
                    [
                        method_cell,
                        f"{int(row.max_iter):2d}",
                        _format_fixed_table_number(row.beta_hat_b),
                        _format_fixed_table_number(row.z_score_beta_hat_b),
                        _format_fixed_table_number(row.bec_count),
                        status,
                    ]
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            f"\\item Scenario: \\texttt{{{scenario}}}; replication~{replication}. The replication was sampled",
            f"  reproducibly (random state~{BOUNDARY_CASE_RANDOM_STATE}) from scenario--replication pairs in which at",
            "  least one method had $\\mathrm{BEC}>10$, then refit with iteration caps",
            "  $K=5,\\,15,\\,25$. Status is reported for the $K=25$ fit only.",
            "  $\\hat{\\beta}_b$ is the fitted binary-covariate coefficient and",
            "  $\\hat{z}_b=\\hat{\\beta}_b/\\mathrm{SE}(\\hat{\\beta}_b)$. BEC (boundary-estimate",
            "  criterion) is the maximum model-to-empirical SE ratio; $\\mathrm{BEC}>10$ flags a",
            "  boundary-warning diagnostic.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_ci(lower: object, upper: object) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "--"
    return f"({_format_table_number(lower)}, {_format_table_number(upper)})"


def _table1_status(converged: object, finite: object) -> str:
    if bool(converged) and bool(finite):
        return "Converged"
    if not bool(finite):
        return "Failed/nonfinite"
    return "Non-converged"


def _to_table2_convergence_latex(table: pd.DataFrame) -> str:
    """Render convergence counts in the compact grouped style used by the paper."""
    if table.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"

    by_scenario = {
        scenario: group.set_index("method")
        for scenario, group in table.groupby("scenario", sort=False)
    }
    row_groups = _table2_row_groups(table["scenario"].unique())
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\caption{Simulation convergence performance across simulation scenarios using disjoint convergence categories. The simulation parameters are set to $\\beta = (\\beta_0, \\beta_b, \\beta_c)^T = (-4, 1.6, 0.2)^T$, $N = 50$, $c = 0.2$, and $\\alpha = 0.4$ in bold, and each row summarizes replications with one parameter altered and all other parameters fixed.}",
        "\\begin{tabular}{ll|rr|rr|rr|rr}",
        "\\hline",
        "\\multicolumn{2}{c|}{Simulation setup} & \\multicolumn{2}{c|}{RR-GEE} & \\multicolumn{2}{c|}{RR-PGEE} & \\multicolumn{2}{c|}{OR-GEE} & \\multicolumn{2}{c}{OR-PGEE} " + r"\\",
        "Parameter & Value & Conv. & Non-conv. & Conv. & Non-conv. & Conv. & Non-conv. & Conv. & Non-conv. " + r"\\",
        "\\hline",
    ]
    wrote_group = False
    for parameter, rows in row_groups:
        present_rows = [row for row in rows if row[0] in by_scenario]
        if not present_rows:
            continue
        if wrote_group:
            lines.append("\\hline")
        wrote_group = True
        for index, (scenario, value, is_default) in enumerate(present_rows):
            label = parameter if index == 0 else ""
            display_value = f"\\textbf{{{value}}}" if is_default else value
            lines.append(
                " & ".join(
                    [label, display_value]
                    + _table2_convergence_cells(by_scenario[scenario])
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\begin{flushleft}",
            "\\footnotesize Counts are BEC $\\leq$ 10 (BEC $>$ 10). Thus, each Conv. or Non-conv. cell reports fits without the boundary-warning diagnostic, with BEC-warning counts in parentheses. The four underlying categories within each method--converged with BEC $\\leq$ 10, converged with BEC $>$ 10, non-converged with BEC $\\leq$ 10, and non-converged with BEC $>$ 10--are mutually exclusive and sum to the number of replications for that scenario. Conv.: finite fit that met the convergence criterion. Non-conv.: fit that did not produce a finite converged result, including failed or non-finite fits. BEC is the boundary estimates criterion, defined as the maximum model-SE ratio; BEC $>$ 10 is retained as an indicator of potential boundary-related instability. For the odds-ratio methods, BEC warnings rarely accompanied finite convergence: OR-PGEE had no BEC $>$ 10 counts in either convergence category, and OR-GEE had only three converged BEC-warning fits across all scenarios, with its BEC warnings concentrated among non-converged fits. This pattern reflects the diagnostic definition: once an OR fit returned finite estimates and satisfied the convergence criterion, the fitted model-SE ratios were generally below the BEC threshold; fits that failed to converge or became non-finite were instead counted as non-converged. $\\beta_0$ and $\\beta_b$: true intercept and binary covariate regression coefficients; $c$: proportion of 1's in the binary covariate; $\\alpha$: within-cluster correlation coefficient; $N$: number of subjects.",
            "\\end{flushleft}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _table2_convergence_cells(scenario: pd.DataFrame) -> list[str]:
    rr_gee = scenario.loc[METHOD_RR_GEE]
    rr_pgee = scenario.loc[METHOD_RR_PGEE]
    or_gee = scenario.loc[METHOD_OR_GEE]
    or_pgee = scenario.loc[METHOD_OR_PGEE]
    return (
        _table2_method_cells(rr_gee)
        + _table2_method_cells(rr_pgee)
        + _table2_method_cells(or_gee)
        + _table2_method_cells(or_pgee)
    )


def _table2_method_cells(method: pd.Series) -> list[str]:
    return [
        f"{int(method['converged_bec_le_10'])} ({int(method['converged_bec_gt_10'])})",
        f"{int(method['non_converged_bec_le_10'])} ({int(method['non_converged_bec_gt_10'])})",
    ]


def _to_table3_bias_mse_latex(table: pd.DataFrame, *, scale: str) -> str:
    """Render bias and MSE in manuscript table style."""
    if table.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"

    if scale == "rr":
        measure = "estimated risk ratio for the binary covariate"
        measure_note = "Bias and MSE are for the estimated risk ratio of the binary covariate."
    elif scale == "beta":
        measure = "estimated binary-covariate coefficient $\\beta_b$"
        measure_note = "Bias and MSE are for the estimated log-risk-ratio coefficient of the binary covariate; for OR-GEE and OR-PGEE the Zhang--Yu transformed risk ratio is placed on the log scale."
    else:
        raise ValueError(f"Unknown Table 3 scale: {scale}")

    common = _to_table3_bias_mse_latex_part(
        table,
        caption=_table3_caption(measure, summary="common"),
        footnote=f"{measure_note} Common-replication summaries use replications for which RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE are all finite and converged with BEC $\\leq 10$; $N_{{\\mathrm{{conv}}}}$ is the number of such replications.",
        scale=scale,
        summary="common",
    )
    method_specific = _to_table3_bias_mse_latex_part(
        table,
        caption=_table3_caption(measure, summary="method_specific"),
        footnote=f"{measure_note} Method-specific summaries use each method's finite and converged replications with BEC $\\leq 10$ independently; $N_{{\\mathrm{{conv}}}}$ is the method-specific number of replications used in each summary.",
        scale=scale,
        summary="method_specific",
    )
    return common + "\n" + method_specific


def _table3_caption(measure: str, *, summary: str) -> str:
    summary_text = {
        "common": "common-replication summary",
        "method_specific": "method-specific summary",
    }[summary]
    return (
        f"Bias and mean-squared error of the {measure} across simulation scenarios "
        f"({summary_text}). Base parameters (in bold) are $\\bm{{\\beta}} = "
        "(\\beta_0, \\beta_b, \\beta_c)^\\top = (-4, 1.6, 0.2)^\\top$, "
        "$N = 50$, $c = 0.2$, and $\\alpha = 0.4$; each row alters one "
        "parameter and holds the rest fixed. Summaries are conditional on finite, "
        "converged estimates."
    )


def _to_table3_bias_mse_latex_part(
    table: pd.DataFrame,
    *,
    caption: str,
    footnote: str,
    scale: str,
    summary: str,
) -> str:
    by_scenario = {
        scenario: group.set_index("method")
        for scenario, group in table.groupby("scenario", sort=False)
    }
    row_groups = _table2_row_groups(table["scenario"].unique())
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        f"\\caption{{{caption}}}",
        f"\\label{{{_table3_label(scale, summary)}}}",
        _table3_tabular_spec(summary),
        "\\toprule",
        _table3_method_header(summary),
        _table3_column_header(summary),
        "\\midrule",
    ]
    wrote_group = False
    for parameter, rows in row_groups:
        present_rows = [row for row in rows if row[0] in by_scenario]
        if not present_rows:
            continue
        if wrote_group:
            lines.append("\\midrule")
        wrote_group = True
        for index, (scenario, value, is_default) in enumerate(present_rows):
            label = parameter if index == 0 else ""
            display_value = f"\\textbf{{{value}}}" if is_default else value
            lines.append(
                " & ".join(
                    [label, display_value]
                    + _table3_bias_mse_cells(
                        by_scenario[scenario],
                        scale=scale,
                        summary=summary,
                    )
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{flushleft}",
            f"\\footnotesize {footnote}",
            "\\end{flushleft}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _table3_label(scale: str, summary: str) -> str:
    if scale == "beta" and summary == "common":
        return "tab:bias_mse_four_methods"
    scale_part = "rr" if scale == "rr" else "beta"
    summary_part = "common" if summary == "common" else "method_specific"
    return f"tab:{scale_part}_bias_mse_four_methods_{summary_part}"


def _table3_bias_mse_cells(
    scenario: pd.DataFrame,
    *,
    scale: str,
    summary: str,
) -> list[str]:
    rr_gee = scenario.loc[METHOD_RR_GEE]
    rr_pgee = scenario.loc[METHOD_RR_PGEE]
    or_gee = scenario.loc[METHOD_OR_GEE]
    or_pgee = scenario.loc[METHOD_OR_PGEE]
    if summary == "common":
        return (
            _table3_method_cells(rr_gee, scale=scale, summary=summary, include_n=True)
            + _table3_method_cells(rr_pgee, scale=scale, summary=summary, include_n=False)
            + _table3_method_cells(or_gee, scale=scale, summary=summary, include_n=False)
            + _table3_method_cells(or_pgee, scale=scale, summary=summary, include_n=False)
        )
    if summary == "method_specific":
        return (
            _table3_method_cells(rr_gee, scale=scale, summary=summary, include_n=True)
            + _table3_method_cells(rr_pgee, scale=scale, summary=summary, include_n=True)
            + _table3_method_cells(or_gee, scale=scale, summary=summary, include_n=True)
            + _table3_method_cells(or_pgee, scale=scale, summary=summary, include_n=True)
        )
    raise ValueError(f"Unknown Table 3 summary: {summary}")


def _table3_tabular_spec(summary: str) -> str:
    if summary == "common":
        return "\\begin{tabular}{l c S[table-format=3.0] *{4}{S[table-format=-2.2] S[table-format=5.2]}}"
    if summary == "method_specific":
        return "\\begin{tabular}{l c *{4}{S[table-format=4.0] S[table-format=-2.2] S[table-format=5.2]}}"
    raise ValueError(f"Unknown Table 3 summary: {summary}")


def _table3_method_header(summary: str) -> str:
    if summary == "common":
        return "\n".join(
            [
                "\\multicolumn{3}{c}{Simulation setup} & \\multicolumn{2}{c}{RR-GEE} & \\multicolumn{2}{c}{RR-PGEE} & \\multicolumn{2}{c}{OR-GEE} & \\multicolumn{2}{c}{OR-PGEE} " + r"\\",
                "\\cmidrule(lr){1-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}",
            ]
        )
    if summary == "method_specific":
        return "\n".join(
            [
                "\\multicolumn{2}{c}{Simulation setup} & \\multicolumn{3}{c}{RR-GEE} & \\multicolumn{3}{c}{RR-PGEE} & \\multicolumn{3}{c}{OR-GEE} & \\multicolumn{3}{c}{OR-PGEE} " + r"\\",
                "\\cmidrule(lr){1-2} \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}",
            ]
        )
    raise ValueError(f"Unknown Table 3 summary: {summary}")


def _table3_column_header(summary: str) -> str:
    if summary == "common":
        return "{Parameter} & {Value} & {$N_{\\mathrm{conv}}$} & {Bias} & {MSE} & {Bias} & {MSE} & {Bias} & {MSE} & {Bias} & {MSE} " + r"\\"
    if summary == "method_specific":
        return "{Parameter} & {Value} & {$N_{\\mathrm{conv}}$} & {Bias} & {MSE} & {$N_{\\mathrm{conv}}$} & {Bias} & {MSE} & {$N_{\\mathrm{conv}}$} & {Bias} & {MSE} & {$N_{\\mathrm{conv}}$} & {Bias} & {MSE} " + r"\\"
    raise ValueError(f"Unknown Table 3 summary: {summary}")


def _table3_method_cells(
    method: pd.Series,
    *,
    scale: str,
    summary: str,
    include_n: bool,
) -> list[str]:
    bias_column = "bias_rr" if scale == "rr" else "bias_beta"
    mse_column = "mse_rr" if scale == "rr" else "mse_beta"
    if summary == "common":
        cells = [
            _format_table3_number(method[bias_column]),
            _format_table3_number(method[mse_column]),
        ]
        if include_n:
            cells.insert(0, "" if pd.isna(method["n_used"]) else str(int(method["n_used"])))
        return cells
    if summary == "method_specific":
        cells = [
            _format_table3_number(method[f"{bias_column}_unconditional"]),
            _format_table3_number(method[f"{mse_column}_unconditional"]),
        ]
        if include_n:
            cells.insert(
                0,
                "" if pd.isna(method["n_unconditional"]) else str(int(method["n_unconditional"])),
            )
        return cells
    raise ValueError(f"Unknown Table 3 summary: {summary}")


def _format_table3_number(value: object) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def _to_table4_variance_coverage_latex(table: pd.DataFrame) -> str:
    """Render variance and coverage summaries in manuscript table style."""
    if table.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"

    methods = [method for method in METHOD_ORDER if method in set(table["method"])]
    by_scenario = {
        scenario: group.set_index("method")
        for scenario, group in table.groupby("scenario", sort=False)
    }
    row_groups = _table2_row_groups(table["scenario"].unique())
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\caption{Variance and coverage performance of the estimated risk ratio for the binary covariate across simulation scenarios. The simulation parameters are set to $\\beta = (\\beta_0, \\beta_b, \\beta_c)^T = (-4, 1.6, 0.2)^T$, $N = 50$, $c = 0.2$, and $\\alpha = 0.4$ in bold, and each row summarizes replications with one parameter altered and all other parameters fixed. Summaries are conditional on finite and converged estimates.}",
        "\\resizebox{\\linewidth}{!}{%",
        _table4_tabular_spec(methods),
        "\\hline",
        _table4_method_header(methods),
        _table4_column_header(methods),
        "\\hline",
    ]
    wrote_group = False
    for parameter, rows in row_groups:
        present_rows = [row for row in rows if row[0] in by_scenario]
        if not present_rows:
            continue
        if wrote_group:
            lines.append("\\hline")
        wrote_group = True
        for index, (scenario, value, is_default) in enumerate(present_rows):
            label = parameter if index == 0 else ""
            display_value = f"\\textbf{{{value}}}" if is_default else value
            lines.append(
                " & ".join(
                    [label, display_value]
                    + _table4_variance_coverage_cells(by_scenario[scenario], methods)
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "}",
            "\\begin{flushleft}",
            "\\footnotesize Summaries use common replications where RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE are finite and converged with BEC $\\leq$ 10. CP: empirical 95\\% confidence interval coverage probability; Mean Var.: average squared reported standard error on the log-effect scale; Emp. Var.: empirical variance of the log-effect estimates; Ratio: Mean Var. divided by Emp. Var.; Mean CI width: average confidence interval width on the risk-ratio scale; N: number of common replications used in the summary.",
            "\\end{flushleft}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _table4_tabular_spec(methods: list[str]) -> str:
    return "\\begin{tabular}{ll|" + "|".join(["rrrrrr"] * len(methods)) + "}"


def _table4_method_header(methods: list[str]) -> str:
    labels = {
        METHOD_RR_GEE: "RR-GEE",
        METHOD_RR_PGEE: "RR-PGEE",
        METHOD_OR_GEE: "OR-GEE",
        METHOD_OR_PGEE: "OR-PGEE",
    }
    parts = ["\\multicolumn{2}{c|}{Simulation setup}"]
    for index, method in enumerate(methods):
        suffix = "" if index == len(methods) - 1 else "|"
        parts.append(f"\\multicolumn{{6}}{{c{suffix}}}{{{labels[method]}}}")
    return " & ".join(parts) + " " + r"\\"


def _table4_column_header(methods: list[str]) -> str:
    method_columns = [
        "N",
        "CP",
        "\\shortstack{Mean\\\\Var.}",
        "\\shortstack{Emp.\\\\Var.}",
        "Ratio",
        "\\shortstack{Mean CI\\\\width}",
    ]
    return " & ".join(["Parameter", "Value"] + method_columns * len(methods)) + " " + r"\\"


def _table4_variance_coverage_cells(scenario: pd.DataFrame, methods: list[str]) -> list[str]:
    cells: list[str] = []
    for method in methods:
        cells.extend(_table4_method_cells(scenario.loc[method]))
    return cells


def _table4_method_cells(method: pd.Series) -> list[str]:
    return [
        str(int(method["n_used"])),
        _format_coverage(method["coverage"]),
        _format_table_number(method["mean_log_variance"]),
        _format_table_number(method["empirical_log_variance"]),
        _format_table_number(method["variance_ratio"]),
        _format_table_number(method["mean_ci_width_rr"]),
    ]


def _format_coverage(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_fixed_table_number(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_table_number(value: object) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value != 0.0 and abs(value) < 0.01:
        return f"{value:.1e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _table2_row_groups(scenarios: object) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    scenario_names = set(scenarios)
    confounded_rows = [
        (scenario, scenario.removeprefix("confounded_shift_").replace("_", "."), False)
        for scenario in sorted(scenario_names)
        if scenario.startswith("confounded_shift_")
    ]
    row_groups = [
        (
            "$\\beta_0$",
            [
                ("base", "-4", True),
                ("beta0_-3", "-3", False),
                ("beta0_-2", "-2", False),
            ],
        ),
        (
            "$\\beta_b$",
            [
                ("beta_b_1.2", "1.2", False),
                ("beta_b_1.4", "1.4", False),
                ("base", "1.6", True),
                ("beta_b_1.8", "1.8", False),
                ("beta_b_2", "2.0", False),
            ],
        ),
        (
            "$c$",
            [
                ("base", "0.2", True),
                ("prop_0.3", "0.3", False),
                ("prop_0.4", "0.4", False),
                ("prop_0.5", "0.5", False),
                ("prop_0.6", "0.6", False),
                ("prop_0.7", "0.7", False),
                ("prop_0.8", "0.8", False),
            ],
        ),
        (
            "$\\alpha$",
            [
                ("alpha_0.2", "0.2", False),
                ("rho_0.2", "0.2", False),
                ("alpha_0.3", "0.3", False),
                ("base", "0.4", True),
                ("alpha_0.5", "0.5", False),
                ("alpha_0.6", "0.6", False),
                ("rho_0.6", "0.6", False),
                ("alpha_0.7", "0.7", False),
                ("alpha_0.8", "0.8", False),
                ("rho_0.8", "0.8", False),
            ],
        ),
        (
            "$N$",
            [
                ("n_25", "25", False),
                ("base", "50", True),
                ("n_75", "75", False),
                ("n_100", "100", False),
            ],
        ),
    ]
    if confounded_rows:
        row_groups.append(("Conf. shift", confounded_rows))
    return row_groups


def _to_markdown(table: pd.DataFrame) -> str:
    """Render a simple Markdown table without optional pandas dependencies."""
    if table.empty:
        return "| empty |\n| --- |\n"
    text = table.astype(object).where(pd.notna(table), "")
    headers = [str(column) for column in text.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in text.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _to_latex(table: pd.DataFrame) -> str:
    """Render a minimal LaTeX tabular without optional pandas dependencies."""
    if table.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"
    text = table.astype(object).where(pd.notna(table), "")
    columns = [str(column).replace("_", "\\_") for column in text.columns]
    colspec = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\hline"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\hline")
    for row in text.itertuples(index=False, name=None):
        values = [str(value).replace("_", "\\_") for value in row]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"
