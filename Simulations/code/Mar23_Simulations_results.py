#!/usr/bin/env python3
"""March 2023 simulation result summaries and figures translated to Python."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import Simulations_results as base

GEEDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Mar23_results")
BETA = np.array([-4.0, 1.6, 0.2])
ALPHAS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
BETA_B = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
NS = np.array([25, 50, 75, 100])
BETA0 = np.array([-4.0, -3.0, -2.0])
GAMMAS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def use_mar23_dir() -> None:
    base.GEEDIR = GEEDIR


def arr(env: dict[str, Any], name: str) -> np.ndarray:
    return base.arr(env, name)


def or_diagnostics(prefix: str, n: int) -> None:
    for i in range(1, n + 1):
        env = base.load_result(GEEDIR / f"{prefix}{i}.RData")
        coefs = arr(env, "coefs_ORpgee")
        iters = arr(env, "iter_ORpgee")
        ratio = arr(env, "SEs_model_ratio_ORpgee")
        complete = base.complete_cases(ratio)
        div = base.se_ok(ratio[complete], "gt")
        print(i)
        print(1000 - int(np.sum(base.complete_cases(coefs))))
        print(int(np.nansum(iters == 25)))
        print("-----")
        base.show(base.table(div))
        base.show(base.table(div & (iters[complete] != 25)))
        base.show(base.table(iters[complete]))
        base.show(base.table(arr(env, "eta_ORpgee")[complete] > 0))


def or_to_log_relative_risk(coefs: np.ndarray) -> np.ndarray:
    p0 = np.exp(coefs[:, 0]) / (1.0 + np.exp(coefs[:, 0]))
    odds_b = np.exp(coefs[:, 1])
    odds_c = np.exp(coefs[:, 2])
    rr_b = odds_b / ((1.0 - p0) + p0 * odds_b)
    rr_c = odds_c / ((1.0 - p0) + p0 * odds_c)
    return np.column_stack([np.log(rr_b), np.log(rr_c)])


def summarize_series(prefix: str, n: int, beta_values: list[np.ndarray], values: np.ndarray, xname: str, *, load_prefix: str | None = None, beta0_sep_guard: bool = False) -> tuple[np.ndarray, ...]:
    load_prefix = prefix if load_prefix is None else load_prefix
    bias_nonsep_gee = np.zeros((n, 3)); bias_complete_gee = np.zeros((n, 3)); bias_nonsep_pgee = np.zeros((n, 3)); bias_complete_pgee = np.zeros((n, 3))
    bias_sep_gee = np.zeros((n, 3)); bias_sep_pgee = np.zeros((n, 3)); mse_nonsep_gee = np.zeros((n, 3)); mse_complete_gee = np.zeros((n, 3))
    mse_nonsep_pgee = np.zeros((n, 3)); mse_complete_pgee = np.zeros((n, 3)); mse_sep_gee = np.zeros((n, 3)); mse_sep_pgee = np.zeros((n, 3))
    bias_complete_orpgee = np.zeros((n, 2)); mse_complete_orpgee = np.zeros((n, 2))
    for j in range(n):
        beta = beta_values[j]
        env = base.load_result(GEEDIR / f"{load_prefix}{j + 1}.RData")
        coefs_gee = arr(env, "coefs_geePK"); coefs_pgee = arr(env, "coefs_PgeePK")
        ratio_gee = arr(env, "SEs_model_ratio_geePK"); ratio_pgee = arr(env, "SEs_model_ratio_PgeePK")
        complete_gee = base.complete_cases(ratio_gee)
        bias_complete_gee[j, :] = base.mean_rows(coefs_gee, complete_gee) - beta
        mse_complete_gee[j, :] = base.mse_rows(coefs_gee, complete_gee, beta)
        nonsep_mask = base.se_ok(ratio_gee, "lt") & base.se_ok(ratio_pgee, "lt") & base.complete_cases(ratio_gee) & base.complete_cases(ratio_pgee) & base.complete_cases(coefs_gee) & arr(env, "conv_geePK").astype(bool)
        nonsep = base.idx_from_mask(nonsep_mask)
        print(len(nonsep))
        or_idx = np.intersect1d(nonsep, base.idx_from_mask(base.complete_cases(arr(env, "coefs_ORpgee")) & (arr(env, "iter_ORpgee") != 25)))
        nonsep = or_idx
        print(len(or_idx)); print("---")
        base.report_se_variance(env, nonsep)
        bias_nonsep_gee[j, :] = base.mean_rows_by_index(coefs_gee, nonsep) - beta
        mse_nonsep_gee[j, :] = base.mse_rows_by_index(coefs_gee, nonsep, beta)
        bias_nonsep_pgee[j, :] = base.mean_rows_by_index(coefs_pgee, nonsep) - beta
        mse_nonsep_pgee[j, :] = base.mse_rows_by_index(coefs_pgee, nonsep, beta)
        sep = base.idx_from_mask(base.se_ok(ratio_gee, "gt") & base.complete_cases(ratio_gee) & base.complete_cases(ratio_pgee))
        if (not beta0_sep_guard) or len(sep) > 1:
            bias_sep_gee[j, :] = base.mean_rows_by_index(coefs_gee, sep) - beta
            mse_sep_gee[j, :] = base.mse_rows_by_index(coefs_gee, sep, beta)
            bias_sep_pgee[j, :] = base.mean_rows_by_index(coefs_pgee, sep) - beta
            mse_sep_pgee[j, :] = base.mse_rows_by_index(coefs_pgee, sep, beta)
        or_rr = or_to_log_relative_risk(arr(env, "coefs_ORpgee"))
        bias_complete_orpgee[j, :] = np.sum(or_rr[or_idx, :], axis=0) / len(or_idx) - beta[1:3]
        mse_complete_orpgee[j, :] = np.sum((or_rr[or_idx, :] - beta[1:3]) ** 2, axis=0) / len(or_idx)
        print("--unconditional--")
        complete_pgee = base.complete_cases(ratio_pgee) & arr(env, "conv_PgeePK").astype(bool) & base.se_ok(ratio_pgee, "lt")
        bias_complete_pgee[j, :] = base.mean_rows(coefs_pgee, complete_pgee) - beta
        mse_complete_pgee[j, :] = base.mse_rows(coefs_pgee, complete_pgee, beta)
        print(int(np.sum(complete_pgee)))
        print("mean SE^2 PGEE"); base.show(np.mean(arr(env, "SEs_PgeePK")[complete_pgee, :] ** 2, axis=0))
        print("variance coefs PGEE"); base.show(np.std(coefs_pgee[complete_pgee, :], axis=0, ddof=1) ** 2)
        base.show(np.mean(arr(env, "SEs_PgeePK")[complete_pgee, :] ** 2, axis=0) / (np.std(coefs_pgee[complete_pgee, :], axis=0, ddof=1) ** 2))
        print(j + 1)
    return (bias_complete_gee, mse_complete_gee, bias_complete_pgee, mse_complete_pgee, bias_nonsep_gee, mse_nonsep_gee, bias_nonsep_pgee, mse_nonsep_pgee, bias_sep_gee, mse_sep_gee, bias_sep_pgee, mse_sep_pgee, bias_complete_orpgee, mse_complete_orpgee)


def alpha_followup() -> None:
    env = base.load_result(GEEDIR / "alpha4.RData")
    idx = base.idx_from_mask(base.se_ok(arr(env, "SEs_model_ratio_geePK"), "lt") & base.complete_cases(arr(env, "SEs_model_ratio_geePK")) & base.complete_cases(arr(env, "SEs_model_ratio_PgeePK")) & arr(env, "conv_geePK").astype(bool))
    print(len(idx))
    base.show(np.mean(arr(env, "SEs_geePK")[idx, :], axis=0)); base.show(np.std(arr(env, "coefs_geePK")[idx, :], axis=0, ddof=1)); base.show(np.mean(arr(env, "SEs_geePK")[idx, :], axis=0) / np.std(arr(env, "coefs_geePK")[idx, :], axis=0, ddof=1))
    base.show(np.mean(arr(env, "SEs_PgeePK")[idx, :], axis=0)); base.show(np.std(arr(env, "coefs_PgeePK")[idx, :], axis=0, ddof=1)); base.show(np.mean(arr(env, "SEs_PgeePK")[idx, :], axis=0) / np.std(arr(env, "coefs_PgeePK")[idx, :], axis=0, ddof=1))


def zsep_plots() -> None:
    old = base.GEEDIR
    try:
        base.GEEDIR = GEEDIR
        env = base.load_result(GEEDIR / "test_alpha3.RData")
        ratio_gee = arr(env, "SEs_model_ratio_geePK"); ratio_pgee = arr(env, "SEs_model_ratio_PgeePK")
        coefs_gee = arr(env, "coefs_geePK"); coefs_pgee = arr(env, "coefs_PgeePK"); ses_gee = arr(env, "SEs_geePK"); ses_pgee = arr(env, "SEs_PgeePK")
        sep = base.idx_from_mask(base.se_ok(ratio_gee, "gt") & base.complete_cases(ratio_gee) & base.complete_cases(ratio_pgee))
        print(len(sep))
        base.scatter_density(coefs_gee[sep, 1], coefs_pgee[sep, 1], GEEDIR / "plots/estimates_binary_sep_scatter.pdf", "GEE beta_b", "PGEE beta_b", (-30, 30), (-30, 30))
        zdat = pd.DataFrame({"x": coefs_gee[sep, 1] / ses_gee[sep, 1], "y": coefs_pgee[sep, 1] / ses_pgee[sep, 1]})
        base.show(zdat.describe())
        base.scatter_density(zdat["x"].to_numpy(), zdat["y"].to_numpy(), GEEDIR / "plots/zscores_binary_sep_scatter.pdf", "GEE z_b", "PGEE z_b", (-120, 120), (-5, 5))
        complete = base.idx_from_mask(base.se_ok(ratio_gee, "le") & base.complete_cases(ratio_gee) & base.complete_cases(ratio_pgee) & arr(env, "conv_geePK").astype(bool))
        print(len(complete))
        base.scatter_density(coefs_gee[complete, 1], coefs_pgee[complete, 1], GEEDIR / "plots/estimates_binary_complete_scatter_vol2.pdf", "RR-GEE beta_b", "RR-PGEE beta_b", (-5, 5), (-5, 5), hline=1.6, vline=1.6, size=(6, 7))
        base.scatter_density(ses_gee[complete, 1], ses_pgee[complete, 1], GEEDIR / "plots/SEs_binary_complete_scatter_vol2.pdf", "RR-GEE sandwich se beta_b", "RR-PGEE sandwich se beta_b", (0, 2), (0, 2), size=(6, 7))
        base.scatter_density(coefs_gee[complete, 1] / ses_gee[complete, 1], coefs_pgee[complete, 1] / ses_pgee[complete, 1], GEEDIR / "plots/zscores_binary_complete_scatter_vol2.pdf", "RR-GEE z_b", "RR-PGEE z_b", (-5, 5), (-5, 5), size=(6, 7))
        sep_null = base.idx_from_mask(base.se_ok(ratio_gee, "lt") & base.complete_cases(ratio_gee) & ~arr(env, "conv_geePK").astype(bool))
        print(len(sep_null)); base.show(coefs_gee[sep_null, :]); base.show(arr(env, "SEs_model_geePK")[sep_null, :]); base.show(coefs_pgee[sep_null, :])
        if "tabdat" in env: base.show(np.asarray(env["tabdat"], dtype=object)[sep_null])
        print(np.mean(np.abs(coefs_gee[complete, 1] - 1.6) / ses_gee[complete, 1] >= base.stats.t.ppf(0.975, 47)))
        print(np.mean(np.abs(coefs_pgee[complete, 1] - 1.6) / ses_pgee[complete, 1] >= base.stats.t.ppf(0.975, 47)))
        se_ratio = pd.concat([pd.DataFrame({"x": ratio_gee[:, 1], "group": "RR-GEE"}), pd.DataFrame({"x": ratio_pgee[:, 1], "group": "RR-PGEE"})])
        g = base.sns.displot(data=se_ratio, x="x", col="group", bins=30); g.set(xlim=(0, 10), xlabel="BEC beta_b")
        (GEEDIR / "plots").mkdir(parents=True, exist_ok=True); g.fig.savefig(GEEDIR / "plots/SEratios_betaB_histogram.pdf"); base.plt.close(g.fig)
    finally:
        base.GEEDIR = old


def main() -> int:
    use_mar23_dir()
    base.diagnostic_counts("test_alpha", 7); or_diagnostics("test_alpha", 7)
    alpha_results = summarize_series("test_alpha", 7, [BETA.copy() for _ in range(7)], ALPHAS, "alpha")
    base.show(alpha_results[0]); base.show(alpha_results[1]); base.show(alpha_results[2][:, 1]); base.show(alpha_results[3][:, 1]); base.show(alpha_results[4][:, 1]); base.show(alpha_results[6][:, 1]); base.show(alpha_results[12][:, 0]); base.show(alpha_results[5][:, 1]); base.show(alpha_results[7][:, 1]); base.show(alpha_results[13][:, 0])
    alpha_followup(); zsep_plots()
    base.diagnostic_counts("BetaB_", 5)
    beta_b_results = base.summarize_series("BetaB_", 5, [np.array([-4.0, b, 0.2]) for b in BETA_B], BETA_B, "beta")
    base.show(beta_b_results[8][:, 1]); base.show(beta_b_results[10][:, 1]); base.show(beta_b_results[9][:, 1]); base.show(beta_b_results[11][:, 1]); base.show(beta_b_results[4][:, 1]); base.show(beta_b_results[6][:, 1]); base.show(beta_b_results[5][:, 1]); base.show(beta_b_results[7][:, 1]); base.show(beta_b_results[2][:, 1]); base.show(beta_b_results[3][:, 1])
    base.diagnostic_counts("test_N_", 6); or_diagnostics("test_N_", 6)
    n_results = summarize_series("test_N_", 6, [BETA.copy() for _ in range(6)], NS, "N")
    base.show(n_results[0]); base.show(n_results[1]); base.show(n_results[4][:, 1]); base.show(n_results[6][:, 1]); base.show(n_results[12][:, 0]); base.show(n_results[5][:, 1]); base.show(n_results[7][:, 1]); base.show(n_results[13][:, 0]); base.show(n_results[2][:, 1]); base.show(n_results[3][:, 1])
    base.diagnostic_counts("test_Beta0", 3); or_diagnostics("test_Beta0", 3)
    beta0_results = summarize_series("test_Beta0", 3, [np.array([b0, 1.6, 0.2]) for b0 in BETA0], BETA0, "beta0", beta0_sep_guard=True)
    base.show(beta0_results[0]); base.show(beta0_results[1]); base.show(beta0_results[4][:, 1]); base.show(beta0_results[6][:, 1]); base.show(beta0_results[12][:, 0]); base.show(beta0_results[5][:, 1]); base.show(beta0_results[7][:, 1]); base.show(beta0_results[13][:, 0]); base.show(beta0_results[2][:, 1]); base.show(beta0_results[3][:, 1])
    base.diagnostic_counts("gamma", 7)
    gamma_results = base.summarize_series("gamma", 7, [BETA.copy() for _ in range(7)], GAMMAS, "gamma", load_prefix="N100_gamma")
    base.show(gamma_results[0]); base.show(gamma_results[1]); base.show(gamma_results[4][:, 1]); base.show(gamma_results[6][:, 1]); base.show(gamma_results[5][:, 1]); base.show(gamma_results[7][:, 1]); base.show(gamma_results[2][:, 1]); base.show(gamma_results[3][:, 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
