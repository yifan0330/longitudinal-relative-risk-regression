#!/usr/bin/env python3
"""Simulation result summaries and figures translated to Python."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

GEEDIR = Path(
    "/well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Sept21_resultsN50_phi"
)
BETA = np.array([-4.0, 1.6, 0.2])
ALPHAS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
BETA_B = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
NS = np.array([25, 50, 75, 100])
BETA0 = np.array([-4.0, -3.0, -2.0])
GAMMAS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def load_result(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        import pyreadr

        return dict(pyreadr.read_r(str(path)))
    except Exception:
        pass
    try:
        import rdata

        return rdata.conversion.convert(rdata.parser.parse_file(str(path)))
    except Exception as exc:
        raise RuntimeError(f"Unable to load result file {path}") from exc


def arr(env: dict[str, Any], name: str) -> np.ndarray:
    value = env[name]
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.to_numpy()
    return np.asarray(value)


def complete_cases(*values: Any) -> np.ndarray:
    masks = []
    for value in values:
        a = np.asarray(value)
        if a.ndim == 1:
            masks.append(~pd.isna(a))
        else:
            masks.append(~pd.isna(a).any(axis=1))
    return np.logical_and.reduce(masks)


def table(values: Any) -> dict[Any, int]:
    a = np.asarray(values).ravel()
    return dict(Counter(a.tolist()))


def show(value: Any) -> None:
    if isinstance(value, np.ndarray):
        print(np.array2string(value, precision=6, suppress_small=False))
    elif isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    else:
        print(value)


def mean_rows(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.sum(x[idx, :], axis=0) / np.sum(idx)


def mean_rows_by_index(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.sum(x[idx, :], axis=0) / len(idx)


def mse_rows(x: np.ndarray, idx: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.sum((x[idx, :] - beta) ** 2, axis=0) / np.sum(idx)


def mse_rows_by_index(x: np.ndarray, idx: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.sum((x[idx, :] - beta) ** 2, axis=0) / len(idx)


def idx_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def se_ok(a: np.ndarray, op: str = "lt") -> np.ndarray:
    if op == "lt":
        return (a[:, 0] < 10) & (a[:, 1] < 10) & (a[:, 2] < 10)
    if op == "le":
        return (a[:, 0] <= 10) & (a[:, 1] <= 10) & (a[:, 2] <= 10)
    return (a[:, 0] > 10) | (a[:, 1] > 10) | (a[:, 2] > 10)


def report_se_variance(env: dict[str, Any], idx: np.ndarray) -> None:
    coefs_gee = arr(env, "coefs_geePK")
    ses_gee = arr(env, "SEs_geePK")
    coefs_pgee = arr(env, "coefs_PgeePK")
    ses_pgee = arr(env, "SEs_PgeePK")
    print("mean SE^2 GEE")
    show(np.mean(ses_gee[idx, :] ** 2, axis=0))
    print("variance coefs GEE")
    show(np.std(coefs_gee[idx, :], axis=0, ddof=1) ** 2)
    show(
        np.mean(ses_gee[idx, :] ** 2, axis=0)
        / (np.std(coefs_gee[idx, :], axis=0, ddof=1) ** 2)
    )
    print("---")
    print("mean SE^2 PGEE")
    show(np.mean(ses_pgee[idx, :] ** 2, axis=0))
    print("variance coefs PGEE")
    show(np.std(coefs_pgee[idx, :], axis=0, ddof=1) ** 2)
    show(
        np.mean(ses_pgee[idx, :] ** 2, axis=0)
        / (np.std(coefs_pgee[idx, :], axis=0, ddof=1) ** 2)
    )


def diagnostic_counts(
    prefix: str, n: int, *, pgee: bool = True, gee: bool = True
) -> None:
    if gee:
        for i in range(1, n + 1):
            env = load_result(GEEDIR / f"{prefix}{i}.RData")
            ratio = arr(env, "SEs_model_ratio_geePK")
            conv = arr(env, "conv_geePK").astype(bool)
            complete = complete_cases(ratio)
            div = se_ok(ratio[complete], "gt")
            print(i)
            print(1000 - int(np.sum(complete)))
            show(table(div))
            show(table(div & ~conv[complete]))
            if "iter_geePK" in env:
                show(table(arr(env, "iter_geePK")[complete]))
            show(table(~conv[complete] & ~div))
            show(table(arr(env, "eta_geePK")[complete] > 0))
            print(int(np.sum((~conv[complete]) & (~div))))
            print("-----")
    if pgee:
        for i in range(1, n + 1):
            env = load_result(GEEDIR / f"{prefix}{i}.RData")
            ratio = arr(env, "SEs_model_ratio_PgeePK")
            conv = arr(env, "conv_PgeePK").astype(bool)
            complete = complete_cases(ratio)
            div = se_ok(ratio[complete], "gt")
            print(i)
            print(1000 - int(np.sum(complete)))
            show(table(div))
            show(table(div & ~conv[complete]))
            if "iter_PgeePK" in env:
                show(table(arr(env, "iter_PgeePK")[complete]))
            show(table(conv[complete]))
            show(table(~conv[complete] & ~div))
            show(table(arr(env, "eta_PgeePK")[complete] > 0))
            print("-----")


def build_bias_frame(
    values: np.ndarray, gee: np.ndarray, pgee: np.ndarray, xname: str
) -> pd.DataFrame:
    values = np.resize(values, gee.shape[0])
    return pd.concat(
        [
            pd.DataFrame(
                {"bias": gee[:, 0], "var": "Intercept", "method": "GEE", xname: values}
            ),
            pd.DataFrame(
                {
                    "bias": pgee[:, 0],
                    "var": "Intercept",
                    "method": "PGEE",
                    xname: values,
                }
            ),
            pd.DataFrame(
                {
                    "bias": gee[:, 1],
                    "var": "Categorical",
                    "method": "GEE",
                    xname: values,
                }
            ),
            pd.DataFrame(
                {
                    "bias": pgee[:, 1],
                    "var": "Categorical",
                    "method": "PGEE",
                    xname: values,
                }
            ),
            pd.DataFrame(
                {"bias": gee[:, 2], "var": "Continuous", "method": "GEE", xname: values}
            ),
            pd.DataFrame(
                {
                    "bias": pgee[:, 2],
                    "var": "Continuous",
                    "method": "PGEE",
                    xname: values,
                }
            ),
        ],
        ignore_index=True,
    )


def plot_bias(df: pd.DataFrame, x: str, title: str | None = None) -> None:
    g = sns.relplot(
        data=df, x=x, y="bias", hue="method", style="method", col="var", kind="scatter"
    )
    if title:
        g.fig.suptitle(title)
    plt.close(g.fig)


def summarize_series(
    prefix: str,
    n: int,
    beta_values: list[np.ndarray],
    values: np.ndarray,
    xname: str,
    *,
    load_prefix: str | None = None,
    beta0_sep_guard: bool = False,
) -> tuple[np.ndarray, ...]:
    load_prefix = prefix if load_prefix is None else load_prefix
    bias_nonsep_gee = np.zeros((n, 3))
    bias_complete_gee = np.zeros((n, 3))
    bias_nonsep_pgee = np.zeros((n, 3))
    bias_complete_pgee = np.zeros((n, 3))
    bias_sep_gee = np.zeros((n, 3))
    bias_sep_pgee = np.zeros((n, 3))
    mse_nonsep_gee = np.zeros((n, 3))
    mse_complete_gee = np.zeros((n, 3))
    mse_nonsep_pgee = np.zeros((n, 3))
    mse_complete_pgee = np.zeros((n, 3))
    mse_sep_gee = np.zeros((n, 3))
    mse_sep_pgee = np.zeros((n, 3))
    for j in range(n):
        beta = beta_values[j]
        env = load_result(GEEDIR / f"{load_prefix}{j + 1}.RData")
        coefs_gee = arr(env, "coefs_geePK")
        coefs_pgee = arr(env, "coefs_PgeePK")
        ratio_gee = arr(env, "SEs_model_ratio_geePK")
        ratio_pgee = arr(env, "SEs_model_ratio_PgeePK")
        complete_gee = complete_cases(ratio_gee)
        bias_complete_gee[j, :] = mean_rows(coefs_gee, complete_gee) - beta
        mse_complete_gee[j, :] = mse_rows(coefs_gee, complete_gee, beta)
        nonsep_mask = (
            se_ok(ratio_gee, "lt")
            & se_ok(ratio_pgee, "lt")
            & complete_cases(ratio_gee)
            & complete_cases(ratio_pgee)
            & arr(env, "conv_geePK").astype(bool)
        )
        nonsep = idx_from_mask(nonsep_mask)
        print(len(nonsep))
        report_se_variance(env, nonsep)
        bias_nonsep_gee[j, :] = mean_rows_by_index(coefs_gee, nonsep) - beta
        mse_nonsep_gee[j, :] = mse_rows_by_index(coefs_gee, nonsep, beta)
        bias_nonsep_pgee[j, :] = mean_rows_by_index(coefs_pgee, nonsep) - beta
        mse_nonsep_pgee[j, :] = mse_rows_by_index(coefs_pgee, nonsep, beta)
        sep = idx_from_mask(
            se_ok(ratio_gee, "gt")
            & complete_cases(ratio_gee)
            & complete_cases(ratio_pgee)
        )
        if (not beta0_sep_guard) or len(sep) > 1:
            bias_sep_gee[j, :] = mean_rows_by_index(coefs_gee, sep) - beta
            mse_sep_gee[j, :] = mse_rows_by_index(coefs_gee, sep, beta)
            bias_sep_pgee[j, :] = mean_rows_by_index(coefs_pgee, sep) - beta
            mse_sep_pgee[j, :] = mse_rows_by_index(coefs_pgee, sep, beta)
        print("--unconditional--")
        complete_pgee = (
            complete_cases(ratio_pgee)
            & arr(env, "conv_PgeePK").astype(bool)
            & se_ok(ratio_pgee, "lt")
        )
        bias_complete_pgee[j, :] = mean_rows(coefs_pgee, complete_pgee) - beta
        mse_complete_pgee[j, :] = mse_rows(coefs_pgee, complete_pgee, beta)
        print(int(np.sum(complete_pgee)))
        print("mean SE^2 PGEE")
        show(np.mean(arr(env, "SEs_PgeePK")[complete_pgee, :] ** 2, axis=0))
        print("variance coefs PGEE")
        show(np.std(coefs_pgee[complete_pgee, :], axis=0, ddof=1) ** 2)
        show(
            np.mean(arr(env, "SEs_PgeePK")[complete_pgee, :] ** 2, axis=0)
            / (np.std(coefs_pgee[complete_pgee, :], axis=0, ddof=1) ** 2)
        )
        print(j + 1)
    plot_bias(
        build_bias_frame(
            values,
            bias_nonsep_gee,
            bias_nonsep_pgee if xname == "alpha" else bias_complete_pgee,
            xname,
        ),
        xname,
    )
    return (
        bias_complete_gee,
        mse_complete_gee,
        bias_complete_pgee,
        mse_complete_pgee,
        bias_nonsep_gee,
        mse_nonsep_gee,
        bias_nonsep_pgee,
        mse_nonsep_pgee,
        bias_sep_gee,
        mse_sep_gee,
        bias_sep_pgee,
        mse_sep_pgee,
    )


def scatter_density(
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    hline: float | None = None,
    vline: float | None = None,
    size: tuple[float, float] = (7, 7),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=size)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) > 2:
        try:
            z = stats.gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        except Exception:
            z = None
        (
            ax.scatter(x, y, c=z, cmap="viridis", alpha=0.3, s=12)
            if z is not None
            else ax.scatter(x, y, alpha=0.3, s=12)
        )
    else:
        ax.scatter(x, y, alpha=0.3, s=12)
    ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.5)
    if hline is not None:
        ax.axhline(hline, linewidth=0.5)
    if vline is not None:
        ax.axvline(vline, linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def zsep_plots() -> None:
    env = load_result(GEEDIR / "alpha3.RData")
    ratio_gee = arr(env, "SEs_model_ratio_geePK")
    ratio_pgee = arr(env, "SEs_model_ratio_PgeePK")
    coefs_gee = arr(env, "coefs_geePK")
    coefs_pgee = arr(env, "coefs_PgeePK")
    ses_gee = arr(env, "SEs_geePK")
    ses_pgee = arr(env, "SEs_PgeePK")
    sep = idx_from_mask(
        se_ok(ratio_gee, "gt") & complete_cases(ratio_gee) & complete_cases(ratio_pgee)
    )
    print(len(sep))
    scatter_density(
        coefs_gee[sep, 1],
        coefs_pgee[sep, 1],
        GEEDIR / "plots/estimates_binary_sep_scatter.pdf",
        "GEE beta_b",
        "PGEE beta_b",
        (-30, 30),
        (-30, 30),
    )
    zdat = pd.DataFrame(
        {
            "x": coefs_gee[sep, 1] / ses_gee[sep, 1],
            "y": coefs_pgee[sep, 1] / ses_pgee[sep, 1],
        }
    )
    show(zdat.describe())
    scatter_density(
        zdat["x"].to_numpy(),
        zdat["y"].to_numpy(),
        GEEDIR / "plots/zscores_binary_sep_scatter.pdf",
        "GEE z_b",
        "PGEE z_b",
        (-120, 120),
        (-5, 5),
    )
    complete = idx_from_mask(
        se_ok(ratio_gee, "le")
        & complete_cases(ratio_gee)
        & complete_cases(ratio_pgee)
        & arr(env, "conv_geePK").astype(bool)
    )
    print(len(complete))
    scatter_density(
        coefs_gee[complete, 1],
        coefs_pgee[complete, 1],
        GEEDIR / "plots/estimates_binary_complete_scatter_vol2.pdf",
        "RR-GEE beta_b",
        "RR-PGEE beta_b",
        (-5, 5),
        (-5, 5),
        hline=1.6,
        vline=1.6,
        size=(6, 7),
    )
    scatter_density(
        ses_gee[complete, 1],
        ses_pgee[complete, 1],
        GEEDIR / "plots/SEs_binary_complete_scatter_vol2.pdf",
        "RR-GEE sandwich se beta_b",
        "RR-PGEE sandwich se beta_b",
        (0, 2),
        (0, 2),
        size=(6, 7),
    )
    scatter_density(
        coefs_gee[complete, 1] / ses_gee[complete, 1],
        coefs_pgee[complete, 1] / ses_pgee[complete, 1],
        GEEDIR / "plots/zscores_binary_complete_scatter_vol2.pdf",
        "RR-GEE z_b",
        "RR-PGEE z_b",
        (-5, 5),
        (-5, 5),
        size=(6, 7),
    )
    sep_null = idx_from_mask(
        se_ok(ratio_gee, "lt")
        & complete_cases(ratio_gee)
        & ~arr(env, "conv_geePK").astype(bool)
    )
    print(len(sep_null))
    show(coefs_gee[sep_null, :])
    show(arr(env, "SEs_model_geePK")[sep_null, :])
    show(coefs_pgee[sep_null, :])
    if "tabdat" in env:
        show(np.asarray(env["tabdat"], dtype=object)[sep_null])
    print(
        np.mean(
            np.abs(coefs_gee[complete, 1] - 1.6) / ses_gee[complete, 1]
            >= stats.t.ppf(0.975, 47)
        )
    )
    print(
        np.mean(
            np.abs(coefs_pgee[complete, 1] - 1.6) / ses_pgee[complete, 1]
            >= stats.t.ppf(0.975, 47)
        )
    )
    se_ratio = pd.concat(
        [
            pd.DataFrame({"x": ratio_gee[:, 1], "group": "RR-GEE"}),
            pd.DataFrame({"x": ratio_pgee[:, 1], "group": "RR-PGEE"}),
        ]
    )
    g = sns.displot(data=se_ratio, x="x", col="group", bins=30)
    g.set(xlim=(0, 10), xlabel="BEC beta_b")
    (GEEDIR / "plots").mkdir(parents=True, exist_ok=True)
    g.fig.savefig(GEEDIR / "plots/SEratios_betaB_histogram.pdf")
    plt.close(g.fig)


def main() -> int:
    diagnostic_counts("alpha", 7)
    alpha_results = summarize_series(
        "alpha", 7, [BETA.copy() for _ in range(7)], ALPHAS, "alpha"
    )
    show(alpha_results[0])
    show(alpha_results[1])
    show(alpha_results[2][:, 1])
    show(alpha_results[3][:, 1])
    show(alpha_results[4][:, 1])
    show(alpha_results[6][:, 1])
    show(alpha_results[5][:, 1])
    show(alpha_results[7][:, 1])
    env = load_result(GEEDIR / "alpha4.RData")
    idx = idx_from_mask(
        se_ok(arr(env, "SEs_model_ratio_geePK"), "lt")
        & complete_cases(arr(env, "SEs_model_ratio_geePK"))
        & complete_cases(arr(env, "SEs_model_ratio_PgeePK"))
        & arr(env, "conv_geePK").astype(bool)
    )
    print(len(idx))
    show(np.mean(arr(env, "SEs_geePK")[idx, :], axis=0))
    show(np.std(arr(env, "coefs_geePK")[idx, :], axis=0, ddof=1))
    show(
        np.mean(arr(env, "SEs_geePK")[idx, :], axis=0)
        / np.std(arr(env, "coefs_geePK")[idx, :], axis=0, ddof=1)
    )
    show(np.mean(arr(env, "SEs_PgeePK")[idx, :], axis=0))
    show(np.std(arr(env, "coefs_PgeePK")[idx, :], axis=0, ddof=1))
    show(
        np.mean(arr(env, "SEs_PgeePK")[idx, :], axis=0)
        / np.std(arr(env, "coefs_PgeePK")[idx, :], axis=0, ddof=1)
    )
    zsep_plots()
    diagnostic_counts("BetaB_", 5)
    beta_b_results = summarize_series(
        "BetaB_", 5, [np.array([-4.0, b, 0.2]) for b in BETA_B], BETA_B, "beta"
    )
    show(beta_b_results[8][:, 1])
    show(beta_b_results[10][:, 1])
    show(beta_b_results[9][:, 1])
    show(beta_b_results[11][:, 1])
    show(beta_b_results[4][:, 1])
    show(beta_b_results[6][:, 1])
    show(beta_b_results[5][:, 1])
    show(beta_b_results[7][:, 1])
    show(beta_b_results[2][:, 1])
    show(beta_b_results[3][:, 1])
    diagnostic_counts("N", 6)
    n_results = summarize_series("N", 6, [BETA.copy() for _ in range(6)], NS, "N")
    show(n_results[0])
    show(n_results[1])
    show(n_results[4][:, 1])
    show(n_results[6][:, 1])
    show(n_results[5][:, 1])
    show(n_results[7][:, 1])
    show(n_results[2][:, 1])
    show(n_results[3][:, 1])
    diagnostic_counts("Beta0_", 3)
    beta0_results = summarize_series(
        "Beta0_",
        3,
        [np.array([b0, 1.6, 0.2]) for b0 in BETA0],
        BETA0,
        "beta0",
        beta0_sep_guard=True,
    )
    show(beta0_results[0])
    show(beta0_results[1])
    show(beta0_results[4][:, 1])
    show(beta0_results[6][:, 1])
    show(beta0_results[5][:, 1])
    show(beta0_results[7][:, 1])
    show(beta0_results[2][:, 1])
    show(beta0_results[3][:, 1])
    diagnostic_counts("gamma", 7)
    gamma_results = summarize_series(
        "gamma",
        7,
        [BETA.copy() for _ in range(7)],
        GAMMAS,
        "gamma",
        load_prefix="N100_gamma",
    )
    show(gamma_results[0])
    show(gamma_results[1])
    show(gamma_results[4][:, 1])
    show(gamma_results[6][:, 1])
    show(gamma_results[5][:, 1])
    show(gamma_results[7][:, 1])
    show(gamma_results[2][:, 1])
    show(gamma_results[3][:, 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
