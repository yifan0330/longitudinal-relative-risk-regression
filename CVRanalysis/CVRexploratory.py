#!/usr/bin/env python3
"""Python translation of CVRanalysis/CVRexploratory.Rmd.

Runs the CVR exploratory summaries and diagnostic plots without executing or
embedding any R code.  It expects the same input files and writes the same
explicit PDF outputs used by the original analysis.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from Sept21_pgee_logPoisson_dispersion_fn import (  # noqa: E402
    GEEDIR,
    IMAGEDIR_VIS1,
    IMAGEDIR_VIS2,
    TEMPDIR,
    load_rdata,
)

BRAIN_MASK_PATH = Path(
    "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii"
)
MNI152_PATH = Path("/well/nichols/users/kindalov/FMRIB/MNI152_T1_2mm_brain.nii.gz")
NAMES_COVS = [
    "Intercept",
    "baseAge",
    "ageDiff",
    "baseCVR",
    "CVRdiff",
    "sexM",
    "headsize",
    "ageBYageDiff",
    "ageBYsexM",
]
QUANTILES_FULL = [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]


def resolve_nifti_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path]
    if not str(path).endswith(".nii") and not str(path).endswith(".nii.gz"):
        candidates.extend([Path(f"{path}.nii.gz"), Path(f"{path}.nii")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def read_nifti(path: str | Path) -> np.ndarray:
    return np.asarray(nib.load(str(resolve_nifti_path(path))).get_fdata())


def r_linear_get(data: np.ndarray, ids_1based: np.ndarray) -> np.ndarray:
    return np.asarray(data).ravel(order="F")[
        np.asarray(ids_1based, dtype=int).reshape(-1) - 1
    ]


def summary_PK(summary_vec: Any, quantiles_vec: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(summary_vec, dtype=float).reshape(-1)
    missing_values = ~np.isfinite(values)
    print("Missing values")
    print(int(missing_values.sum()))
    print("---")
    values = values[~missing_values]
    if values.size == 0:
        output_list = {
            "quantiles": pd.Series(np.nan, index=list(quantiles_vec)),
            "mean": np.nan,
            "sd": np.nan,
            "zeroes": 0,
        }
    else:
        output_list = {
            "quantiles": pd.Series(
                np.quantile(values, quantiles_vec), index=list(quantiles_vec)
            ),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
            "zeroes": int(np.sum(values == 0)),
        }
    print(output_list)
    return output_list


def r_summary(values: Any) -> pd.Series:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = pd.Series(
            {
                "Min.": np.nan,
                "1st Qu.": np.nan,
                "Median": np.nan,
                "Mean": np.nan,
                "3rd Qu.": np.nan,
                "Max.": np.nan,
            }
        )
    else:
        out = pd.Series(
            {
                "Min.": np.min(arr),
                "1st Qu.": np.quantile(arr, 0.25),
                "Median": np.median(arr),
                "Mean": np.mean(arr),
                "3rd Qu.": np.quantile(arr, 0.75),
                "Max.": np.max(arr),
            }
        )
    print(out)
    return out


def r_table(values: Any) -> pd.Series:
    arr = pd.Series(np.asarray(values).reshape(-1))
    out = arr.value_counts(dropna=False).sort_index()
    print(out)
    return out


def _payload_object(payload: Any, name: str) -> Any:
    if isinstance(payload, Mapping):
        if name in payload:
            return payload[name]
        if len(payload) == 1:
            return next(iter(payload.values()))
    raise KeyError(f"{name} not found in loaded data")


def _item_len(item: Any) -> int:
    if isinstance(item, Mapping):
        return len(item)
    if isinstance(item, (str, bytes)):
        return 0
    try:
        return len(item)
    except TypeError:
        return 0


def _field(item: Any, key: str, one_based_index: int) -> Any:
    if isinstance(item, Mapping):
        if key in item:
            return item[key]
        keys = list(item.keys())
        if len(keys) >= one_based_index:
            return item[keys[one_based_index - 1]]
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) >= one_based_index:
            return item[one_based_index - 1]
    raise KeyError(key)


def _as_numeric_array(value: Any) -> np.ndarray:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.to_numpy()
    return np.asarray(value, dtype=float)


def _se_trace_ratios(
    output_all: Sequence[Any], expected_len: int, n_cov: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    ratios = np.full((n_cov, len(output_all)), np.nan)
    iterations = np.full(len(output_all), np.nan)
    for idx, item in enumerate(output_all):
        has_named_trace = isinstance(item, Mapping) and "beta_se_model_trace" in item
        if not has_named_trace and _item_len(item) != expected_len:
            continue
        try:
            trace = _as_numeric_array(_field(item, "beta_se_model_trace", 3))
            if trace.size == 0:
                continue
            trace = (
                trace.reshape((-1, n_cov), order="F")
                if trace.ndim == 1
                else np.asarray(trace, dtype=float)
            )
            if trace.shape[1] != n_cov and trace.shape[0] == n_cov:
                trace = trace.T
            if trace.shape[1] != n_cov:
                trace = trace.reshape((-1, n_cov), order="F")
            first = trace[0, :]
            ratio_trace = trace / first
            ratios[:, idx] = ratio_trace[-1, :]
            iterations[idx] = np.asarray(
                _field(item, "iterations", 7), dtype=float
            ).reshape(-1)[0]
        except Exception:
            continue
    return ratios, iterations


def _load_output_all(path: Path, name: str = "output_all") -> Sequence[Any]:
    payload = load_rdata(path)
    output_all = _payload_object(payload, name)
    if isinstance(output_all, np.ndarray):
        return list(output_all.reshape(-1))
    if isinstance(output_all, pd.DataFrame):
        return output_all.to_dict("records")
    return list(output_all)


def _covariate_flags(se_ratios: np.ndarray, threshold: float) -> list[np.ndarray]:
    flags = []
    for i in range(se_ratios.shape[0]):
        flag = se_ratios[i, :] > threshold
        flag[~np.isfinite(flag)] = False
        flags.append(flag)
        print(int(flag.sum()))
    return flags


def _diagnostic_hist(values: np.ndarray, title: str, out: Path | None = None) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(np.asarray(values, dtype=float)[np.isfinite(values)], bins=50)
    plt.title(title)
    plt.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
    plt.close()


def _histogram_se_ratios(se_ratios_gee: np.ndarray, se_ratios_pgee: np.ndarray) -> None:
    dat_SEratio = pd.concat(
        [
            pd.DataFrame({"x": se_ratios_gee[5, :], "group": "RR-GEE"}),
            pd.DataFrame({"x": se_ratios_pgee[5, :], "group": "RR-PGEE"}),
        ],
        ignore_index=True,
    )
    out = GEEDIR / "plots" / "geeVSpgee" / "Sept_SEratios_sex_histogram.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = sns.FacetGrid(
        dat_SEratio.replace([np.inf, -np.inf], np.nan).dropna(),
        col="group",
        height=5,
        aspect=1,
    )
    grid.map_dataframe(sns.histplot, x="x", bins=30)
    grid.set(xlim=(0, 10), xlabel="BEC(beta[6])")
    grid.tight_layout()
    grid.savefig(out)
    plt.close("all")


def _density_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out: Path,
    xlabel: str,
    ylabel: str,
    limits: tuple[float, float],
) -> None:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    if not df.empty:
        plt.hexbin(
            df["x"],
            df["y"],
            gridsize=100,
            cmap="viridis",
            mincnt=1,
            bins="log",
            alpha=0.75,
        )
        plt.colorbar(label="log10(count)")
    lo, hi = limits
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=0.5)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _analyse_se_ratios(
    label: str, result_path: Path, expected_len: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"# {label}")
    output_all = _load_output_all(result_path)
    se_trace_last_iter, iters = _se_trace_ratios(output_all, expected_len=expected_len)
    for i, cov in enumerate(NAMES_COVS):
        print(cov)
        summary_PK(se_trace_last_iter[i, :], QUANTILES_FULL)
    _diagnostic_hist(
        se_trace_last_iter[5, :],
        f"SE ratio: Sex",
        GEEDIR / "plots" / "geeVSpgee" / f"{label}_SEratio_sex_histogram.pdf",
    )
    cov_flags = _covariate_flags(se_trace_last_iter, threshold=threshold)
    all_flags = np.logical_and.reduce(cov_flags)
    print(np.where(all_flags)[0] + 1)
    idx_temp = np.where(
        cov_flags[0]
        | cov_flags[1]
        | cov_flags[2]
        | cov_flags[3]
        | cov_flags[4]
        | cov_flags[5]
        | cov_flags[6]
        | cov_flags[7]
        | cov_flags[8]
    )[0]
    print(len(idx_temp))
    return se_trace_last_iter, iters, idx_temp


def main() -> None:
    brain_mask = read_nifti(BRAIN_MASK_PATH)
    _ = read_nifti(MNI152_PATH)
    _ = read_nifti(IMAGEDIR_VIS1 / "CVR_empir_prob_mask.nii.gz")
    _ = read_nifti(IMAGEDIR_VIS2 / "CVR_empir_prob_mask.nii.gz")
    voxel_IDs = np.loadtxt(TEMPDIR / "voxel_IDs_CVR.dat", dtype=int).reshape(-1)

    se_ratios_GEE, iters_gee, idx_temp_gee = _analyse_se_ratios(
        "gee",
        GEEDIR / "results_July_gee_interaction" / "results_CVR_GEE_1578subjs.Rdata",
        expected_len=8,
        threshold=10,
    )

    sexM_est = read_nifti(
        GEEDIR / "results_July_gee_interaction" / f"estimate_{NAMES_COVS[5]}_GEE"
    )
    summary_PK(r_linear_get(sexM_est, voxel_IDs), [0, 0.25, 0.5, 0.75, 0.99, 1])
    exclude = np.where(se_ratios_GEE[3, :] > 10)[0]
    keep_voxels = np.delete(voxel_IDs, exclude)
    summary_PK(r_linear_get(sexM_est, keep_voxels), [0, 0.25, 0.5, 0.75, 0.99, 1])
    for i, cov in enumerate(NAMES_COVS):
        print(i + 1)
        print(cov)
        summary_PK(np.delete(se_ratios_GEE[i, :], idx_temp_gee), [0, 0.5, 1])
        print(float(np.nanstd(np.delete(se_ratios_GEE[i, :], idx_temp_gee), ddof=1)))
    _ = read_nifti(GEEDIR / "results_July_gee_interaction" / f"se_{NAMES_COVS[5]}_GEE")
    summary_PK(r_linear_get(sexM_est, voxel_IDs), [0, 0.25, 0.5, 0.75, 0.99, 1])
    _diagnostic_hist(
        np.delete(se_ratios_GEE[5, :], idx_temp_gee), "SE ratio: Sex, filtered"
    )
    NA_ids_gee = np.where(~np.isfinite(se_ratios_GEE[5, :]))[0] + 1
    print(f"NA_ids_gee: {len(NA_ids_gee)}")
    r_summary(iters_gee)
    r_table(iters_gee)
    r_summary(iters_gee[idx_temp_gee] < 25)
    print(se_ratios_GEE.shape)

    se_ratios_PGEE, iters_pgee, idx_temp_pgee = _analyse_se_ratios(
        "pgee",
        GEEDIR / "results_Sept_pgee_interaction" / "results_CVR_PGEE_1578subjs.Rdata",
        expected_len=9,
        threshold=20,
    )
    NA_ids_pgee = np.where(~np.isfinite(se_ratios_PGEE[5, :]))[0] + 1
    print(f"NA_ids_pgee: {len(NA_ids_pgee)}")
    r_summary(iters_pgee)
    r_table(iters_pgee)
    r_summary(iters_pgee[idx_temp_pgee] < 25)

    _histogram_se_ratios(se_ratios_GEE, se_ratios_PGEE)

    z_sex_gee = read_nifti(
        GEEDIR / "results_July_gee_interaction" / f"zscore_{NAMES_COVS[5]}_GEE.nii.gz"
    )
    z_sex_pgee = read_nifti(
        GEEDIR / "results_Sept_pgee_interaction" / f"zscore_{NAMES_COVS[5]}_GEE.nii.gz"
    )
    _density_scatter(
        r_linear_get(z_sex_gee, voxel_IDs[idx_temp_gee]),
        r_linear_get(z_sex_pgee, voxel_IDs[idx_temp_gee]),
        GEEDIR / "plots" / "geeVSpgee" / "Sept_zscores_sex_sep_scatter.pdf",
        "GEE Sex z-scores",
        "PGEE Sex z-scores",
        (-50, 50),
    )
    nonsep = np.delete(np.arange(len(voxel_IDs)), idx_temp_gee)
    _density_scatter(
        r_linear_get(z_sex_gee, voxel_IDs[nonsep]),
        r_linear_get(z_sex_pgee, voxel_IDs[nonsep]),
        GEEDIR / "plots" / "geeVSpgee" / "Sept_zscores_sex_nonsep_scatter.pdf",
        "GEE Sex z-scores",
        "PGEE Sex z-scores",
        (-20, 20),
    )


if __name__ == "__main__":
    main()
