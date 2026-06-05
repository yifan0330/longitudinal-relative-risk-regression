#!/usr/bin/env python3
"""Python translation of UKB_logPoissonGEE.Rmd analysis chunks."""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_io import load_pickle_data
from gee_logPoisson_dispersion_fn import gee_dispersion_run
from gee_logPoisson_fn import gee_run
from gee_logPoisson_penalty_fn import gee_penalty_run
from GEE_logPoisson_run import (
    GEEDIR,
    TEMPDIR,
    design_matrix,
    load_ids,
    load_lesions,
    load_visits,
)

PGEEDIR = PROJECT_ROOT / 'PGEE_Mondol'
IMAGEDIR_VIS1 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis'
IMAGEDIR_VIS2 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis'
BRAIN_MASK = Path(
    str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii')
)
NAMES_COVS = ["Intercept", "avg_age", "age_diff", "sexM"]


def read_nifti(path: Path | str) -> np.ndarray:
    p = Path(path)
    candidates = [p]
    if not p.suffix:
        candidates += [Path(str(p) + ".nii.gz"), Path(str(p) + ".nii")]
    for candidate in candidates:
        if candidate.exists():
            return np.asarray(nib.load(str(candidate)).get_fdata())
    raise FileNotFoundError(p)


def load_output(path: Path):
    return load_pickle_data(path)


def load_voxel_ids() -> np.ndarray:
    return (
        pd.read_csv(TEMPDIR / "voxel_IDs_atleast6.dat", sep=r"\s+", header=None)
        .iloc[:, 0]
        .to_numpy(int)
    )


def flat_values(image: np.ndarray, voxel_ids: np.ndarray) -> np.ndarray:
    return np.asarray(image).ravel(order="F")[
        np.asarray(voxel_ids, dtype=int).reshape(-1) - 1
    ]


def summary_PK(summary_vec, quantiles_vec):
    arr = np.asarray(summary_vec, dtype=float).reshape(-1)
    missing = np.isnan(arr)
    print("Missing values")
    print(int(missing.sum()))
    print("---")
    arr = arr[~missing]
    return {
        "quantiles": (
            np.quantile(arr, quantiles_vec)
            if arr.size
            else np.full(len(quantiles_vec), np.nan)
        ),
        "mean": float(np.mean(arr)) if arr.size else np.nan,
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        "zeroes": int(np.sum(arr == 0)),
    }


def describe(x):
    return pd.Series(np.asarray(x, dtype=float).reshape(-1)).describe(
        percentiles=[0.25, 0.5, 0.75]
    )


def map_summary(
    result_dir: Path,
    prefix: str,
    cov: str,
    voxel_ids: np.ndarray,
    quantiles=(0, 0.25, 0.5, 0.75, 0.99, 1),
):
    data = read_nifti(result_dir / f"{prefix}_{cov}_GEE")
    out = summary_PK(flat_values(data, voxel_ids), quantiles)
    print(result_dir, prefix, cov, out)
    return data, out


def se_trace_last(
    output_all, key="beta_se_model_trace", n_cov=4, expected_len=(6, 8, 9)
) -> np.ndarray:
    cols = []
    for item in output_all:
        y = None
        if isinstance(item, dict) and key in item:
            y = np.asarray(item[key], dtype=float)
        elif hasattr(item, "__len__") and len(item) in expected_len:
            for candidate in (2, 4):
                try:
                    maybe = item[candidate]
                    y = np.asarray(
                        maybe[key] if isinstance(maybe, dict) else maybe, dtype=float
                    )
                    break
                except Exception:
                    pass
        if y is None or y.size == 0:
            cols.append(np.full(n_cov, np.nan))
            continue
        if y.ndim == 1:
            y = y.reshape(-1, n_cov)
        cols.append((y / y[0, :])[-1, :])
    return np.column_stack(cols) if cols else np.empty((n_cov, 0))


def make_panel_from_arrays(
    i: int,
    lesions_vis1: np.ndarray,
    lesions_vis2: np.ndarray,
    ids: np.ndarray,
    df_visits: pd.DataFrame,
) -> pd.DataFrame:
    n_subj = lesions_vis1.shape[1]
    panel = pd.DataFrame(
        {
            "y": np.r_[lesions_vis1[i], lesions_vis2[i]],
            "vis": np.r_[np.ones(n_subj, int), np.full(n_subj, 2, int)],
            "eid_8107": np.r_[ids, ids].astype(str),
        }
    )
    panel = panel.merge(df_visits, on="eid_8107", how="left")
    panel["age_diff_vis2"] = panel["age_diff_vis2"].to_numpy() * np.tile(
        [1, -1], n_subj
    )
    return panel.sort_values(["eid_8107", "vis"]).reset_index(drop=True)


def fit_statsmodels_poisson(panel: pd.DataFrame):
    X = design_matrix(panel)
    groups = np.repeat(
        np.arange(len(panel) // panel["vis"].nunique()), panel["vis"].nunique()
    )
    model = sm.GEE(
        panel["y"].to_numpy(),
        X,
        groups=groups,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(maxiter=10)


def plot_scatter(
    x, y, title, out_path, xlim=(0, 10), ylim=(0, 10), xlabel=None, ylabel=None
):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=4, marker=".")
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def inspect_voxel(lesions_vis1, lesions_vis2, idx, ids, df_visits, mode="original"):
    panel = make_panel_from_arrays(idx, lesions_vis1, lesions_vis2, ids, df_visits)
    print(pd.crosstab(panel["y"], panel["vis"]))
    print(pd.crosstab(panel["y"], panel["sexM"]))
    n_visits = panel["vis"].nunique()
    n_subj = len(panel) // n_visits
    X = design_matrix(panel)
    if mode == "penalty":
        return gee_penalty_run(
            panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable"
        )
    if mode == "dispersion":
        return gee_dispersion_run(
            panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable"
        )
    return gee_run(
        panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable"
    )


def separation_flags(
    lesions_vis1: np.ndarray,
    lesions_vis2: np.ndarray,
    ids: np.ndarray,
    df_visits: pd.DataFrame,
):
    sex_lookup = df_visits.set_index("eid_8107")["sexM"]
    sex = pd.Series(ids.astype(str)).map(sex_lookup).to_numpy()
    sex_sep = np.zeros(lesions_vis1.shape[0], dtype=bool)
    visit_sep = np.zeros(lesions_vis1.shape[0], dtype=bool)
    for i in range(lesions_vis1.shape[0]):
        y = np.r_[lesions_vis1[i], lesions_vis2[i]]
        s = np.r_[sex, sex]
        v = np.r_[
            np.ones(lesions_vis1.shape[1], int), np.full(lesions_vis1.shape[1], 2, int)
        ]
        f_les = np.sum((y == 1) & (s == 0))
        m_les = np.sum((y == 1) & (s == 1))
        v1_les = np.sum((y == 1) & (v == 1))
        v2_les = np.sum((y == 1) & (v == 2))
        sex_sep[i] = (f_les == 0 and m_les > 0) or (f_les > 0 and m_les == 0)
        visit_sep[i] = (v1_les == 0 and v2_les > 0) or (v1_les > 0 and v2_les == 0)
        if (i + 1) % 1000 == 0:
            print(i + 1)
    return sex_sep, visit_sep


def main() -> int:
    voxel_ids = load_voxel_ids()
    brain_mask = read_nifti(BRAIN_MASK)
    empir_prob_vis1 = read_nifti(IMAGEDIR_VIS1 / "empir_prob_mask.nii.gz")
    empir_prob_vis2 = read_nifti(IMAGEDIR_VIS2 / "empir_prob_mask.nii.gz")

    for result in [
        "results_poisson_sexM_independent",
        "results_poisson_sexM_independent_geePK",
        "results_poisson_sexM_exch_geePK",
    ]:
        for cov in NAMES_COVS:
            map_summary(GEEDIR / result, "se", cov, voxel_ids)
    alpha_map = read_nifti(GEEDIR / "results_poisson_sexM_exch_geePK" / "alpha_GEE")
    print(summary_PK(flat_values(alpha_map, voxel_ids), [0, 0.25, 0.5, 0.75, 0.99, 1]))
    print(int(np.sum(alpha_map[~np.isnan(alpha_map)] > 1)))
    iter_map = read_nifti(GEEDIR / "results_poisson_sexM_exch_geePK" / "iterations_GEE")
    print(summary_PK(flat_values(iter_map, voxel_ids), [0, 0.25, 0.5, 0.75, 0.99, 1]))
    print(int(np.sum(iter_map[~np.isnan(iter_map)] == 10)))

    se_intercept = read_nifti(
        GEEDIR / "results_poisson_sexM_exch_geePK" / "se_Intercept_GEE"
    )
    intercept_potential = np.where(flat_values(se_intercept, voxel_ids) > 6.92)[0]
    print(len(intercept_potential))
    output_all = load_output(
        GEEDIR / "results_poisson_sexM_exch_geePK" / "results_exch_GEE.pkl"
    ).get("output_all", [])
    if len(intercept_potential):
        print(output_all[intercept_potential[0]])
    print(int(np.sum(flat_values(iter_map, voxel_ids)[intercept_potential] == 10)))

    NA_ids = np.where(np.isnan(flat_values(iter_map, voxel_ids)))[0]
    print(
        summary_PK(
            flat_values(empir_prob_vis1, voxel_ids[NA_ids]),
            [0, 0.25, 0.5, 0.75, 0.99, 1],
        )
    )
    print(
        summary_PK(
            flat_values(empir_prob_vis2, voxel_ids[NA_ids]),
            [0, 0.25, 0.5, 0.75, 0.99, 1],
        )
    )

    ids = load_ids()
    df_visits = load_visits()
    lesions_vis1, lesions_vis2 = load_lesions()
    if len(NA_ids):
        model = inspect_voxel(
            lesions_vis1[NA_ids], lesions_vis2[NA_ids], 0, ids, df_visits
        )
        print(model["beta"], model["beta_se_sandwich"], model["alpha"])
        model_phi = inspect_voxel(
            lesions_vis1[NA_ids],
            lesions_vis2[NA_ids],
            0,
            ids,
            df_visits,
            mode="dispersion",
        )
        print(
            model_phi["beta"],
            model_phi["beta_se_sandwich"],
            model_phi["alpha"],
            1 / model_phi["phi"],
        )

    age_map = read_nifti(GEEDIR / "results_poisson_sexM_exch_geePK" / "se_avg_age_GEE")
    time_map = read_nifti(
        GEEDIR / "results_poisson_sexM_exch_geePK" / "se_age_diff_GEE"
    )
    NA_ids_time = np.where(
        np.isnan(flat_values(time_map, voxel_ids))
        & ~np.isnan(flat_values(age_map, voxel_ids))
    )[0]
    time_est = read_nifti(
        GEEDIR / "results_poisson_sexM_exch_geePK" / "estimate_age_diff_GEE"
    )
    sexM_est = read_nifti(
        GEEDIR / "results_poisson_sexM_exch_geePK" / "estimate_sexM_GEE"
    )
    print(flat_values(time_est, voxel_ids[NA_ids_time]))
    print(flat_values(sexM_est, voxel_ids[NA_ids_time]))
    if len(NA_ids_time):
        same = np.sum(
            np.all(lesions_vis1[NA_ids_time] == lesions_vis2[NA_ids_time], axis=1)
        )
        print(same)
        inspect_voxel(
            lesions_vis1[NA_ids_time],
            lesions_vis2[NA_ids_time],
            min(1, len(NA_ids_time) - 1),
            ids,
            df_visits,
        )

    alpha_map2 = read_nifti(GEEDIR / "results_poisson_sexM_exch_geePK" / "alpha_GEE")
    NA_ids_age = np.where(
        np.isnan(flat_values(age_map, voxel_ids))
        & ~np.isnan(flat_values(alpha_map2, voxel_ids))
    )[0]
    print(flat_values(alpha_map2, voxel_ids[NA_ids_age]))

    last_sandwich = se_trace_last(output_all, key="beta_se_sandwich_trace")
    for j in range(4):
        print(summary_PK(last_sandwich[j], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    idx_SEratio = np.where(last_sandwich[0] > 3.4)[0]
    intersect_SEcheck = np.intersect1d(intercept_potential, idx_SEratio)
    if len(intersect_SEcheck):
        gee_model = inspect_voxel(
            lesions_vis1[intersect_SEcheck],
            lesions_vis2[intersect_SEcheck],
            min(3, len(intersect_SEcheck) - 1),
            ids,
            df_visits,
        )
        print(gee_model["beta_se_model"], gee_model["beta_se_sandwich"])

    sex_separated, visit_separated = separation_flags(
        lesions_vis1, lesions_vis2, ids, df_visits
    )
    print(
        int(np.sum(sex_separated)),
        int(np.sum(visit_separated)),
        int(np.sum(sex_separated & visit_separated)),
    )
    print(np.intersect1d(np.where(sex_separated)[0], idx_SEratio))

    output_all_max25 = load_output(
        GEEDIR / "results_poisson_sexM_exch_geePK_max25" / "results_exch_GEE.pkl"
    ).get("output_all", [])
    last_model = se_trace_last(
        output_all_max25, key="beta_se_model_trace", expected_len=(8, 9)
    )
    for j in range(4):
        print(summary_PK(last_model[j], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    idx_max25 = np.where(np.isnan(last_model[0]))[0]
    if len(intercept_potential):
        penalty_model = inspect_voxel(
            lesions_vis1[intercept_potential],
            lesions_vis2[intercept_potential],
            0,
            ids,
            df_visits,
            mode="penalty",
        )
        orig_model = inspect_voxel(
            lesions_vis1[intercept_potential],
            lesions_vis2[intercept_potential],
            0,
            ids,
            df_visits,
        )
        print(penalty_model["beta"], orig_model["beta"])
        print(penalty_model["beta_se_sandwich"], orig_model["beta_se_sandwich"])
        print(penalty_model["iterations"], orig_model["iterations"])
        print(penalty_model["alpha"], orig_model["alpha"])

    est_map = read_nifti(
        GEEDIR / "results_poisson_sexM_exch_geePK_max25" / "estimate_sexM_GEE"
    )
    dvrg_idx = sex_separated
    print(
        summary_PK(
            flat_values(est_map, voxel_ids)[dvrg_idx], [0, 0.25, 0.5, 0.75, 0.99, 1]
        )
    )
    pgee_est = read_nifti(
        GEEDIR / "results_penalty_poisson_sexM_exch_geePK_max25" / "estimate_sexM_GEE"
    )
    print(
        summary_PK(
            flat_values(pgee_est, voxel_ids)[dvrg_idx], [0, 0.25, 0.5, 0.75, 0.99, 1]
        )
    )
    _ = brain_mask, idx_max25, PGEEDIR
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
