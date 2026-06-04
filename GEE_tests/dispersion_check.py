#!/usr/bin/env python3
"""Python translation of dispersion_check.Rmd analysis chunks."""
from __future__ import annotations

from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import statsmodels.api as sm

from GEE_logPoisson_dispersed_run import TEMPDIR, GEEDIR, load_ids, load_visits, load_lesions, design_matrix, make_panel, run_parallel, save_output
from gee_logPoisson_fn import gee_run
from gee_logPoisson_penalty_fn import gee_penalty_run
from gee_logPoisson_dispersion_fn import gee_dispersion_run

PGEEDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/PGEE_Mondol")
IMAGEDIR_VIS1 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis")
IMAGEDIR_VIS2 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis")
BRAIN_MASK = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii")
NAMES_COVS = ["Intercept", "avg_age", "age_diff", "sexM"]
RESULT_DIR = GEEDIR / "results_poisson_sexM_exch_geePK_max25"
TEMP_DIR = GEEDIR / "temp_poisson_sexM_exch_geePK_max25"


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
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            import pyreadr  # type: ignore
            return dict(pyreadr.read_r(str(path)))
        except Exception as exc:
            raise RuntimeError(f"Could not load nested result file {path}; convert it with pyreadr/rdata support") from exc


def load_voxel_ids() -> np.ndarray:
    return pd.read_csv(TEMPDIR / "voxel_IDs_atleast6.dat", sep=r"\s+", header=None).iloc[:, 0].to_numpy(int)


def flat_values(image: np.ndarray, voxel_ids: np.ndarray) -> np.ndarray:
    return np.asarray(image).ravel(order="F")[np.asarray(voxel_ids, dtype=int).reshape(-1) - 1]


def summary_PK(summary_vec, quantiles_vec):
    arr = np.asarray(summary_vec, dtype=float).reshape(-1)
    missing = np.isnan(arr)
    print("Missing values")
    print(int(missing.sum()))
    print("---")
    arr = arr[~missing]
    return {"quantiles": np.quantile(arr, quantiles_vec) if arr.size else np.full(len(quantiles_vec), np.nan), "mean": float(np.mean(arr)) if arr.size else np.nan, "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan, "zeroes": int(np.sum(arr == 0))}


def se_trace_last(output_all, key="beta_se_model_trace", n_cov=4, expected_len=(6, 8, 9)):
    traces = []
    last = []
    for item in output_all:
        y = None
        if isinstance(item, dict) and key in item:
            y = np.asarray(item[key], dtype=float)
        elif hasattr(item, "__len__") and len(item) in expected_len:
            for candidate in (2, 4):
                try:
                    maybe = item[candidate]
                    y = np.asarray(maybe[key] if isinstance(maybe, dict) else maybe, dtype=float)
                    break
                except Exception:
                    pass
        if y is None or y.size == 0:
            ratio = np.full((1, n_cov), np.nan)
        else:
            if y.ndim == 1:
                y = y.reshape(-1, n_cov)
            ratio = y / y[0, :]
        traces.append(ratio)
        last.append(ratio[-1, :])
    return traces, np.column_stack(last) if last else np.empty((n_cov, 0))


def make_panel_from_arrays(i: int, lesions_vis1: np.ndarray, lesions_vis2: np.ndarray, ids: np.ndarray, df_visits: pd.DataFrame) -> pd.DataFrame:
    n_subj = lesions_vis1.shape[1]
    panel = pd.DataFrame({"y": np.r_[lesions_vis1[i], lesions_vis2[i]], "vis": np.r_[np.ones(n_subj, int), np.full(n_subj, 2, int)], "eid_8107": np.r_[ids, ids].astype(str)})
    panel = panel.merge(df_visits, on="eid_8107", how="left")
    panel["age_diff_vis2"] = panel["age_diff_vis2"].to_numpy() * np.tile([1, -1], n_subj)
    return panel.sort_values(["eid_8107", "vis"]).reset_index(drop=True)


def fit_statsmodels_poisson(panel: pd.DataFrame, scale="X2"):
    X = design_matrix(panel)
    n_visits = panel["vis"].nunique()
    groups = np.repeat(np.arange(len(panel) // n_visits), n_visits)
    model = sm.GEE(panel["y"].to_numpy(), X, groups=groups, family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable())
    return model.fit(maxiter=10, scale=scale)


def fit_gee(i, subset_idx, ids, df_visits):
    panel = make_panel(i, subset_idx, ids, df_visits)
    n_visits = panel["vis"].nunique()
    n_subj = len(panel) // n_visits
    X = design_matrix(panel)
    model = gee_dispersion_run(panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable", tol=1e-3, max_iter=10)
    if (i + 1) % 10 == 0:
        print(i + 1)
    model["voxel"] = i + 1
    return model


def refit_subset(subset_idx, ids, df_visits, label):
    start = time.time()
    output = run_parallel(lambda i: fit_gee(i, subset_idx, ids, df_visits), len(subset_idx), 1)
    print(time.time() - start)
    save_output(output, TEMP_DIR / f"{label}.RData")
    return output


def plot_hist(values, title, out_path):
    plt.figure()
    plt.hist(np.asarray(values)[~np.isnan(values)])
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    voxel_ids = load_voxel_ids()
    brain_mask = read_nifti(BRAIN_MASK)
    empir_prob_vis1 = read_nifti(IMAGEDIR_VIS1 / "empir_prob_mask.nii.gz")
    empir_prob_vis2 = read_nifti(IMAGEDIR_VIS2 / "empir_prob_mask.nii.gz")

    output_all = load_output(RESULT_DIR / "results_exch_GEE.Rdata").get("output_all", [])
    _, se_trace_last_iter = se_trace_last(output_all, key="beta_se_model_trace", expected_len=(8, 9))
    for idx, name in enumerate(NAMES_COVS):
        print(name, summary_PK(se_trace_last_iter[idx], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    plot_hist(se_trace_last_iter[3], "SE ratio: Sex", GEEDIR / "figures" / "dispersion_se_ratio_sex.png")
    print(int(np.sum(se_trace_last_iter[3] > 10000)))

    sexM_est = read_nifti(RESULT_DIR / "estimate_sexM_GEE")
    print(summary_PK(flat_values(sexM_est, voxel_ids), [0, 0.25, 0.5, 0.75, 0.99, 1]))
    high_ratio = np.where(se_trace_last_iter[3] > 10000)[0]
    keep = np.setdiff1d(np.arange(len(voxel_ids)), high_ratio)
    print(summary_PK(flat_values(sexM_est, voxel_ids[keep]), [0, 0.25, 0.5, 0.75, 0.99, 1]))
    NA_ids = np.where(np.isnan(se_trace_last_iter[3]))[0]

    ids = load_ids(); df_visits = load_visits(); lesions_vis1, lesions_vis2 = load_lesions()
    if len(NA_ids):
        subset_lesions1 = lesions_vis1[NA_ids]
        subset_lesions2 = lesions_vis2[NA_ids]
        panel = make_panel_from_arrays(min(9, len(NA_ids) - 1), subset_lesions1, subset_lesions2, ids, df_visits)
        print(pd.crosstab(panel["y"], panel["vis"]))
        print(pd.crosstab(panel["y"], panel["sexM"]))
        n_visits = panel["vis"].nunique(); n_subj = len(panel) // n_visits; X = design_matrix(panel)
        gee_model = gee_run(panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable")
        print(gee_model)
        gee_model_phi = gee_dispersion_run(panel["y"].to_numpy(), X, n_subj, n_visits, covariance="Exchangeable")
        print("--beta--", gee_model_phi["beta"])
        print("--SE sandwich--", gee_model_phi["beta_se_sandwich"])
        print("--alpha--", gee_model_phi["alpha"])
        print("--phi--", 1 / gee_model_phi["phi"])
        try:
            print(fit_statsmodels_poisson(panel).summary())
        except Exception as exc:
            print(f"statsmodels comparison failed: {exc}")

    dispersed_file = TEMP_DIR / "dispersed.RData"
    if dispersed_file.exists():
        output_dispersed = load_output(dispersed_file).get("output_dispersed", load_output(dispersed_file))
    elif len(NA_ids):
        output_dispersed = refit_subset(NA_ids[:100], ids, df_visits, "dispersed")
    else:
        output_dispersed = []
    if output_dispersed:
        _, last_d = se_trace_last(output_dispersed, key="beta_se_model_trace", expected_len=(8, 9))
        print(summary_PK(last_d[3], [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))
        print(int(np.sum(last_d[3] > 100)), int(np.sum(last_d[3] > 10)))
        alpha = [item.get("alpha", np.nan) if isinstance(item, dict) else np.nan for item in output_dispersed]
        print(summary_PK(alpha, [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

    high_file = TEMP_DIR / "highSE.RData"
    if high_file.exists():
        output_high = load_output(high_file).get("output_highSE", load_output(high_file))
    elif len(high_ratio):
        output_high = refit_subset(high_ratio, ids, df_visits, "highSE")
    else:
        output_high = []
    if output_high:
        traces_h, _ = se_trace_last(output_high, key="beta_se_model_trace", expected_len=(8, 9))
        stacked = np.vstack([t for t in traces_h if t.size])
        if stacked.shape[0] > 39:
            print(int(np.sum(stacked[39] > 100)))
            print(summary_PK(stacked[39], [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

    alpha_penalized = read_nifti(GEEDIR / "results_penalty_poisson_sexM_exch_geePK_max25" / "alpha_GEE.nii.gz")
    alpha_original = read_nifti(GEEDIR / "results_poisson_sexM_exch_geePK_max25" / "alpha_GEE.nii.gz")
    mask = ~np.isnan(alpha_original) & (alpha_original != 0)
    print(pd.Series(alpha_original[mask]).describe())
    print(pd.Series(alpha_penalized[mask]).describe())
    plt.figure()
    plt.scatter(alpha_original[mask], alpha_penalized[mask], s=4, marker=".")
    plt.xlabel("original GEE, phi=1")
    plt.ylabel("penalized GEE, phi=1")
    lo, hi = plt.xlim(); plt.plot([lo, hi], [lo, hi], color="red")
    (GEEDIR / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(GEEDIR / "figures" / "alpha_original_vs_penalized.png", bbox_inches="tight")
    plt.close()
    _ = brain_mask, empir_prob_vis1, empir_prob_vis2, PGEEDIR, gee_penalty_run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
