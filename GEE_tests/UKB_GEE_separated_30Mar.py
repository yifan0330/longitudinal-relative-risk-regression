#!/usr/bin/env python3
"""Python translation of UKB_GEE_separated_30Mar.Rmd analysis chunks."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from GEE_logPoisson_run import (
    GEEDIR,
    TEMPDIR,
    design_matrix,
    load_ids,
    load_lesions,
    load_visits,
)

NAMES_COVS = ["Intercept", "avg_age", "age_diff", "sexM"]
RESULT_GEE = GEEDIR / "results_poisson_sexM_exch_geePK_max25"
RESULT_PGEE = GEEDIR / "results_penalty_poisson_sexM_exch_geePK_max25"


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
            raise RuntimeError(
                f"Could not read {path}; nested RData may require conversion"
            ) from exc


def load_voxel_ids() -> np.ndarray:
    return (
        pd.read_csv(TEMPDIR / "voxel_IDs_atleast6.dat", sep=r"\s+", header=None)
        .iloc[:, 0]
        .to_numpy(int)
    )


def flat_values(image: np.ndarray, voxel_ids: np.ndarray) -> np.ndarray:
    return image.ravel(order="F")[np.asarray(voxel_ids, dtype=int).reshape(-1) - 1]


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


def sex_separation_flags(
    lesions_vis1: np.ndarray,
    lesions_vis2: np.ndarray,
    ids: np.ndarray,
    df_visits: pd.DataFrame,
) -> np.ndarray:
    flags = np.zeros(lesions_vis1.shape[0], dtype=bool)
    sex_lookup = df_visits.set_index("eid_8107")["sexM"]
    sex = pd.Series(ids.astype(str)).map(sex_lookup).to_numpy()
    for i in range(lesions_vis1.shape[0]):
        y = np.r_[lesions_vis1[i], lesions_vis2[i]]
        s = np.r_[sex, sex]
        lesion_f = np.sum((y == 1) & (s == 0))
        lesion_m = np.sum((y == 1) & (s == 1))
        flags[i] = (lesion_f == 0 and lesion_m > 0) or (lesion_f > 0 and lesion_m == 0)
        if (i + 1) % 1000 == 0:
            print(i + 1)
    return flags


def se_trace_last(
    output_all, key="beta_se_model_trace", n_cov=4, expected_len=(8, 9)
) -> np.ndarray:
    cols = []
    for item in output_all:
        if isinstance(item, dict) and key in item:
            y = np.asarray(item[key], dtype=float)
        elif hasattr(item, "__len__") and len(item) in expected_len:
            try:
                y = np.asarray(item[2][key], dtype=float)
            except Exception:
                cols.append(np.full(n_cov, np.nan))
                continue
        else:
            cols.append(np.full(n_cov, np.nan))
            continue
        ratio = y / y[0, :]
        cols.append(ratio[-1, :])
    return np.column_stack(cols)


def plot_hist_pair(values_a, values_b, title_a, title_b, out_path, xlim):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(values_a[~np.isnan(values_a)])
    axes[0].set_title(title_a)
    axes[0].set_xlim(*xlim)
    axes[1].hist(values_b[~np.isnan(values_b)])
    axes[1].set_title(title_b)
    axes[1].set_xlim(*xlim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_by_separation(
    result_dir: Path, dvrg_idx: np.ndarray, voxel_ids: np.ndarray, label: str
):
    est_map = read_nifti(result_dir / f"estimate_{NAMES_COVS[3]}_GEE")
    vals = flat_values(est_map, voxel_ids)
    print(
        label,
        "separated",
        summary_PK(vals[dvrg_idx], [0, 0.25, 0.5, 0.75, 0.99, 1]),
        int(np.sum(dvrg_idx)),
    )
    print(
        label,
        "not separated",
        summary_PK(vals[~dvrg_idx], [0, 0.25, 0.5, 0.75, 0.99, 1]),
        int(np.sum(~dvrg_idx)),
    )
    plot_hist_pair(
        vals[dvrg_idx],
        vals[~dvrg_idx],
        f"{label}, sex-separated",
        f"{label}, not sex-separated",
        GEEDIR / "figures" / f"{label}_sexM_est_hist.png",
        (-30, 30) if label == "GEE" else (-5, 5),
    )


def main() -> int:
    voxel_ids = load_voxel_ids()
    lesions_vis1, lesions_vis2 = load_lesions()
    ids = load_ids()
    df_visits = load_visits()
    dvrg_idx = sex_separation_flags(lesions_vis1, lesions_vis2, ids, df_visits)
    print(pd.Series(dvrg_idx).value_counts())

    summarize_by_separation(RESULT_GEE, dvrg_idx, voxel_ids, "GEE")
    output_all = load_output(RESULT_GEE / "results_exch_GEE.Rdata").get(
        "output_all", []
    )
    last = se_trace_last(output_all, expected_len=(6, 8))
    print(summary_PK(last[3], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    print(summary_PK(last[3, dvrg_idx], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    print(summary_PK(last[3, ~dvrg_idx], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    plot_hist_pair(
        last[3, dvrg_idx],
        last[3, ~dvrg_idx],
        "SE / GEE, sex-separated",
        "SE / GEE, not sex-separated",
        GEEDIR / "figures" / "GEE_se_ratio_hist.png",
        (0, 5),
    )

    summarize_by_separation(RESULT_PGEE, dvrg_idx, voxel_ids, "pGEE")
    output_all_p = load_output(RESULT_PGEE / "results_exch_GEE.Rdata").get(
        "output_all", []
    )
    last_p = se_trace_last(output_all_p, expected_len=(8, 9))
    print(summary_PK(last_p[3], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    print(summary_PK(last_p[3, dvrg_idx], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    print(summary_PK(last_p[3, ~dvrg_idx], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]))
    plot_hist_pair(
        last_p[3, dvrg_idx],
        last_p[3, ~dvrg_idx],
        "SE / pGEE, sex-separated",
        "SE / pGEE, not sex-separated",
        GEEDIR / "figures" / "pGEE_se_ratio_hist.png",
        (0, 5),
    )

    gee_est_map = read_nifti(RESULT_GEE / f"estimate_{NAMES_COVS[3]}_GEE")
    pgee_est_map = read_nifti(RESULT_PGEE / f"estimate_{NAMES_COVS[3]}_GEE")
    pgee_alpha_map = read_nifti(RESULT_PGEE / "alpha_GEE")
    idx_nan = np.isnan(gee_est_map)
    print(int(np.sum(idx_nan)))
    print(pd.Series(pgee_est_map[idx_nan]).describe())
    print(pd.Series(pgee_alpha_map[idx_nan]).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
