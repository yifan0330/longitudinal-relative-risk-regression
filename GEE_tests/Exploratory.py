#!/usr/bin/env python3
"""Python translation of Exploratory.Rmd analysis chunks."""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from data_io import load_array_data

TEMPDIR = PROJECT_ROOT / 'prelim/temp'
GEEDIR = PROJECT_ROOT / 'GEE_tests'
PGEEDIR = PROJECT_ROOT / 'PGEE_Mondol'
IMAGEDIR_VIS1 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis'
IMAGEDIR_VIS2 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis'
BRAIN_MASK = Path(
    str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii')
)
NAMES_COVS = ["Intercept", "avg_age", "age_diff"]


def load_rdata(path: Path) -> dict:
    return load_array_data(path)


def read_nifti(path: Path | str) -> np.ndarray:
    p = Path(path)
    candidates = [p]
    if not p.suffix:
        candidates += [Path(str(p) + ".nii.gz"), Path(str(p) + ".nii")]
    for candidate in candidates:
        if candidate.exists():
            return np.asarray(nib.load(str(candidate)).get_fdata())
    raise FileNotFoundError(p)


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


def r_summary(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return pd.Series(arr).describe(percentiles=[0.25, 0.5, 0.75])


def load_voxel_ids() -> np.ndarray:
    return (
        pd.read_csv(TEMPDIR / "voxel_IDs_atleast6.dat", sep=r"\s+", header=None)
        .iloc[:, 0]
        .to_numpy(int)
    )


def cross_tabs_subjects(
    lesions_vis1: np.ndarray, lesions_vis2: np.ndarray
) -> pd.DataFrame:
    out = np.column_stack(
        [
            np.sum((lesions_vis1 == 0) & (lesions_vis2 == 0), axis=0),
            np.sum((lesions_vis1 == 0) & (lesions_vis2 == 1), axis=0),
            np.sum((lesions_vis1 == 1) & (lesions_vis2 == 0), axis=0),
            np.sum((lesions_vis1 == 1) & (lesions_vis2 == 1), axis=0),
        ]
    ).astype(float)
    out = np.column_stack(
        [
            out,
            (out[:, 2] + out[:, 3]) / lesions_vis1.shape[0],
            (out[:, 1] + out[:, 3]) / lesions_vis1.shape[0],
        ]
    )
    return pd.DataFrame(out, columns=["00", "01", "10", "11", "prb_vis1", "prb_vis2"])


def cross_tabs_voxels(
    lesions_vis1: np.ndarray, lesions_vis2: np.ndarray
) -> pd.DataFrame:
    out = np.column_stack(
        [
            np.sum((lesions_vis1 == 0) & (lesions_vis2 == 0), axis=1),
            np.sum((lesions_vis1 == 0) & (lesions_vis2 == 1), axis=1),
            np.sum((lesions_vis1 == 1) & (lesions_vis2 == 0), axis=1),
            np.sum((lesions_vis1 == 1) & (lesions_vis2 == 1), axis=1),
        ]
    ).astype(float)
    out = np.column_stack(
        [
            out,
            (out[:, 2] + out[:, 3]) / lesions_vis1.shape[1],
            (out[:, 1] + out[:, 3]) / lesions_vis1.shape[1],
        ]
    )
    return pd.DataFrame(out, columns=["00", "01", "10", "11", "prb_vis1", "prb_vis2"])


def plot_compare(x, y, title, out_path, xlim=None, ylim=None):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, marker=".", s=4)
    plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    lo, hi = plt.xlim()
    plt.plot([lo, hi], [lo, hi])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def summarize_maps(
    base_dir: Path, result_dir: str, voxel_ids: np.ndarray, cov_names=NAMES_COVS
):
    for cov in cov_names:
        print(f"## {base_dir / result_dir}: {cov}")
        for prefix in ["estimate", "se", "zscore"]:
            img = read_nifti(base_dir / result_dir / f"{prefix}_{cov}_GEE")
            vals = flat_values(img, voxel_ids)
            print(prefix, summary_PK(vals, [0, 0.25, 0.5, 0.75, 1]))


def main() -> int:
    voxel_ids = load_voxel_ids()
    brain_mask = read_nifti(BRAIN_MASK)
    empir_prob_vis1 = read_nifti(IMAGEDIR_VIS1 / "empir_prob_mask.nii.gz")
    empir_prob_vis2 = read_nifti(IMAGEDIR_VIS2 / "empir_prob_mask.nii.gz")
    lesions = load_array_data(
        TEMPDIR / "lesions_atleast6.npz", required=("lesions_vis1", "lesions_vis2")
    )
    lesions_vis1 = np.asarray(lesions["lesions_vis1"], dtype=float)
    lesions_vis2 = np.asarray(lesions["lesions_vis2"], dtype=float)

    changes_subjects = cross_tabs_subjects(lesions_vis1, lesions_vis2)
    print(changes_subjects.describe())
    print(r_summary(changes_subjects["prb_vis2"] - changes_subjects["prb_vis1"]))
    print(r_summary(changes_subjects["prb_vis2"] / changes_subjects["prb_vis1"]))

    changes_voxels = cross_tabs_voxels(lesions_vis1, lesions_vis2)
    print(changes_voxels.describe())
    print(r_summary(changes_voxels["prb_vis2"] - changes_voxels["prb_vis1"]))
    print(r_summary(changes_voxels["prb_vis2"] / (changes_voxels["prb_vis1"] + 1e-6)))

    summarize_maps(GEEDIR, "results_atleast6", voxel_ids)
    time_se = read_nifti(GEEDIR / "results_atleast6" / "se_age_diff_GEE")
    temp_idx = np.where(time_se.ravel(order="F") > 1.6)[0] + 1
    z_map = read_nifti(GEEDIR / "results_atleast6" / "zscore_age_diff_GEE")
    voxel_idx = np.where(np.isnan(flat_values(z_map, voxel_ids)))[0] + 1
    (GEEDIR / "temp_atleast6").mkdir(parents=True, exist_ok=True)
    np.savetxt(GEEDIR / "temp_atleast6" / "voxel_IDs_NAs.dat", voxel_idx, fmt="%d")

    estimate_intercept = read_nifti(
        GEEDIR / "results_atleast6" / "estimate_Intercept_GEE"
    )
    estimate_age = read_nifti(GEEDIR / "results_atleast6" / "estimate_avg_age_GEE")
    estimate_time = read_nifti(GEEDIR / "results_atleast6" / "estimate_age_diff_GEE")
    xbeta60 = estimate_intercept + estimate_age * 60
    xbeta70 = estimate_intercept + estimate_age * 70
    rr_age = (np.exp(xbeta70) / (1 + np.exp(xbeta70))) / (
        np.exp(xbeta60) / (1 + np.exp(xbeta60))
    )
    print(summary_PK(flat_values(rr_age, voxel_ids), [0, 0.25, 0.5, 0.75, 1]))
    xbeta_vis1 = estimate_intercept + estimate_age * 60 + estimate_time * -5
    xbeta_vis2 = estimate_intercept + estimate_age * 60 + estimate_time * 5
    rr_stable = (
        np.exp(estimate_time * 10) * (1 + np.exp(xbeta_vis1)) / (1 + np.exp(xbeta_vis2))
    )
    print(summary_PK(flat_values(rr_stable, voxel_ids), [0, 0.25, 0.5, 0.75, 1]))

    summarize_maps(PGEEDIR, "results_atleast6", voxel_ids)
    for cov, title in zip(NAMES_COVS, ["Intercept", "Age", "Time"]):
        z_gee = read_nifti(GEEDIR / "results_atleast6" / f"zscore_{cov}_GEE")
        z_pgee = read_nifti(PGEEDIR / "results_atleast6" / f"zscore_{cov}_GEE")
        xv = flat_values(z_gee, voxel_ids)
        yv = flat_values(z_pgee, voxel_ids)
        ok = ~np.isnan(xv) & ~np.isnan(yv)
        plot_compare(
            xv[ok],
            yv[ok],
            title,
            GEEDIR / "figures" / f"exploratory_{cov}_gee_vs_pgee.png",
            (-15, 10),
            (-15, 10),
        )
    _ = brain_mask, empir_prob_vis1, empir_prob_vis2, temp_idx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
