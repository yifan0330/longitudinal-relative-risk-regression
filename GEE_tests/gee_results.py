#!/usr/bin/env python3
from __future__ import annotations

import math
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

TEMPDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp")
GEEDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/GEE_tests")
BRAIN_MASK_FILE = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii")
RESULT_IN_DIR = GEEDIR / "temp_penalty_poisson_sexM_exch_geePK_max25"
RESULT_OUT_DIR = GEEDIR / "results_penalty_poisson_sexM_exch_geePK_max25"


def read_table(path: Path, header="infer") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=header, engine="python")


def load_rdata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError("Reading .RData requires pyreadr.") from exc
    return dict(pyreadr.read_r(str(path)))


def load_output(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def summarize(values) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    print(pd.Series(arr).describe())


def subset_indices(n_rows: int, subset_size: int, j_1based: int) -> np.ndarray:
    return np.arange(subset_size * (j_1based - 1), min(subset_size * j_1based, n_rows), dtype=int)


def put_values(mask_img: nib.Nifti1Image, voxel_ids_1based: np.ndarray, values: np.ndarray) -> nib.Nifti1Image:
    data = np.asarray(mask_img.get_fdata()).copy()
    data[data != 0] = 0
    flat = data.ravel(order="F")
    flat[voxel_ids_1based.astype(int) - 1] = np.asarray(values, dtype=float).reshape(-1)
    return nib.Nifti1Image(flat.reshape(data.shape, order="F"), mask_img.affine, mask_img.header)


def write_nifti(img: nib.Nifti1Image, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    filename = prefix if str(prefix).endswith((".nii", ".nii.gz")) else Path(str(prefix) + ".nii.gz")
    nib.save(img, filename)


def main() -> int:
    brain_mask = nib.load(str(BRAIN_MASK_FILE))
    voxel_ids = read_table(TEMPDIR / "voxel_IDs_atleast6.dat", header=None).iloc[:, 0].to_numpy(dtype=int)
    P = 4
    estimates = np.zeros((len(voxel_ids), P), dtype=float)
    stderror = np.zeros_like(estimates)
    alpha = np.zeros((len(voxel_ids), 1), dtype=float)
    iterations = np.zeros((len(voxel_ids), 1), dtype=float)
    output_all = [None] * len(voxel_ids)

    subset_size = 1000
    lesions_vis1 = np.asarray(load_rdata(TEMPDIR / "lesions_atleast6.RData")["lesions_vis1"])
    n_subsets = math.ceil(lesions_vis1.shape[0] / subset_size)
    print(n_subsets)

    for j in range(1, n_subsets + 1):
        print(j)
        subset_idx = subset_indices(lesions_vis1.shape[0], subset_size, j)
        output = load_output(RESULT_IN_DIR / f"GEE_subset_{j}.RData")
        for k, item in enumerate(output):
            row = subset_idx[k]
            if isinstance(item, dict):
                estimates[row, :] = np.asarray(item.get("beta", np.repeat(np.nan, P)), dtype=float)[:P]
                stderror[row, :] = np.asarray(item.get("beta_se_sandwich", np.repeat(np.nan, P)), dtype=float)[:P]
                alpha[row, 0] = float(item.get("alpha", np.nan))
                iterations[row, 0] = float(item.get("iterations", np.nan))
            else:
                estimates[row, :] = np.nan
                stderror[row, :] = np.nan
                alpha[row, 0] = np.nan
                iterations[row, 0] = np.nan
            output_all[row] = item

    zscores = estimates / stderror
    names_covs = ["Intercept", "avg_age", "age_diff", "sexM"]
    for i, name in enumerate(names_covs):
        img = put_values(brain_mask, voxel_ids, estimates[:, i])
        summarize(np.asarray(img.get_fdata()).ravel(order="F")[voxel_ids - 1])
        write_nifti(img, RESULT_OUT_DIR / f"estimate_{name}_GEE")

        img = put_values(brain_mask, voxel_ids, stderror[:, i])
        summarize(np.asarray(img.get_fdata()).ravel(order="F")[voxel_ids - 1])
        write_nifti(img, RESULT_OUT_DIR / f"se_{name}_GEE")

        img = put_values(brain_mask, voxel_ids, zscores[:, i])
        summarize(np.asarray(img.get_fdata()).ravel(order="F")[voxel_ids - 1])
        write_nifti(img, RESULT_OUT_DIR / f"zscore_{name}_GEE")
        print("-----")

    img = put_values(brain_mask, voxel_ids, alpha[:, 0])
    summarize(np.asarray(img.get_fdata()).ravel(order="F")[voxel_ids - 1])
    write_nifti(img, RESULT_OUT_DIR / "alpha_GEE")

    img = put_values(brain_mask, voxel_ids, iterations[:, 0])
    summarize(np.asarray(img.get_fdata()).ravel(order="F")[voxel_ids - 1])
    write_nifti(img, RESULT_OUT_DIR / "iterations_GEE")

    RESULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULT_OUT_DIR / "results_exch_GEE.Rdata").open("wb") as f:
        pickle.dump({
            "estimates": estimates,
            "stderror": stderror,
            "zscores": zscores,
            "alpha": alpha,
            "iterations": iterations,
            "output_all": output_all,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
