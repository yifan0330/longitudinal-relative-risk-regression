#!/usr/bin/env python3
"""Map voxelwise GEE subset outputs back to NIfTI images."""

import gc
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def read_table(path, header=True, sep=None):
    return pd.read_csv(
        path,
        sep=sep if sep is not None else r"\s+",
        header=0 if header else None,
        engine="python",
    )


def write_table(df, path, header=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(path, sep=" ", index=False, header=header)


def load_nifti(path):
    p = Path(path)
    candidates = (
        [p] if p.exists() else [Path(str(p) + ext) for ext in (".nii.gz", ".nii")]
    )
    for candidate in candidates:
        if candidate.exists():
            img = nib.load(str(candidate))
            return img.get_fdata(), img
    raise FileNotFoundError(path)


def save_nifti(data, path, like=None):
    out = Path(path)
    if not (str(out).endswith(".nii") or str(out).endswith(".nii.gz")):
        out = Path(str(out) + ".nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    affine = like.affine if like is not None else np.eye(4)
    header = like.header.copy() if like is not None else None
    nib.save(nib.Nifti1Image(np.asarray(data), affine, header), str(out))


def r_linear_get(arr, indices):
    return np.asarray(arr).ravel(order="F")[np.asarray(indices).astype(int).ravel() - 1]


def r_linear_set(arr, indices, values):
    out = np.asarray(arr).copy()
    out.ravel(order="F")[np.asarray(indices).astype(int).ravel() - 1] = values
    return out


def load_rdata(path):
    path = str(path)
    try:
        import pyreadr

        res = pyreadr.read_r(path)
        return dict(res.items())
    except Exception:
        pass
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_rdata(path, **objects):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(objects, fh, protocol=pickle.HIGHEST_PROTOCOL)


def vector_summary(x, name="x"):
    s = pd.Series(np.asarray(x).ravel()).dropna()
    print(
        f"{name}: empty"
        if s.empty
        else f"{name}: min={s.min()} median={s.median()} mean={s.mean()} max={s.max()} n={len(s)}"
    )


def binarize(data, threshold=0.5):
    return (np.asarray(data) >= threshold).astype(np.uint8)


WORKDIR = str(PROJECT_ROOT / 'Basel_data')


def _extract(item, key, width):
    if isinstance(item, dict) and "error" not in item:
        arr = np.asarray(item.get(key, []), dtype=float).ravel()
        out = np.full(width, np.nan)
        out[: min(width, arr.size)] = arr[:width]
        return out
    return np.full(width, np.nan)


def map_results(
    model_dir="temp_age", result_dir="results_age", names_covs=("Intercept", "age")
):
    empir_bs, like = load_nifti(Path(WORKDIR) / "empir_bs")
    voxel_ids = (
        read_table(Path(WORKDIR) / "voxel_IDs_5subj.dat", header=False)
        .to_numpy()
        .ravel()
        .astype(int)
    )
    p = len(names_covs)
    estimates = np.zeros((len(voxel_ids), p))
    stderror = np.zeros_like(estimates)
    for j in range(1, int(math.ceil(len(voxel_ids) / 1000)) + 1):
        start, stop = (j - 1) * 1000, min(j * 1000, len(voxel_ids))
        output = load_rdata(
            Path(WORKDIR) / "GEE" / model_dir / f"GEE_subset_{j}.RData"
        )["output"]
        estimates[start:stop] = np.vstack([_extract(x, "beta", p) for x in output])
        stderror[start:stop] = np.vstack(
            [_extract(x, "beta_se_sandwich", p) for x in output]
        )
    zscores = estimates / stderror
    template = np.zeros_like(empir_bs)
    for i, name in enumerate(names_covs):
        save_nifti(
            r_linear_set(template, voxel_ids, estimates[:, i]),
            Path(WORKDIR) / "GEE" / result_dir / f"estimate_{name}_GEE",
            like,
        )
        save_nifti(
            r_linear_set(template, voxel_ids, stderror[:, i]),
            Path(WORKDIR) / "GEE" / result_dir / f"se_{name}_GEE",
            like,
        )
        save_nifti(
            r_linear_set(template, voxel_ids, zscores[:, i]),
            Path(WORKDIR) / "GEE" / result_dir / f"zscore_{name}_GEE",
            like,
        )
    vector_summary(stderror, "stderror")
    vector_summary(estimates, "estimates")
    vector_summary(zscores, "zscores")


if __name__ == "__main__":
    map_results()
