#!/usr/bin/env python3
"""Summarize visit-to-visit lesion changes by subject and voxel."""

import gc
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as st
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
    for c in candidates:
        if c.exists():
            img = nib.load(str(c))
            return img.get_fdata(), img
    raise FileNotFoundError(path)


def save_nifti(data, path, like=None):
    out = Path(path)
    if not (str(out).endswith(".nii") or str(out).endswith(".nii.gz")):
        out = Path(str(out) + ".nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.asarray(data),
            like.affine if like is not None else np.eye(4),
            like.header.copy() if like is not None else None,
        ),
        str(out),
    )


def r_linear_get(arr, indices):
    return np.asarray(arr).ravel(order="F")[np.asarray(indices).astype(int).ravel() - 1]


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


def binarize(data):
    return (np.asarray(data) >= 0.5).astype(np.uint8)


def center(s):
    return s - np.nanmean(s)


def describe(x, name):
    s = pd.Series(np.asarray(x).ravel()).dropna()
    print(
        f"{name}: n={len(s)} min={s.min() if len(s) else np.nan} mean={s.mean() if len(s) else np.nan} max={s.max() if len(s) else np.nan}"
    )


TEMPDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp"
IMAGEDIR_VIS1 = "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis"
IMAGEDIR_VIS2 = "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis"


def main():
    obj = load_rdata(Path(TEMPDIR) / "lesions_p2greaterp1.RData")
    v1, v2 = np.asarray(obj["lesions_vis1"]), np.asarray(obj["lesions_vis2"])
    changes_subjects = np.column_stack(
        [
            np.sum((v1 == 0) & (v2 == 0), axis=0),
            np.sum((v1 == 0) & (v2 == 1), axis=0),
            np.sum((v1 == 1) & (v2 == 0), axis=0),
            np.sum((v1 == 1) & (v2 == 1), axis=0),
        ]
    )
    changes_subjects = np.column_stack(
        [
            changes_subjects,
            (changes_subjects[:, 2] + changes_subjects[:, 3]) / v1.shape[0],
            (changes_subjects[:, 1] + changes_subjects[:, 3]) / v1.shape[0],
        ]
    )
    changes_voxels = np.column_stack(
        [
            np.sum((v1 == 0) & (v2 == 0), axis=1),
            np.sum((v1 == 0) & (v2 == 1), axis=1),
            np.sum((v1 == 1) & (v2 == 0), axis=1),
            np.sum((v1 == 1) & (v2 == 1), axis=1),
        ]
    )
    changes_voxels = np.column_stack(
        [
            changes_voxels,
            (changes_voxels[:, 2] + changes_voxels[:, 3]) / v1.shape[1],
            (changes_voxels[:, 1] + changes_voxels[:, 3]) / v1.shape[1],
        ]
    )
    print(
        pd.DataFrame(
            changes_subjects, columns=["00", "01", "10", "11", "prb_vis1", "prb_vis2"]
        ).describe()
    )
    print(
        pd.DataFrame(
            changes_voxels, columns=["00", "01", "10", "11", "prb_vis1", "prb_vis2"]
        ).describe()
    )


if __name__ == "__main__":
    main()
