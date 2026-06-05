#!/usr/bin/env python3
"""Prepare Basel covariates, empirical lesion maps, and lesion matrices."""

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


IMAGEDIR = "/well/nichols/users/kfh142/data/Basel"
WORKDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/Basel_data/"
VISIT_FILES = {
    "bs": "bs/pd_segm_t2_to_tal.nii.gz",
    "y1": "y1/pd_segm_t2_to_tal.nii.gz",
    "y2": "y2/pd_segm_t2_to_tal.nii.gz",
}


def subject_image(pt, visit):
    data, _ = load_nifti(Path(IMAGEDIR) / "data" / str(pt) / VISIT_FILES[visit])
    return binarize(data)


def build_complete_covariates():
    df = pd.read_csv(Path(IMAGEDIR) / "AllSubjs_AgeSex.csv")
    df = df[np.floor(df["PT"] / 1000) == 2].copy()
    for visit in VISIT_FILES:
        df[visit] = 0
    for idx, row in df.iterrows():
        base = Path(IMAGEDIR) / "data" / str(row.iloc[0])
        if base.is_dir():
            for visit, subpath in VISIT_FILES.items():
                df.loc[idx, visit] = int((base / subpath).exists())
    complete = df.loc[df[["bs", "y1", "y2"]].sum(axis=1) == 3, df.columns[:3]]
    age_sex_path = Path(WORKDIR) / "AgeSex_complete.dat"
    if age_sex_path.exists():
        complete = read_table(age_sex_path, header=True)
    scores = pd.read_csv(Path(IMAGEDIR) / "20100507 AGE-SEX-EDSS-RLPNO-DISCR.csv").iloc[
        :, 13:16
    ]
    scores = scores.rename(columns={scores.columns[0]: "PT"})
    complete = complete.merge(scores, on="PT", how="left").sort_values("PT")
    complete = complete[
        complete["CPEVENT.2"].isin(["BASELINE", "MONTH12", "MONTH24"])
    ].copy()
    complete["vis"] = complete["CPEVENT.2"].map(
        {"BASELINE": 1, "MONTH12": 2, "MONTH24": 3}
    )
    complete = complete.sort_values(["PT", "vis"])
    for visit in (2, 3):
        mask = (complete["vis"] == visit) & (complete["DISCRS"] == "UCH")
        complete.loc[mask, "DISCRS"] = complete["DISCRS"].shift(1).loc[mask]
    write_table(
        complete.iloc[:, [0, 5, 1, 2, 4]], Path(WORKDIR) / "AgeSexMStype_complete.dat"
    )
    return complete


def empirical_maps(df):
    subjects = pd.unique(df["PT"])
    outputs = {}
    like = None
    for visit in VISIT_FILES:
        acc = None
        for i, pt in enumerate(subjects, start=1):
            data, like = load_nifti(
                Path(IMAGEDIR) / "data" / str(pt) / VISIT_FILES[visit]
            )
            acc = binarize(data).astype(float) if acc is None else acc + binarize(data)
            print(i)
        outputs[visit] = acc / len(subjects)
        save_nifti(outputs[visit], Path(WORKDIR) / f"empir_{visit}", like)
        vector_summary(outputs[visit][outputs[visit] != 0], f"empir_{visit}")
    return outputs


def all_subjects(datafile, voxel_ids, visit):
    subjects = pd.unique(datafile["PT"])
    out = np.zeros((len(voxel_ids), len(subjects)), dtype=np.uint8)
    for k, pt in enumerate(subjects):
        out[:, k] = r_linear_get(subject_image(pt, visit), voxel_ids)
        print(k + 1)
    return out


def main():
    df = build_complete_covariates()
    empir = empirical_maps(df)
    voxel_idx = (
        np.flatnonzero(
            (empir["bs"] > 0.025) | (empir["y1"] > 0.025) | (empir["y2"] > 0.025)
        )
        + 1
    )
    print(voxel_idx.shape)
    write_table(
        pd.DataFrame(voxel_idx), Path(WORKDIR) / "voxel_IDs_5subj.dat", header=False
    )
    lesions = {}
    for visit in ("bs", "y1", "y2"):
        key = f"lesions_{visit}"
        lesions[key] = all_subjects(df, voxel_idx, visit)
        save_rdata(Path(WORKDIR) / f"{key}_5subj.RData", **{key: lesions[key]})
    bs, y1, y2 = lesions["lesions_bs"], lesions["lesions_y1"], lesions["lesions_y2"]
    changes = np.column_stack(
        [
            np.sum((bs == a) & (y1 == b) & (y2 == c), axis=1)
            for a, b, c in [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
            ]
        ]
    )
    print(
        pd.DataFrame(
            changes, columns=["000", "001", "011", "111", "010", "100", "101", "110"]
        ).describe()
    )


if __name__ == "__main__":
    main()
