#!/usr/bin/env python3
"""Prepare longitudinal lesion matrices and cleaned visit covariates."""

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


TEMPDIR = str(PROJECT_ROOT / 'prelim/temp')
IMAGEDIR_VIS1 = str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis')
IMAGEDIR_VIS2 = str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis')
BRAIN_MASK = (
    str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii')
)


def make_empirical(df, imagedir, outname):
    acc = None
    like = None
    for i, eid in enumerate(df["eid_8107"], start=1):
        data, like = load_nifti(Path(imagedir) / f"{eid}_T2_lesions_MNI_bin.nii.gz")
        acc = binarize(data).astype(float) if acc is None else acc + binarize(data)
        if i % 100 == 0:
            print(i)
    empir = acc / len(df)
    save_nifti(empir, Path(imagedir) / outname, like)
    return empir, like


def all_subj(datafile, imagedir, voxel_ids):
    out = np.zeros((len(voxel_ids), len(datafile)), dtype=np.uint8)
    for k, eid in enumerate(datafile.iloc[:, 0], start=0):
        img, _ = load_nifti(Path(imagedir) / f"{eid}_T2_lesions_MNI_bin.nii.gz")
        out[:, k] = r_linear_get(img, voxel_ids)
        if (k + 1) % 100 == 0:
            print(k + 1)
    return out


def build_cleaned_visits(ids):
    vis1 = read_table(
        str(PROJECT_ROOT / 'funpack/Vis0.tsv'), sep="\t"
    ).rename(
        columns=lambda c: (
            "eid_34077"
            if c
            == read_table(
                str(PROJECT_ROOT / 'funpack/Vis0.tsv'),
                sep="\t",
            ).columns[0]
            else c
        )
    )
    vis2 = read_table(
        str(PROJECT_ROOT / 'funpack/Vis2.tsv'), sep="\t"
    )
    vis2 = vis2.rename(columns={vis2.columns[0]: "eid_34077"})
    vis3 = read_table(
        str(PROJECT_ROOT / 'funpack/Vis3.tsv'), sep="\t"
    )
    vis3 = vis3.rename(columns={vis3.columns[0]: "eid_34077"})
    bridge = read_table(
        "/well/nichols/projects/UKB/SMS/bridge_8107_34077.tsv", header=True
    )
    bridge.columns = ["eid_8107", "eid_34077"]
    ids_df = pd.DataFrame({"eid_8107": ids})
    vis1 = ids_df.merge(vis1.merge(bridge, on="eid_34077"), on="eid_8107", how="left")
    vis2 = ids_df.merge(vis2.merge(bridge, on="eid_34077"), on="eid_8107", how="left")
    vis3 = ids_df.merge(vis3.merge(bridge, on="eid_34077"), on="eid_8107", how="left")
    df = vis1.merge(vis2, on=["eid_8107", "eid_34077"]).merge(
        vis3, on=["eid_8107", "eid_34077"]
    )
    yob, mob = df["X34.0.0"], df["X52.0.0"]
    birth = yob + (mob - 0.5) / 12.0
    for src, dst in [("X53.2.0", "age_vis2"), ("X53.3.0", "age_vis3")]:
        d = pd.to_datetime(df[src])
        df[dst] = d.dt.year + (d.dt.dayofyear - 1) / 365.2425 - birth
    df["avg_age"] = (df["age_vis2"] + df["age_vis3"]) / 2
    conditions = [
        1081,
        1082,
        1083,
        1086,
        1240,
        1243,
        1244,
        1245,
        1246,
        1247,
        1258,
        1259,
        1261,
        1262,
        1263,
        1264,
        1266,
        1408,
        1409,
        1410,
        1434,
        1491,
        1583,
        1626,
    ]
    cols = list(df.columns[7:41]) + list(df.columns[51:85])
    condition_idx = df[cols].isin(conditions).any(axis=1)
    df = df.loc[~condition_idx].copy()
    df = df[df["X21000.0.0"].isin([1001, 1002, 1003, 1])]
    write_table(df, Path(TEMPDIR) / "df_visits_cleaned_Apr2021.dat")
    return df


def main():
    brain, _ = load_nifti(BRAIN_MASK)
    subjects = (
        read_table(
            str(PROJECT_ROOT.parent / 'bianca_1vis_2vis_overlap_filenames_Apr2021.txt'),
            header=False,
        )
        .iloc[:, 0]
        .astype(str)
    )
    ids = subjects.str.replace(r"_.*", "", regex=True)
    df_path = Path(TEMPDIR) / "df_visits_cleaned_Apr2021.dat"
    df = read_table(df_path) if df_path.exists() else build_cleaned_visits(ids)
    empir1, _ = make_empirical(df, IMAGEDIR_VIS1, "Apr2021_cleaned_empir_prob_mask")
    empir2, _ = make_empirical(df, IMAGEDIR_VIS2, "Apr2021_cleaned_empir_prob_mask")
    voxel_idx = np.flatnonzero(((empir1 + empir2) >= 6 / len(df)) & (brain == 1)) + 1
    write_table(
        pd.DataFrame(voxel_idx),
        Path(TEMPDIR) / "voxel_IDs_atleast6_cleaned_Apr2021.dat",
        header=False,
    )
    lesions_vis1 = all_subj(df, IMAGEDIR_VIS1, voxel_idx)
    lesions_vis2 = all_subj(df, IMAGEDIR_VIS2, voxel_idx)
    save_rdata(
        Path(TEMPDIR) / "lesions_atleast6_cleaned_Apr2021.RData",
        lesions_vis1=lesions_vis1,
        lesions_vis2=lesions_vis2,
    )
    print(
        pd.Series(
            empir2.ravel(order="F")[voxel_idx - 1]
            - empir1.ravel(order="F")[voxel_idx - 1]
        ).describe()
    )


if __name__ == "__main__":
    main()
