#!/usr/bin/env python3
"""Create CVR lesion matrices and run example log-Poisson GEE fits."""

import gc
import math
import pickle
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf


def read_table(path, header=True, sep=None):
    return pd.read_csv(path, sep=sep if sep is not None else r"\s+", header=0 if header else None, engine="python")


def write_table(df, path, header=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(path, sep=" ", index=False, header=header)


def load_nifti(path):
    p = Path(path)
    candidates = [p] if p.exists() else [Path(str(p) + ext) for ext in (".nii.gz", ".nii")]
    for c in candidates:
        if c.exists():
            img = nib.load(str(c)); return img.get_fdata(), img
    raise FileNotFoundError(path)


def save_nifti(data, path, like=None):
    out = Path(path)
    if not (str(out).endswith(".nii") or str(out).endswith(".nii.gz")):
        out = Path(str(out) + ".nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), like.affine if like is not None else np.eye(4), like.header.copy() if like is not None else None), str(out))


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
    s = pd.Series(np.asarray(x).ravel()).dropna(); print(f"{name}: n={len(s)} min={s.min() if len(s) else np.nan} mean={s.mean() if len(s) else np.nan} max={s.max() if len(s) else np.nan}")



TEMPDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp"
IMAGEDIR_VIS1 = "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis"
IMAGEDIR_VIS2 = "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis"
BRAIN_MASK = "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii"

def pgee_log_poisson(y, x, n_subj, n_visits, max_iter=25):
    # The custom R gee_penalty_run/gee_dispersion_run code has no exact statsmodels equivalent; this uses exchangeable log-Poisson GEE with robust covariance.
    groups = np.repeat(np.arange(int(n_subj)), int(n_visits))
    model = sm.GEE(y, x, groups=groups, family=sm.families.Poisson(sm.families.links.Log()), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit(maxiter=max_iter)
    return {"beta": res.params, "beta_se_sandwich": res.bse, "zscore": res.params / res.bse, "alpha": getattr(res.cov_struct, "dep_params", None)}

def prepare_complete_df(cvr=True):
    obj = load_rdata(Path(TEMPDIR) / "CVR_9June2021.Rdata")
    complete_df = obj.get("complete_df")
    if not isinstance(complete_df, pd.DataFrame):
        complete_df = pd.DataFrame(complete_df)
    if cvr:
        complete_df = complete_df.iloc[:, 1:] if complete_df.columns[0] not in ["eid_8107"] else complete_df
        complete_df = complete_df.rename(columns={complete_df.columns[1]: "sexM"})
        complete_df["headsize"] = (complete_df["X25000.2.0"] + complete_df["X25000.3.0"]) / 2
        keep = ["eid_8107","age_vis2","age_vis3","sexM","headsize","CVR_vis2","CVR_vis3"]
        out = complete_df[keep].copy()
        out["age_diff"] = out["age_vis3"] - out["age_vis2"]
        out["CVR_diff"] = out["CVR_vis3"] - out["CVR_vis2"]
        for col in ["age_vis2","headsize","CVR_vis2"]: out[col] = center(out[col])
        return out
    out = complete_df.dropna(subset=["systolic_vis1","systolic_vis2"]).copy()
    out = out.rename(columns={out.columns[2]: "sexM"})
    out["headsize"] = (out["X25000.2.0"] + out["X25000.3.0"]) / 2
    out = out[["eid_8107","age_vis2","age_vis3","sexM","headsize","systolic_vis1","systolic_vis2"]]
    out["age_diff"] = out["age_vis3"] - out["age_vis2"]
    out["systolic_diff"] = out["systolic_vis2"] - out["systolic_vis1"]
    for col in ["age_vis2","headsize","systolic_vis1"]: out[col] = center(out[col])
    return out

def fit_voxel(lesions_vis1, lesions_vis2, complete_df, voxel_idx, cvr=True):
    n_subj = lesions_vis1.shape[-1]
    y1 = np.asarray(lesions_vis1[voxel_idx]).ravel(); y2 = np.asarray(lesions_vis2[voxel_idx]).ravel()
    panel = pd.DataFrame({"y": np.r_[y1,y2], "vis": np.r_[np.ones(n_subj),np.full(n_subj,2)], "eid_8107": np.r_[complete_df["eid_8107"], complete_df["eid_8107"]]})
    panel = panel.merge(complete_df, on="eid_8107", how="left").sort_values(["eid_8107","vis"])
    panel["age_diff"] *= np.tile([0,1], n_subj)
    if cvr:
        panel["CVR_diff"] *= np.tile([0,1], n_subj)
        x = np.column_stack([np.ones(len(panel)), panel["age_vis2"], panel["age_diff"], panel["CVR_vis2"], panel["CVR_diff"], panel["sexM"], panel["headsize"], panel["age_vis2"]*panel["age_diff"], panel["age_vis2"]*panel["sexM"]])
    else:
        panel["systolic_diff"] *= np.tile([0,1], n_subj)
        x = np.column_stack([np.ones(len(panel)), panel["age_vis2"], panel["age_diff"], panel["systolic_vis1"], panel["systolic_diff"], panel["sexM"], panel["headsize"], panel["age_vis2"]*panel["age_diff"], panel["age_vis2"]*panel["sexM"]])
    return pgee_log_poisson(panel["y"].to_numpy(float), x.astype(float), len(panel)//2, 2)

def make_empirical(df, imagedir, outname):
    acc=None; like=None
    for i,eid in enumerate(df["eid_8107"], start=1):
        img, like = load_nifti(Path(imagedir)/f"{eid}_T2_lesions_MNI_bin.nii.gz")
        acc = binarize(img).astype(float) if acc is None else acc + binarize(img)
        if i % 100 == 0: print(i)
    empir = acc / len(df); save_nifti(empir, Path(imagedir)/outname, like); return empir

def all_subj(datafile, imagedir, voxel_ids):
    out=np.zeros((len(voxel_ids),len(datafile)), dtype=np.uint8)
    for k,eid in enumerate(datafile["eid_8107"]):
        img,_=load_nifti(Path(imagedir)/f"{eid}_T2_lesions_MNI_bin.nii.gz")
        out[:,k]=r_linear_get(img, voxel_ids)
        if (k+1)%100==0: print(k+1)
    return out

def main():
    complete_df = prepare_complete_df(cvr=True)
    brain,_=load_nifti(BRAIN_MASK)
    empir1=make_empirical(complete_df, IMAGEDIR_VIS1, "CVR_empir_prob_mask")
    empir2=make_empirical(complete_df, IMAGEDIR_VIS2, "CVR_empir_prob_mask")
    voxel_idx=np.flatnonzero(((empir1+empir2)>=6/len(complete_df)) & (brain==1))+1
    write_table(pd.DataFrame(voxel_idx), Path(TEMPDIR)/"voxel_IDs_CVR.dat", header=False)
    lesions_vis1=all_subj(complete_df, IMAGEDIR_VIS1, voxel_idx); lesions_vis2=all_subj(complete_df, IMAGEDIR_VIS2, voxel_idx)
    save_rdata(Path(TEMPDIR)/"lesions_atleast6_CVR.RData", lesions_vis1=lesions_vis1, lesions_vis2=lesions_vis2)
    model=fit_voxel(lesions_vis1, lesions_vis2, complete_df, 6999, cvr=True)
    save_rdata(Path(TEMPDIR)/"CVR_temp.Rdata", gee_model=model, pgee_model=model)
if __name__ == "__main__": main()
