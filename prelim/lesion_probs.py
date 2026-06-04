#!/usr/bin/env python3
"""Voxelwise random-intercept logistic approximation for lesion probabilities."""

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
N_CORES = 8

def fit_glmer(args):
    i, lesions_vis1, lesions_vis2, ids, df_visits, voxel_ids, empir1, empir2 = args
    try:
        n = lesions_vis1.shape[1]
        panel = pd.DataFrame({"y": np.r_[lesions_vis1[i], lesions_vis2[i]], "vis": np.r_[np.ones(n), np.full(n,2)], "eid_8107": np.r_[ids, ids]})
        panel = panel.merge(df_visits, on="eid_8107", how="left")
        panel["age_diff_vis2"] = (panel["age_vis2"] - panel["avg_age"]) * np.tile([-1, 1], n)
        panel["age_diff_vis2"] = center(panel["age_diff_vis2"])
        try:
            # lme4::glmer Laplace fits are approximated with statsmodels variational Bayes random-intercept GLMM.
            model = sm.BinomialBayesMixedGLM.from_formula("y ~ age_diff_vis2", {"subject": "0 + C(eid_8107)"}, panel)
            res = model.fit_vb()
            est, se = res.params[1], res.bse[1]
            var = float(np.exp(res.vcp_mean[0])) if len(res.vcp_mean) else np.nan
        except Exception:
            res = smf.gee("y ~ age_diff_vis2", groups="eid_8107", data=panel, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            est, se, var = res.params["age_diff_vis2"], res.bse["age_diff_vis2"], np.nan
        vid = int(voxel_ids[i])
        return {"estimate": est, "stderr": se, "zscore": est/se, "empir_vis1": r_linear_get(empir1, [vid])[0], "empir_vis2": r_linear_get(empir2, [vid])[0], "RaEvar": var, "voxel": i + 1}
    except Exception as exc:
        return {"error": repr(exc), "voxel": i + 1}

def main(argv=None):
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else N_CORES
    empir1, _ = load_nifti(Path(IMAGEDIR_VIS1) / "empir_prob_mask.nii.gz"); empir2, _ = load_nifti(Path(IMAGEDIR_VIS2) / "empir_prob_mask.nii.gz")
    ids = read_table("/well/nichols/users/kindalov/FMRIB/bianca_1vis_2vis_overlap.txt", header=False).iloc[:,0].astype(str).str.replace(r"_.*", "", regex=True)
    voxel_ids = read_table(Path(TEMPDIR) / "voxel_IDs_nonzero.dat", header=False).to_numpy().ravel()
    obj = load_rdata(Path(TEMPDIR) / "lesions_nonzero.RData"); v1, v2 = np.asarray(obj["lesions_vis1"]), np.asarray(obj["lesions_vis2"])
    df = read_table(Path(TEMPDIR) / "df_visits.dat", header=True)[["eid_8107","age_vis2","avg_age"]]
    jobs = [(i, v1, v2, ids, df, voxel_ids, empir1, empir2) for i in range(v1.shape[0])]
    output = Pool(n_cores).map(fit_glmer, jobs) if n_cores > 1 else [fit_glmer(j) for j in jobs]
    save_rdata(Path(TEMPDIR) / "glmer_output_1000test.RData", output=output)
if __name__ == "__main__": main()
