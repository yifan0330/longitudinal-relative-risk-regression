#!/usr/bin/env python3
"""Voxelwise logistic GLM helper ported from glmer_fit.R."""

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


def _build_formula(datafile, n_covs_cat, n_covs_cont):
    names = list(datafile.columns)
    cat = [f"C({names[i]})" for i in range(1, n_covs_cat + 1)]
    cont_idx = range(n_covs_cat + 1, len(names) - 1)
    cont = [names[i] for i in cont_idx] if n_covs_cont else []
    for name in cont:
        datafile[name] = center(pd.to_numeric(datafile[name], errors="coerce"))
    rhs = " + ".join(cat + cont) or "1"
    return "voxel_lesion ~ " + rhs, datafile


def fit_glm_fn(datafile, lesionmat, n_covs_cat, n_covs_cont, GLMmethod=2, link_fn="logit", outputdir=None, subset=None, n_cores=8):
    datafile = datafile.copy()
    lesionmat = np.asarray(lesionmat)
    n_subjects = len(datafile)
    n_covs = datafile.shape[1] - 2
    if n_covs != n_covs_cat + n_covs_cont:
        raise ValueError("Data table covariates do not match supplied counts")
    formula, datafile = _build_formula(datafile, n_covs_cat, n_covs_cont)
    print(formula)
    family = sm.families.Binomial(sm.families.links.Probit() if link_fn == "probit" else sm.families.links.Logit())

    def fit_one(j):
        current = datafile.copy()
        current.insert(0, "voxel_lesion", lesionmat[j, :n_subjects].astype(int))
        try:
            # brglm2 mean-bias reduction has no exact statsmodels equivalent; standard GLM is used for this port.
            result = smf.glm(formula, family=family, data=current).fit(maxiter=10000 if GLMmethod == 2 else 100)
            return {"parameter": list(result.params.index), "zscore": result.tvalues.to_numpy(), "estimate": result.params.to_numpy(), "stderr": result.bse.to_numpy(), "converged_br": getattr(result, "converged", True), "status_br": True, "voxel": j + 1}
        except Exception as exc:
            return {"error": repr(exc), "voxel": j + 1}
    output = [fit_one(i) for i in range(lesionmat.shape[0])]
    method = "ML" if GLMmethod == 1 else "meanBR"
    if outputdir is not None:
        name = f"GLM_{method}_Nvars_{n_covs}_results.RData" if subset is None else f"GLM_subset_{subset}_{method}_Nvars_{n_covs}_results.RData"
        save_rdata(Path(outputdir) / name, output=output)
    return output
