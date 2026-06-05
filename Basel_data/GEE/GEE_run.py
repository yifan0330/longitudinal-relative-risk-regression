#!/usr/bin/env python3
"""Python port of GEE_run.R: voxelwise binomial GEE using statsmodels."""

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


N_CORES = 8
TEMPDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/Basel_data/"
GEEDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/Basel_data/GEE/"
COV_FILE = "AgeSex_complete.dat"
FORMULA = "y ~ AGE + vis"
OUT_SUBDIR = "temp_vis"
MERGE_KEYS = ["PT"]
VIS_VALUES = [-1, 0, 1]


def _load_lesions():
    bs = np.asarray(load_rdata(Path(TEMPDIR) / "lesions_bs_5subj.RData")["lesions_bs"])
    y1 = np.asarray(load_rdata(Path(TEMPDIR) / "lesions_y1_5subj.RData")["lesions_y1"])
    y2 = np.asarray(load_rdata(Path(TEMPDIR) / "lesions_y2_5subj.RData")["lesions_y2"])
    return bs, y1, y2


def fit_gee(args):
    local_i, global_idx, df_vars = args
    try:
        lesions_bs, lesions_y1, lesions_y2 = _load_lesions()
        n_subj = lesions_bs.shape[1]
        y = np.concatenate(
            [lesions_bs[global_idx], lesions_y1[global_idx], lesions_y2[global_idx]]
        ).astype(float)
        pt_values = pd.unique(df_vars["PT"])
        panel = pd.DataFrame(
            {"y": y, "vis": np.repeat(VIS_VALUES, n_subj), "PT": np.tile(pt_values, 3)}
        )
        panel = panel.merge(df_vars, on=MERGE_KEYS, how="inner").sort_values(
            ["PT", "vis"]
        )
        n_visits = panel["vis"].nunique()
        n_subjects = int(len(panel) / n_visits)
        panel["cluster_id"] = np.repeat(np.arange(n_subjects), n_visits)
        result = smf.gee(
            FORMULA,
            groups="cluster_id",
            data=panel,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit(scale=1.0)
        return {
            "beta": result.params.to_numpy(),
            "beta_se_sandwich": result.bse.to_numpy(),
            "beta_names": list(result.params.index),
            "alpha": getattr(result.cov_struct, "dep_params", None),
            "voxel": int(local_i + 1),
        }
    except Exception as exc:
        return {"error": repr(exc), "voxel": int(local_i + 1)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n_cores = int(argv[0]) if argv else N_CORES
    print("Start running gee foreach fn")
    df_vars = read_table(Path(TEMPDIR) / COV_FILE, header=True)
    lesions_bs = _load_lesions()[0]
    subset_size = 1000
    n_subsets = int(math.ceil(lesions_bs.shape[0] / subset_size))
    n_rows = lesions_bs.shape[0]
    del lesions_bs
    gc.collect()
    for j in range(1, n_subsets + 1):
        idx = np.arange((j - 1) * subset_size, min(j * subset_size, n_rows))
        print(j)
        start = time.time()
        jobs = [(k, int(global_idx), df_vars) for k, global_idx in enumerate(idx)]
        if n_cores > 1 and len(jobs) > 1:
            with Pool(processes=n_cores) as pool:
                output = pool.map(fit_gee, jobs)
        else:
            output = [fit_gee(job) for job in jobs]
        print(time.time() - start)
        save_rdata(Path(GEEDIR) / OUT_SUBDIR / f"GEE_subset_{j}.RData", output=output)


if __name__ == "__main__":
    main()
