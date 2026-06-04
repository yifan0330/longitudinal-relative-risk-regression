#!/usr/bin/env python3
"""Python translation of CVRanalysis/pgee_results.R.

Combines per-subset GEE outputs, writes the same NIfTI result filenames, and
recreates the main slice/scatter PDF plots using nibabel and matplotlib/seaborn.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from Sept21_pgee_logPoisson_dispersion_fn import (  # noqa: E402
    GEEDIR, IMAGEDIR_VIS1, IMAGEDIR_VIS2, TEMPDIR, load_rdata, save_rdata,
)

BRAIN_MASK_PATH = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii")
MNI152_PATH = Path("/well/nichols/users/kindalov/FMRIB/MNI152_T1_2mm_brain.nii.gz")
RESULT_DIR = GEEDIR / "results_Sept_pgee_interaction"
TEMP_OUTPUT_DIR = GEEDIR / "temp_Sept_pgee_interaction"
PLOT_DIR = RESULT_DIR / "plots"
NAMES_COVS = ["Intercept", "baseAge", "ageDiff", "baseCVR", "CVRdiff", "sexM", "headsize", "ageBYageDiff", "ageBYsexM"]
NAMES_COVS_PLOTS = ["Intercept", "Age (visit 1)", "Time difference", "CVR (visit 1)", "CVR diff.", "Sex", "Head size", "Age (visit 1):Time diff.", "Age (visit 1):Sex"]


def read_img(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata()), img


def write_img(data: np.ndarray, template: nib.Nifti1Image, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem if str(stem).endswith((".nii", ".nii.gz")) else Path(str(stem) + ".nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(data), template.affine, template.header), str(out))


def r_linear_get(data: np.ndarray, ids_1based: np.ndarray) -> np.ndarray:
    return data.ravel(order="F")[ids_1based - 1]


def r_linear_set(data: np.ndarray, ids_1based: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat = data.ravel(order="F")
    flat[ids_1based - 1] = np.asarray(values).reshape(-1)
    return flat.reshape(data.shape, order="F")


def finite_summary(name: str, values: np.ndarray) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        print(f"{name}: min={vals.min():.4g} median={np.median(vals):.4g} mean={vals.mean():.4g} max={vals.max():.4g}")
    else:
        print(f"{name}: all missing")


def extract_vector(item: dict, key: str, length: int) -> np.ndarray:
    if not isinstance(item, dict) or "error" in item or key not in item:
        return np.full(length, np.nan)
    arr = np.asarray(item[key], dtype=float).reshape(-1)
    if arr.size < length:
        return np.r_[arr, np.full(length - arr.size, np.nan)]
    return arr[:length]


def collect_results(n_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    estimates = np.zeros((n_voxels, len(NAMES_COVS)))
    stderror = np.zeros_like(estimates)
    alpha = np.zeros((n_voxels, 1))
    phi = np.zeros((n_voxels, 1))
    iterations = np.zeros((n_voxels, 1))
    output_all: list = [None] * n_voxels
    subset_size = 500
    n_subsets = int(math.ceil(n_voxels / subset_size))
    print(n_subsets)
    for j in range(1, n_subsets + 1):
        print(j)
        start = subset_size * (j - 1)
        stop = n_voxels if j == n_subsets else subset_size * j
        subset_idx = np.arange(start, stop)
        payload = load_rdata(TEMP_OUTPUT_DIR / f"GEE_subset_{j}.RData")
        output = payload.get("output", payload if isinstance(payload, list) else [])
        for offset, row in enumerate(subset_idx):
            item = output[offset] if offset < len(output) else {}
            estimates[row, :] = extract_vector(item, "beta", len(NAMES_COVS))
            stderror[row, :] = extract_vector(item, "beta_se_sandwich", len(NAMES_COVS))
            alpha[row, 0] = extract_vector(item, "alpha", 1)[0]
            phi[row, 0] = extract_vector(item, "phi", 1)[0]
            iterations[row, 0] = extract_vector(item, "iterations", 1)[0]
            output_all[row] = item
    return estimates, stderror, alpha, phi, iterations, output_all


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    tmp = np.empty_like(adj)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def plot_slice(slice_num: int, img: np.ndarray, mask: np.ndarray, mni152: np.ndarray, voxel_ids: np.ndarray,
               name: str | Path, title: str = "", legend: bool = False, vmin: float = -5, vmax: float = 5,
               cmap: str = "RdBu_r", fdr: bool = False, fdr_threshold: float = 0.05) -> None:
    z = img.copy().astype(float)
    valid = np.zeros(z.size, dtype=bool)
    valid[voxel_ids - 1] = True
    z.ravel(order="F")[~valid] = np.nan
    z[mask != 1] = np.nan
    if fdr:
        vals = r_linear_get(z, voxel_ids)
        adj = bh_adjust(2 * stats.norm.sf(np.abs(vals)))
        keep = np.zeros(z.size, dtype=bool)
        keep[voxel_ids - 1] = adj <= fdr_threshold
        z.ravel(order="F")[~keep] = np.nan
    z = np.clip(z, vmin, vmax)
    out = Path(f"{name}{slice_num}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    sl = max(slice_num - 1, 0)
    plt.figure(figsize=(7, 7))
    plt.imshow(np.rot90(mni152[:, :, sl]), cmap="gray", origin="lower")
    overlay = np.ma.masked_invalid(np.rot90(z[:, :, sl]))
    im = plt.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.78, origin="lower")
    plt.title(title, color="black")
    plt.axis("off")
    if legend:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def scatter_pdf(x: np.ndarray, y: np.ndarray, out: Path, xlabel: str, ylabel: str, limit: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(9, 7))
    if len(df):
        sns.kdeplot(data=df, x="x", y="y", fill=True, levels=20, cmap="viridis", thresh=0.02)
        sns.scatterplot(data=df.sample(min(len(df), 5000), random_state=1), x="x", y="y", s=5, alpha=0.15, color="black", linewidth=0)
    plt.xlim(0, limit); plt.ylim(0, limit)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    sns.despine()
    plt.tight_layout(); plt.savefig(out); plt.close()


def main() -> None:
    brain_mask, mask_template = read_img(BRAIN_MASK_PATH)
    mni152, _ = read_img(MNI152_PATH)
    empir_prob_vis1, _ = read_img(IMAGEDIR_VIS1 / "CVR_empir_prob_mask.nii.gz")
    empir_prob_vis2, _ = read_img(IMAGEDIR_VIS2 / "CVR_empir_prob_mask.nii.gz")
    voxel_ids = np.loadtxt(TEMPDIR / "voxel_IDs_CVR.dat", dtype=int).reshape(-1)

    estimates, stderror, alpha, phi, iterations, output_all = collect_results(len(voxel_ids))
    zscores = estimates / stderror
    image_zero = brain_mask.copy(); image_zero[image_zero != 0] = 0

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for i, cov in enumerate(NAMES_COVS):
        est_img = r_linear_set(image_zero.copy(), voxel_ids, estimates[:, i])
        se_img = r_linear_set(image_zero.copy(), voxel_ids, stderror[:, i])
        z_img = r_linear_set(image_zero.copy(), voxel_ids, zscores[:, i])
        write_img(est_img, mask_template, RESULT_DIR / f"estimate_{cov}_GEE")
        finite_summary(f"se_{cov}", r_linear_get(se_img, voxel_ids))
        write_img(se_img, mask_template, RESULT_DIR / f"se_{cov}_GEE")
        write_img(z_img, mask_template, RESULT_DIR / f"zscore_{cov}_GEE")
        print("-----")

    write_img(r_linear_set(image_zero.copy(), voxel_ids, alpha), mask_template, RESULT_DIR / "alpha_GEE")
    write_img(r_linear_set(image_zero.copy(), voxel_ids, phi), mask_template, RESULT_DIR / "phi_GEE")
    write_img(r_linear_set(image_zero.copy(), voxel_ids, iterations), mask_template, RESULT_DIR / "iterations_GEE")
    save_rdata(RESULT_DIR / "results_CVR_PGEE_1578subjs.Rdata", estimates=estimates, stderror=stderror,
               zscores=zscores, alpha=alpha, phi=phi, iterations=iterations, output_all=output_all,
               names_covs=NAMES_COVS)

    for i, cov in enumerate(NAMES_COVS):
        z_img, _ = read_img(RESULT_DIR / f"zscore_{cov}_GEE.nii.gz")
        finite_summary(f"|z|>1.96 {cov}", np.array([np.sum(np.abs(r_linear_get(z_img, voxel_ids)) > 1.96)]))
        plot_slice(45, z_img, brain_mask, mni152, voxel_ids,
                   RESULT_DIR / "plots" / f"{cov}_zscores_nofdr_z",
                   title=NAMES_COVS_PLOTS[i], legend=(i in (4, 8)), vmin=-5, vmax=5, cmap="RdBu_r")

    # Empirical probability plots.
    emp1 = np.sqrt(empir_prob_vis1.copy())
    for sl, leg in ((40, False), (45, False), (50, True)):
        plot_slice(sl, emp1, brain_mask, mni152, voxel_ids, GEEDIR / "plots" / "empirical" / "empir_vis1_z",
                   legend=leg, vmin=0, vmax=0.7, cmap="viridis")
    rr_emp = empir_prob_vis2 / empir_prob_vis1
    for sl, leg in ((40, False), (45, False), (50, True)):
        plot_slice(sl, rr_emp, brain_mask, mni152, voxel_ids, GEEDIR / "plots" / "empirical" / "empir_RR_z",
                   legend=leg, vmin=0, vmax=2, cmap="RdBu_r")

    # Alpha/phi diagnostic plots for GEE and PGEE where inputs exist.
    for subdir, prefix, result_name in (("gee", "", "results_July_gee_interaction"), ("pgee", "Sept_", "results_Sept_pgee_interaction")):
        for param, bounds in (("alpha", (-1, 1)), ("phi", (0, 2))):
            path = GEEDIR / result_name / f"{param}_GEE.nii.gz"
            if path.exists():
                img, _ = read_img(path)
                for sl, leg in ((40, False), (45, False), (50, True)):
                    plot_slice(sl, img, brain_mask, mni152, voxel_ids, GEEDIR / "plots" / subdir / f"{prefix}{param}_z",
                               title=param if sl == 40 else "", legend=leg, vmin=bounds[0], vmax=bounds[1], cmap="RdBu_r")

    # Risk-ratio images and scatter plots for age and CVR terms.
    rr_specs = [("baseAge", "rr_age_z", "Age (visit 1)", 2), ("ageDiff", "rr_ageDiff_z", "Time difference", 2),
                ("baseCVR", "rr_CVR_z", "CVR (visit 1)", 3), ("CVRdiff", "rr_CVRDiff_z", "CVR diff.", 3)]
    rr_images = {}
    for cov, plotname, title, vmax in rr_specs:
        beta, _ = read_img(RESULT_DIR / f"estimate_{cov}_GEE.nii.gz")
        rr = np.exp(beta); rr.ravel(order="F")[np.setdiff1d(np.arange(rr.size), voxel_ids - 1)] = np.nan
        rr_images[cov] = rr
        write_img(rr, mask_template, RESULT_DIR / f"rr_{cov}_GEE")
        for sl, leg in ((40, False), (45, False), (50, True)):
            plot_slice(sl, rr, brain_mask, mni152, voxel_ids, GEEDIR / "plots" / "pgee" / "rr" / plotname,
                       title=title if sl == 40 else "", legend=leg, vmin=0, vmax=vmax, cmap="RdBu_r")

    emp_vals = r_linear_get(empir_prob_vis1, voxel_ids)
    groups = [np.where(emp_vals < 0.0025)[0], np.where((emp_vals < 0.005) & (emp_vals >= 0.0025))[0],
              np.where((emp_vals < 0.01) & (emp_vals >= 0.005))[0], np.where(emp_vals >= 0.01)[0]]
    age_x = r_linear_get(rr_images["baseAge"], voxel_ids); age_y = r_linear_get(rr_images["ageDiff"], voxel_ids)
    cvr_x = r_linear_get(rr_images["baseCVR"], voxel_ids); cvr_y = r_linear_get(rr_images["CVRdiff"], voxel_ids)
    scatter_pdf(age_x, age_y, GEEDIR / "plots" / "pgee" / "rr" / "rr_age_ageDiff_scatter.pdf", "PGEE Age (visit 1)", "PGEE Time difference", 2)
    scatter_pdf(cvr_x, cvr_y, GEEDIR / "plots" / "pgee" / "rr" / "rr_CVR_CVRDiff_scatter.pdf", "PGEE CVR (visit 1)", "PGEE CVR difference", 3)
    for idx, grp in enumerate(groups, start=1):
        scatter_pdf(age_x[grp], age_y[grp], GEEDIR / "plots" / "pgee" / "rr" / f"rr_age_ageDiff_scatter_empir{idx}.pdf", "PGEE Age (visit 1)", "PGEE Time difference", 2)
        scatter_pdf(cvr_x[grp], cvr_y[grp], GEEDIR / "plots" / "pgee" / "rr" / f"rr_CVR_CVRDiff_scatter_empir{idx}.pdf", "PGEE CVR (visit 1)", "PGEE CVR difference", 3)


if __name__ == "__main__":
    main()
