#!/usr/bin/env python3
"""May 2021 UK Biobank GEE/PGEE analysis translated to Python."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

TEMPDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp")
GEEDIR = Path("/well/nichols/users/kindalov/FMRIB/Longitudinal/Apr2021_GEE")
IMAGEDIR_VIS1 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis")
IMAGEDIR_VIS2 = Path("/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis")
BRAIN_MASK_PATH = Path(
    "/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii"
)
MNI152_PATH = Path("/well/nichols/users/kindalov/FMRIB/MNI152_T1_2mm_brain.nii.gz")

NAMES_COVS = ["Intercept", "baseAge", "ageDiff", "sexM", "headsize", "ageBYsexM"]
NAMES_COVS_PLOTS = [
    "Intercept",
    "Age (visit 1)",
    "Time difference",
    "Sex",
    "Head size",
    "Age (visit 1):Sex",
]
QUANTS_FULL = [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]


def read_payload(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            return obj
        return {"output_all": obj}
    except Exception:
        pass
    try:
        import rdata

        return rdata.conversion.convert(rdata.parser.parse_file(path))
    except Exception:
        pass
    try:
        import pyreadr

        return dict(pyreadr.read_r(str(path)))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read {path}; install rdata or pyreadr, or provide a Python pickle payload"
        ) from exc


def read_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    return img, np.asanyarray(img.dataobj).astype(float)


def read_voxel_ids(
    path: Path = TEMPDIR / "voxel_IDs_atleast6_cleaned_Apr2021.dat",
) -> np.ndarray:
    return (
        pd.read_csv(path, header=None, sep=r"\s+", engine="python")
        .to_numpy()
        .reshape(-1)
        .astype(int)
    )


def flat_values(img: np.ndarray, voxel_ids: np.ndarray) -> np.ndarray:
    return np.asarray(img).ravel(order="F")[np.asarray(voxel_ids, dtype=int) - 1]


def set_outside_voxels_nan(img: np.ndarray, voxel_ids: np.ndarray) -> np.ndarray:
    out = np.array(img, copy=True, dtype=float)
    flat = out.ravel(order="F")
    keep = np.zeros(flat.shape, dtype=bool)
    keep[np.asarray(voxel_ids, dtype=int) - 1] = True
    flat[~keep] = np.nan
    return out


def summary_pk(values: Iterable[float], quantiles: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    missing = int(np.isnan(arr).sum())
    arr = arr[~np.isnan(arr)]
    out = {"missing_values": missing}
    if arr.size == 0:
        out.update({"quantiles": {}, "mean": np.nan, "sd": np.nan, "zeroes": 0})
        return out
    out.update(
        {
            "quantiles": {float(q): float(np.quantile(arr, q)) for q in quantiles},
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "zeroes": int(np.sum(arr == 0)),
        }
    )
    return out


def print_summary(
    label: str, values: Iterable[float], quantiles: Iterable[float] = QUANTS_FULL
) -> dict[str, Any]:
    out = summary_pk(values, quantiles)
    print(label)
    print(out)
    return out


def _is_missing_output(obj: Any) -> bool:
    return (
        obj is None
        or (isinstance(obj, float) and np.isnan(obj))
        or (isinstance(obj, dict) and "error" in obj)
    )


def _len_output(obj: Any) -> int:
    if _is_missing_output(obj):
        return 0
    if isinstance(obj, dict):
        return len(obj)
    try:
        return len(obj)
    except TypeError:
        return 0


def _field(obj: Any, position: int, names: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        keys = list(obj.keys())
        if 0 <= position - 1 < len(keys):
            return obj[keys[position - 1]]
        raise KeyError(names[0])
    return obj[position - 1]


def load_output_all(path: Path) -> list[Any]:
    payload = read_payload(path)
    if "output_all" in payload:
        obj = payload["output_all"]
    elif "output" in payload:
        obj = payload["output"]
    else:
        obj = next(iter(payload.values()))
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].tolist() if obj.shape[1] == 1 else obj.to_dict("records")
    if isinstance(obj, np.ndarray):
        return obj.reshape(-1).tolist()
    return list(obj)


def matrix_by_cols(values: Any, ncol: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] == ncol:
            return arr
        if arr.shape[0] == ncol:
            return arr.T
    arr = arr.reshape(-1)
    return arr.reshape((-1, ncol), order="F")


def se_ratio_last_iter(output_all: list[Any], p: int, expected_len: int) -> np.ndarray:
    cols = []
    for out in output_all:
        if _is_missing_output(out) or (
            _len_output(out) not in {expected_len, expected_len - 1, expected_len + 1}
            and not isinstance(out, dict)
        ):
            cols.append(np.full(p, np.nan))
            continue
        try:
            trace = _field(out, 3, ("beta_se_model_trace", "se_model_trace"))
            mat = matrix_by_cols(trace, p)
            ratio = mat / mat[0, :]
            cols.append(ratio[-1, :])
        except Exception:
            cols.append(np.full(p, np.nan))
    return np.column_stack(cols)


def extract_iterations(output_all: list[Any], expected_len: int) -> np.ndarray:
    vals = []
    for out in output_all:
        if _is_missing_output(out):
            vals.append(np.nan)
            continue
        try:
            vals.append(
                float(np.asarray(_field(out, 7, ("iterations",))).reshape(-1)[0])
            )
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals)


def extract_model_se(output_all: list[Any], p: int, covariate_index: int) -> np.ndarray:
    vals = []
    for out in output_all:
        if _is_missing_output(out):
            vals.append(np.nan)
            continue
        try:
            se = _field(out, 2, ("beta_se_model", "se_model"))
            mat = np.asarray(se, dtype=float)
            vals.append(float(mat.reshape(-1)[covariate_index]))
        except Exception:
            try:
                trace = matrix_by_cols(
                    _field(out, 3, ("beta_se_model_trace", "se_model_trace")), p
                )
                vals.append(float(trace[-1, covariate_index]))
            except Exception:
                vals.append(np.nan)
    return np.asarray(vals)


def extract_hat_indices(output_all: list[Any]) -> np.ndarray | None:
    vals = []
    for out in output_all:
        if _is_missing_output(out):
            continue
        try:
            vals.append(np.asarray(_field(out, 8, ("H", "hat"))).reshape(-1))
        except Exception:
            continue
    if not vals:
        return None
    return np.concatenate(vals)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_se_histogram(
    se_ratios_gee: np.ndarray, se_ratios_pgee: np.ndarray, out_path: Path
) -> None:
    dat = pd.DataFrame(
        {
            "x": np.r_[se_ratios_gee[3, :], se_ratios_pgee[3, :]],
            "group": ["GEE"] * se_ratios_gee.shape[1]
            + ["PGEE"] * se_ratios_pgee.shape[1],
        }
    )
    grid = sns.FacetGrid(
        dat, col="group", height=5, aspect=1.4, sharex=True, sharey=False
    )
    grid.map_dataframe(sns.histplot, x="x", bins=30)
    grid.set(xlim=(0, 10), xlabel="Sex SE ratio")
    grid.figure.savefig(out_path)
    plt.close(grid.figure)


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    limits: tuple[float, float],
    smooth: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=4, alpha=0.15, edgecolors="none")
    if smooth:
        try:
            sns.regplot(x=x, y=y, lowess=True, scatter=False, color="red", ax=ax)
        except Exception:
            pass
    lo, hi = limits
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="red" if xlabel.startswith("p") else "black",
        linewidth=0.7,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_figure(out_path)


def p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.minimum.accumulate(
        (ranked.size / np.arange(ranked.size, 0, -1)) * ranked[::-1]
    )[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(pv)
    tmp[order] = adjusted
    out[ok] = tmp
    return out


def zscores_plot(
    slice_num: int,
    mni152: np.ndarray,
    img: np.ndarray,
    mask: np.ndarray,
    voxel_ids: np.ndarray,
    fdr: bool,
    fdr_threshold: float,
    name: Path,
    legend: bool,
    vmin: float,
    vmax: float,
    title: str = "",
    cmap: str = "RdBu_r",
) -> None:
    z_score = set_outside_voxels_nan(img, voxel_ids)
    z_score = np.where(mask == 1, z_score, np.nan)
    if fdr:
        values = flat_values(z_score, voxel_ids)
        raw_p = 2 * norm.sf(np.abs(values))
        adj = p_adjust_bh(raw_p)
        print("-------------")
        print("FDR 5%")
        print(pd.Series(adj <= fdr_threshold).value_counts(dropna=False))
        if np.any(adj <= fdr_threshold):
            threshold = np.nanmin(np.abs(values[adj <= fdr_threshold]))
            print(f"FDR adjusted z-score threshold is +/- {threshold}")
        flat = z_score.ravel(order="F")
        flat[np.asarray(voxel_ids)[adj > fdr_threshold] - 1] = np.nan
    z_score = np.clip(z_score, vmin, vmax)
    idx = slice_num - 1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(mni152[:, :, idx]), cmap="gray", interpolation="nearest")
    overlay = np.ma.masked_invalid(np.rot90(z_score[:, :, idx]))
    im = ax.imshow(
        overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, interpolation="nearest"
    )
    ax.set_axis_off()
    if title:
        ax.text(
            0.03,
            0.97,
            title,
            color="white",
            fontsize=18,
            transform=ax.transAxes,
            va="top",
        )
    if legend:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(Path(f"{name}{slice_num}.pdf"))


def analyze_se_ratios(
    result_path: Path, p: int, expected_len: int, names_covs: list[str], prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_all = load_output_all(result_path)
    se_ratios = se_ratio_last_iter(output_all, p, expected_len)
    for i, name in enumerate(names_covs):
        print_summary(f"{prefix} SE ratio: {name}", se_ratios[i, :])
    covs = [np.nan_to_num(se_ratios[i, :] > 10, nan=False) for i in range(p)]
    for i, cov in enumerate(covs):
        print(f"{prefix} {names_covs[i]} SE ratio > 10: {int(cov.sum())}")
    idx_temp = np.where(np.logical_or.reduce(covs))[0]
    print(
        f"{prefix} all covariates > 10:", np.where(np.logical_and.reduce(covs))[0] + 1
    )
    print(f"{prefix} ageDiff or sexM > 10:", np.where(covs[2] | covs[3])[0] + 1)
    na_ids = np.where(np.isnan(se_ratios[3, :]))[0]
    iters = extract_iterations(output_all, expected_len)
    print_summary(f"{prefix} iterations", iters, [0, 0.25, 0.5, 0.75, 1])
    print(f"{prefix} iteration table")
    print(pd.Series(iters).value_counts(dropna=False).sort_index())
    return se_ratios, idx_temp, na_ids


def sex_estimate_summaries(
    results_dir: Path,
    names_covs: list[str],
    voxel_ids: np.ndarray,
    exclude_idx: np.ndarray,
) -> None:
    _, sex_est = read_nifti(results_dir / f"estimate_{names_covs[3]}_GEE.nii.gz")
    print_summary(
        "sexM estimate: all voxels",
        flat_values(sex_est, voxel_ids),
        [0, 0.25, 0.5, 0.75, 0.99, 1],
    )
    keep = np.ones(len(voxel_ids), dtype=bool)
    keep[exclude_idx] = False
    print_summary(
        "sexM estimate: non-separated voxels",
        flat_values(sex_est, voxel_ids[keep]),
        [0, 0.25, 0.5, 0.75, 0.99, 1],
    )


def compare_zscore_plots(
    gee_dir: Path,
    pgee_dir: Path,
    plot_dir: Path,
    names_covs: list[str],
    voxel_ids: np.ndarray,
    idx_temp_gee: np.ndarray,
    output_all_gee: list[Any] | None = None,
    include_model_variance: bool = False,
) -> None:
    _, z_sex_gee_img = read_nifti(gee_dir / f"zscore_{names_covs[3]}_GEE.nii.gz")
    _, z_sex_pgee_img = read_nifti(pgee_dir / f"zscore_{names_covs[3]}_GEE.nii.gz")
    z_sex_gee = flat_values(z_sex_gee_img, voxel_ids)
    z_sex_pgee = flat_values(z_sex_pgee_img, voxel_ids)
    plot_scatter(
        z_sex_gee[idx_temp_gee],
        z_sex_pgee[idx_temp_gee],
        plot_dir / "zscores_sex_sep_scatter.pdf",
        "GEE Sex z-scores",
        "PGEE Sex z-scores",
        (-20, 20),
    )
    keep = np.ones(len(voxel_ids), dtype=bool)
    keep[idx_temp_gee] = False
    plot_scatter(
        z_sex_gee[keep],
        z_sex_pgee[keep],
        plot_dir / "zscores_sex_nonsep_scatter.pdf",
        "GEE Sex z-scores",
        "PGEE Sex z-scores",
        (-20, 20),
    )
    if include_model_variance and output_all_gee is not None:
        _, beta_img = read_nifti(gee_dir / f"estimate_{names_covs[3]}_GEE.nii.gz")
        beta = flat_values(beta_img, voxel_ids)
        model_se = extract_model_se(output_all_gee, len(names_covs), 3)
        z_model = beta / model_se
        plot_scatter(
            z_model[idx_temp_gee],
            z_sex_pgee[idx_temp_gee],
            plot_dir / "zscores_sex_sep_modelVar_scatter.pdf",
            "GEE Sex z-scores",
            "PGEE Sex z-scores",
            (-20, 20),
        )
        plot_scatter(
            z_model[keep],
            z_sex_pgee[keep],
            plot_dir / "zscores_sex_nonsep_modelVar_scatter.pdf",
            "GEE Sex z-scores",
            "PGEE Sex z-scores",
            (-20, 20),
        )


def zscore_image_series(
    results_dir: Path,
    names_covs: list[str],
    names_covs_plots: list[str],
    voxel_ids: np.ndarray,
    brain_mask: np.ndarray,
    mni152: np.ndarray,
    fdr: bool = False,
) -> None:
    p = len(names_covs)
    for i, name in enumerate(names_covs):
        _, z_temp = read_nifti(results_dir / f"zscore_{name}_GEE.nii.gz")
        z_vals = flat_values(set_outside_voxels_nan(z_temp, voxel_ids), voxel_ids)
        print(name, int(np.sum(np.abs(z_vals[~np.isnan(z_vals)]) > 1.96)))
        zscores_plot(
            45,
            mni152,
            z_temp,
            brain_mask,
            voxel_ids,
            fdr,
            0.05,
            results_dir / "plots" / f'{name}_zscores_{"fdr" if fdr else "nofdr"}_z',
            i == p - 1,
            -5,
            5,
            names_covs_plots[i],
        )


def scalar_image_series(
    results_dir: Path,
    image_name: str,
    voxel_ids: np.ndarray,
    brain_mask: np.ndarray,
    mni152: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    _, img = read_nifti(results_dir / f"{image_name}_GEE.nii.gz")
    for slice_num, legend in [(40, False), (45, False), (50, True)]:
        zscores_plot(
            slice_num,
            mni152,
            img,
            brain_mask,
            voxel_ids,
            False,
            0.05,
            results_dir / "plots" / f"{image_name}_z",
            legend,
            vmin,
            vmax,
            title if slice_num == 40 else "",
            cmap,
        )


def empirical_probability_plots(
    voxel_ids: np.ndarray, brain_mask: np.ndarray, mni152: np.ndarray, plot_dir: Path
) -> np.ndarray:
    _, empir_prob_vis1 = read_nifti(
        IMAGEDIR_VIS1 / "Apr2021_cleaned_empir_prob_mask.nii.gz"
    )
    _, empir_prob_vis2 = read_nifti(
        IMAGEDIR_VIS2 / "Apr2021_cleaned_empir_prob_mask.nii.gz"
    )
    for label, img, title in [
        ("empir_vis1", empir_prob_vis1, "sqrt(p1)"),
        ("empir_vis2", empir_prob_vis2, "sqrt(p2)"),
    ]:
        for slice_num, legend in [(40, False), (45, False), (50, True)]:
            zscores_plot(
                slice_num,
                mni152,
                np.sqrt(img),
                brain_mask,
                voxel_ids,
                False,
                0.05,
                plot_dir / f"{label}_z",
                legend,
                0,
                0.7,
                title if slice_num == 40 else "",
                "viridis",
            )
    rr = empir_prob_vis2 / empir_prob_vis1
    finite_ids = voxel_ids[np.isfinite(flat_values(rr, voxel_ids))]
    for slice_num, legend in [(40, False), (45, False), (50, True)]:
        zscores_plot(
            slice_num,
            mni152,
            rr,
            brain_mask,
            finite_ids,
            False,
            0.05,
            plot_dir / "empir_RR_z",
            legend,
            0,
            2,
            "p2/p1" if slice_num == 40 else "",
            "RdBu_r",
        )
    return finite_ids


def empirical_scatterplots(voxel_ids: np.ndarray, plot_dir: Path) -> None:
    _, p1_img = read_nifti(IMAGEDIR_VIS1 / "Apr2021_cleaned_empir_prob_mask.nii.gz")
    _, p2_img = read_nifti(IMAGEDIR_VIS2 / "Apr2021_cleaned_empir_prob_mask.nii.gz")
    x = flat_values(p1_img, voxel_ids)
    y = flat_values(p2_img, voxel_ids)
    print(
        "p2 > p1, p2 == p1, p2 < p1:",
        int(np.sum(y > x)),
        int(np.sum(y == x)),
        int(np.sum(y < x)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, lim in zip(axes.ravel(), [1, 0.5, 0.25, 0.1, 0.05, 0.01]):
        ax.scatter(x, y, s=2, alpha=0.1, edgecolors="none")
        ax.plot([0, lim], [0, lim], linestyle="--", color="red", linewidth=0.7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("p1")
        ax.set_ylabel("p2")
    _save_figure(plot_dir / "empir_p2_vs_p1_scatter_grid.pdf")


def alpha_rr_plot(voxel_ids: np.ndarray, results_dir: Path, plot_dir: Path) -> None:
    _, p1_img = read_nifti(IMAGEDIR_VIS1 / "Apr2021_cleaned_empir_prob_mask.nii.gz")
    _, p2_img = read_nifti(IMAGEDIR_VIS2 / "Apr2021_cleaned_empir_prob_mask.nii.gz")
    _, alpha_img = read_nifti(results_dir / "alpha_GEE.nii.gz")
    p1 = flat_values(p1_img, voxel_ids)
    rr = flat_values(p2_img, voxel_ids) / p1
    alpha = flat_values(alpha_img, voxel_ids)
    keep = p1 != 0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rr[keep], alpha[keep], s=3, alpha=0.12, edgecolors="none")
    try:
        sns.regplot(
            x=rr[keep], y=alpha[keep], lowess=True, scatter=False, color="red", ax=ax
        )
    except Exception:
        pass
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("p2/p1")
    ax.set_ylabel("alpha")
    _save_figure(plot_dir / "alpha_vs_empir_RR.pdf")


def separated_voxel_plot(
    voxel_ids: np.ndarray,
    idx_temp_gee: np.ndarray,
    brain_mask: np.ndarray,
    mni152: np.ndarray,
    plot_dir: Path,
) -> None:
    _, empir_prob_vis1 = read_nifti(
        IMAGEDIR_VIS1 / "Apr2021_cleaned_empir_prob_mask.nii.gz"
    )
    img = set_outside_voxels_nan(empir_prob_vis1, voxel_ids)
    flat = img.ravel(order="F")
    flat[voxel_ids - 1] = np.nan
    flat[voxel_ids[idx_temp_gee] - 1] = 10
    zscores_plot(
        50,
        mni152,
        img,
        brain_mask,
        voxel_ids,
        False,
        0.05,
        plot_dir / "sep_z",
        False,
        0,
        10,
        "",
        "viridis",
    )


def lesion_volume_scatter_and_gee(
    interaction_with_age_diff: bool, plot_dir: Path | None = None
) -> Any:
    import statsmodels.api as sm

    df = pd.read_csv(
        TEMPDIR / "df_visits_cleaned_Apr2021.dat", sep=r"\s+", engine="python"
    )
    cols = list(df.columns)
    cols[2] = "sexM"
    df.columns = cols
    ids = df["eid_8107"].to_numpy()
    n_subj = len(df)
    df["headsize"] = (df["X25000.2.0"] + df["X25000.3.0"]) / 2
    df = df[
        [
            "eid_8107",
            "age_vis2",
            "age_vis3",
            "sexM",
            "headsize",
            "X25781.2.0",
            "X25781.3.0",
        ]
    ].copy()
    df["age_diff"] = df["age_vis3"] - df["age_vis2"]
    df["age_vis2"] = df["age_vis2"] - df["age_vis2"].mean()
    df["headsize"] = df["headsize"] - df["headsize"].mean()
    y1 = df["X25781.2.0"].to_numpy(float)
    y2 = df["X25781.3.0"].to_numpy(float)
    print_summary("Summary lesion volume - visit 1", y1, [0, 0.25, 0.5, 0.75, 1])
    print_summary("Summary lesion volume - visit 2", y2, [0, 0.25, 0.5, 0.75, 1])
    if plot_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, lim in zip(axes, [10000, 2500, 1000]):
            ax.scatter(y1, y2, s=4, alpha=0.12, edgecolors="none")
            ax.plot([0, lim], [0, lim], linestyle="--", color="red", linewidth=0.7)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("volume1")
            ax.set_ylabel("volume2")
        _save_figure(plot_dir / "lesion_volume_scatter_grid.pdf")
    panel = pd.DataFrame(
        {
            "y": np.r_[y1, y2],
            "vis": np.r_[np.ones(n_subj, dtype=int), np.full(n_subj, 2, dtype=int)],
            "eid_8107": np.r_[ids, ids],
        }
    )
    panel = (
        panel.merge(df, on="eid_8107", how="left")
        .sort_values(["eid_8107", "vis"])
        .reset_index(drop=True)
    )
    panel["age_diff"] = panel["age_diff"].to_numpy() * np.tile([0, 1], n_subj)
    y = np.log(panel["y"].replace(0, np.nan).to_numpy(float))
    exog_parts = [
        np.ones(len(panel)),
        panel["age_vis2"],
        panel["age_diff"],
        panel["sexM"],
        panel["headsize"],
        panel["age_vis2"] * panel["sexM"],
    ]
    names = ["Intercept", "age_vis2", "age_diff", "sexM", "headsize", "age_vis2:sexM"]
    if interaction_with_age_diff:
        exog_parts.append(panel["age_vis2"] * panel["age_diff"])
        names.append("age_vis2:age_diff")
    exog = np.column_stack([np.asarray(c, dtype=float) for c in exog_parts])
    keep = np.isfinite(y) & np.all(np.isfinite(exog), axis=1)
    model = sm.GEE(
        y[keep],
        exog[keep, :],
        groups=panel.loc[keep, "eid_8107"],
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()
    print(
        pd.DataFrame(
            {
                "coef": result.params,
                "std_err": result.bse,
                "z": result.tvalues,
                "p": result.pvalues,
            },
            index=names,
        )
    )
    return result


def main() -> None:
    voxel_ids = read_voxel_ids()
    _, brain_mask = read_nifti(BRAIN_MASK_PATH)
    _, mni152 = read_nifti(MNI152_PATH)
    gee_results_dir = GEEDIR / "results_gee_exch"
    pgee_results_dir = GEEDIR / "results_pgee_exch"
    gee_output = load_output_all(gee_results_dir / "results_exch_GEE_2635subjs.Rdata")
    se_ratios_gee, idx_temp_gee, _ = analyze_se_ratios(
        gee_results_dir / "results_exch_GEE_2635subjs.Rdata",
        len(NAMES_COVS),
        8,
        NAMES_COVS,
        "GEE",
    )
    sex_estimate_summaries(gee_results_dir, NAMES_COVS, voxel_ids, idx_temp_gee)
    for i, name in enumerate(NAMES_COVS):
        keep = np.ones(se_ratios_gee.shape[1], dtype=bool)
        keep[idx_temp_gee] = False
        print_summary(
            f"GEE non-separated SE ratio: {name}", se_ratios_gee[i, keep], [0, 0.5, 1]
        )
    se_ratios_pgee, idx_temp_pgee, _ = analyze_se_ratios(
        pgee_results_dir / "results_exch_GEE_2635subjs.Rdata",
        len(NAMES_COVS),
        9,
        NAMES_COVS,
        "PGEE",
    )
    hat = extract_hat_indices(
        load_output_all(pgee_results_dir / "results_exch_GEE_2635subjs.Rdata")
    )
    if hat is not None:
        print("PGEE high leverage indices:", np.where(hat > 18 / (2 * 2635))[0] + 1)
    plot_se_histogram(
        se_ratios_gee,
        se_ratios_pgee,
        gee_results_dir / "plots" / "SEratios_sex_histogram.pdf",
    )
    compare_zscore_plots(
        gee_results_dir,
        pgee_results_dir,
        gee_results_dir / "plots",
        NAMES_COVS,
        voxel_ids,
        idx_temp_gee,
        gee_output,
        include_model_variance=True,
    )
    zscore_image_series(
        gee_results_dir,
        NAMES_COVS,
        NAMES_COVS_PLOTS,
        voxel_ids,
        brain_mask,
        mni152,
        fdr=True,
    )
    zscore_image_series(
        pgee_results_dir,
        NAMES_COVS,
        NAMES_COVS_PLOTS,
        voxel_ids,
        brain_mask,
        mni152,
        fdr=True,
    )
    scalar_image_series(
        gee_results_dir, "phi", voxel_ids, brain_mask, mni152, 0, 2, "phi"
    )
    scalar_image_series(
        pgee_results_dir, "phi", voxel_ids, brain_mask, mni152, 0, 2, "phi"
    )
    scalar_image_series(
        gee_results_dir, "alpha", voxel_ids, brain_mask, mni152, -1, 1, "alpha"
    )
    scalar_image_series(
        pgee_results_dir, "alpha", voxel_ids, brain_mask, mni152, -1, 1, "alpha"
    )
    finite_ids = empirical_probability_plots(
        voxel_ids, brain_mask, mni152, gee_results_dir / "plots"
    )
    empirical_scatterplots(finite_ids, gee_results_dir / "plots")
    alpha_rr_plot(voxel_ids, pgee_results_dir, gee_results_dir / "plots")
    separated_voxel_plot(
        voxel_ids, idx_temp_gee, brain_mask, mni152, gee_results_dir / "plots"
    )
    lesion_volume_scatter_and_gee(
        interaction_with_age_diff=False, plot_dir=gee_results_dir / "plots"
    )


if __name__ == "__main__":
    main()
