#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

TEMPDIR = Path('/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp')
GEEDIR = Path('/well/nichols/users/kindalov/FMRIB/Longitudinal/Apr2021_GEE')
IMAGEDIR_VIS1 = Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis')
IMAGEDIR_VIS2 = Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis')


def _read_pickle_or_rdata(path: Path) -> dict:
    try:
        with Path(path).open('rb') as f:
            return pickle.load(f)
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
        raise RuntimeError(f'Cannot read {path}; install rdata/pyreadr or use Python-generated pickle output') from exc


def _as_array(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    return np.asarray(obj)


def _load_lesions():
    data = _read_pickle_or_rdata(TEMPDIR / 'lesions_atleast6_cleaned_Apr2021.RData')
    return _as_array(data['lesions_vis1']), _as_array(data['lesions_vis2'])


def _load_nifti(path: Path):
    img = nib.load(str(path))
    return img, np.asanyarray(img.dataobj).astype(float)


def _save_like(data, like_img, stem: Path):
    path = Path(str(stem) if str(stem).endswith(('.nii', '.nii.gz')) else str(stem) + '.nii.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, like_img.affine, like_img.header), str(path))


def _set_voxels(template, voxel_ids, values):
    out = np.array(template, copy=True, dtype=float)
    flat = out.ravel(order='F')
    flat[voxel_ids - 1] = np.asarray(values).reshape(-1)
    return out


def _mask_outside(data, voxel_ids):
    out = np.array(data, copy=True, dtype=float)
    flat = out.ravel(order='F')
    keep = np.zeros(flat.shape, dtype=bool)
    keep[voxel_ids - 1] = True
    flat[~keep] = np.nan
    return out


def _summary(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {'min': np.nanmin(arr), 'median': np.nanmedian(arr), 'mean': np.nanmean(arr), 'max': np.nanmax(arr), 'nan': int(np.isnan(arr).sum())}


def main():
    brain_img, brain_mask = _load_nifti(Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii'))
    _load_nifti(IMAGEDIR_VIS1 / 'Apr2021_cleaned_empir_prob_mask.nii.gz')
    _load_nifti(IMAGEDIR_VIS2 / 'Apr2021_cleaned_empir_prob_mask.nii.gz')
    voxel_ids = pd.read_csv(TEMPDIR / 'voxel_IDs_atleast6_cleaned_Apr2021.dat', header=None, sep=r'\s+', engine='python').to_numpy().reshape(-1).astype(int)
    p = 7
    estimates = np.zeros((len(voxel_ids), p))
    stderror = np.zeros((len(voxel_ids), p))
    alpha = np.zeros((len(voxel_ids), 1))
    phi = np.zeros((len(voxel_ids), 1))
    iterations = np.zeros((len(voxel_ids), 1))
    output_all = [None] * len(voxel_ids)
    subset_size = 1000
    lesions_vis1, _ = _load_lesions()
    n_subsets = int(np.ceil(len(voxel_ids) / subset_size))
    print(n_subsets)
    for j in range(1, n_subsets + 1):
        print(j)
        start = subset_size * (j - 1)
        stop = min(len(voxel_ids), len(lesions_vis1)) if j == n_subsets else min(subset_size * j, len(voxel_ids))
        subset_idx = np.arange(start, stop)
        data = _read_pickle_or_rdata(GEEDIR / 'temp_July_gee_interaction' / f'GEE_subset_{j}.RData')
        output = data.get('output', data)
        for offset, out in enumerate(output):
            row = subset_idx[offset]
            output_all[row] = out
            if isinstance(out, dict) and len(out) >= 7:
                estimates[row, :] = np.asarray(out['beta']).reshape(-1)[:p]
                stderror[row, :] = np.asarray(out['beta_se_sandwich']).reshape(-1)[:p]
                alpha[row, 0] = out['alpha']
                phi[row, 0] = out['phi']
                iterations[row, 0] = out['iterations']
            else:
                estimates[row, :] = np.nan
                stderror[row, :] = np.nan
                alpha[row, 0] = np.nan
                phi[row, 0] = np.nan
                iterations[row, 0] = np.nan
    zscores = estimates / stderror
    names_covs = ['Intercept', 'baseAge', 'ageDiff', 'sexM', 'headsize', 'ageBYageDiff', 'ageBYsexM']
    image_template = np.where(brain_mask != 0, 0.0, brain_mask).astype(float)
    outdir = GEEDIR / 'results_July_gee_interaction'
    for i, name in enumerate(names_covs):
        _save_like(_set_voxels(image_template, voxel_ids, estimates[:, i]), brain_img, outdir / f'estimate_{name}_GEE')
        se_img = _set_voxels(image_template, voxel_ids, stderror[:, i])
        print(_summary(se_img.ravel(order='F')[voxel_ids - 1]))
        _save_like(se_img, brain_img, outdir / f'se_{name}_GEE')
        _save_like(_set_voxels(image_template, voxel_ids, zscores[:, i]), brain_img, outdir / f'zscore_{name}_GEE')
        print('-----')
    for label, values in [('alpha', alpha), ('phi', phi), ('iterations', iterations)]:
        img = _set_voxels(image_template, voxel_ids, values)
        print(_summary(img.ravel(order='F')[voxel_ids - 1]))
        _save_like(img, brain_img, outdir / f'{label}_GEE')
    with (outdir / 'results_interaction_GEE_2635subjs.Rdata').open('wb') as f:
        pickle.dump({k: v for k, v in locals().items() if k not in {'brain_img'}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    _, coef_age = _load_nifti(outdir / f'estimate_{names_covs[1]}_GEE.nii.gz')
    _, coef_ageDiff = _load_nifti(outdir / f'estimate_{names_covs[2]}_GEE.nii.gz')
    _, coef_ageBYageDiff = _load_nifti(outdir / f'estimate_{names_covs[5]}_GEE.nii.gz')
    for label, arr in [('RR_age', np.exp(coef_age)), ('RR_ageDiff', np.exp(coef_ageDiff)), ('RR_ageBYageDiff', np.exp(coef_age + coef_ageDiff + coef_ageBYageDiff))]:
        _save_like(_mask_outside(arr, voxel_ids), brain_img, outdir / label)
    plt.figure(); plt.plot(np.exp(coef_age).ravel(order='F')[voxel_ids - 1], np.exp(coef_ageDiff).ravel(order='F')[voxel_ids - 1], '.')
    plt.figure(); plt.plot(np.exp(coef_age).ravel(order='F')[voxel_ids - 1], np.exp(coef_age + coef_ageDiff + coef_ageBYageDiff).ravel(order='F')[voxel_ids - 1], '.')


if __name__ == '__main__':
    main()
