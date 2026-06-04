#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

TEMPDIR = Path('/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp')
GEEDIR = Path('/well/nichols/users/kindalov/FMRIB/Longitudinal/Apr2021_GEE')
IMAGEDIR_VIS1 = Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw1vis')
IMAGEDIR_VIS2 = Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm_subjsw2vis')


def _load(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(float)


def _fvals(arr, ids):
    return arr.ravel(order='F')[ids - 1]


def _which(voxel_ids, value):
    idx = np.flatnonzero(voxel_ids == value)
    print(idx + 1)
    return idx


def _plot_voxel(voxel_id, title, voxel_ids, intercept, age, ageDiff, ageBYageDiff):
    idx = _which(voxel_ids, voxel_id)
    if len(idx) == 0:
        return
    v = voxel_ids[idx[0]]
    pos = v - 1
    age_test = np.arange(63.1 - 2 * 7.2, 63.1 + 2 * 7.2 + 1e-9, 0.2) - 63.1
    get = lambda a: a.ravel(order='F')[pos]
    plt.figure()
    plt.plot(age_test + 63.1, np.exp(get(intercept) + get(age) * age_test), linestyle='--')
    age_segments_diff_plot = np.r_[np.arange(50, 56), np.arange(60, 66), np.arange(70, 76)]
    age_segments_diff_est = np.tile(np.arange(0, 6), 3)
    age_segments_age = np.r_[np.repeat(50, 6), np.repeat(60, 6), np.repeat(70, 6)] - 63.1
    prob_age_diff = np.exp(get(intercept) + get(age) * age_segments_age + get(ageDiff) * age_segments_diff_est)
    prob_inter = np.exp(get(intercept) + get(age) * age_segments_age + get(ageDiff) * age_segments_diff_est + get(ageBYageDiff) * age_segments_age * age_segments_diff_est)
    for sl in [slice(0, 6), slice(6, 12), slice(12, 18)]:
        plt.plot(age_segments_diff_plot[sl], prob_age_diff[sl], linewidth=2)
        plt.plot(age_segments_diff_plot[sl], prob_inter[sl], linewidth=2, color='red')
    plt.xlabel('Age (visit 1)')
    plt.ylabel('Lesion probability (PGEE estimated)')
    plt.title(title)


def main():
    _load(Path('/well/nichols/users/kindalov/FMRIB/T2_lesions_MNI_2mm/MNI152_T1_2mm_brain_mask.nii'))
    empir_prob_vis1 = _load(IMAGEDIR_VIS1 / 'Apr2021_cleaned_empir_prob_mask.nii.gz')
    empir_prob_vis2 = _load(IMAGEDIR_VIS2 / 'Apr2021_cleaned_empir_prob_mask.nii.gz')
    voxel_ids = pd.read_csv(TEMPDIR / 'voxel_IDs_atleast6_cleaned_Apr2021.dat', header=None, sep=r'\s+', engine='python').to_numpy().reshape(-1).astype(int)
    names = ['Intercept', 'baseAge', 'ageDiff', 'sexM', 'headsize', 'ageBYageDiff', 'ageBYsexM']
    intercept = _load(GEEDIR / 'results_July_pgee_interaction' / f'estimate_{names[0]}_GEE.nii.gz')
    age = _load(GEEDIR / 'results_July_pgee_interaction' / f'estimate_{names[1]}_GEE.nii.gz')
    ageDiff = _load(GEEDIR / 'results_July_pgee_interaction' / f'estimate_{names[2]}_GEE.nii.gz')
    ageBYageDiff = _load(GEEDIR / 'results_July_pgee_interaction' / f'estimate_{names[5]}_GEE.nii.gz')
    for voxel, title in [(442745, 'Deep WM, voxel 30, 70, 45'), (394150, 'Deep WM, voxel 29, 81, 40'),
                         (491255, 'Periventricular WM, voxel 37, 58, 50'), (433016, 'Periventricular WM, voxel 38, 72, 44')]:
        _plot_voxel(voxel, title, voxel_ids, intercept, age, ageDiff, ageBYageDiff)
    plt.show()


if __name__ == '__main__':
    main()
