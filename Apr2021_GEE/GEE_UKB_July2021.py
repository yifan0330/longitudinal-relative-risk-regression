#!/usr/bin/env python3
"""July 2021 UK Biobank GEE/PGEE interaction analysis translated to Python."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from GEE_UKB_May2021 import (
    BRAIN_MASK_PATH,
    GEEDIR,
    IMAGEDIR_VIS1,
    IMAGEDIR_VIS2,
    MNI152_PATH,
    TEMPDIR,
    analyze_se_ratios,
    compare_zscore_plots,
    extract_hat_indices,
    lesion_volume_scatter_and_gee,
    load_output_all,
    plot_se_histogram,
    read_nifti,
    read_voxel_ids,
    scalar_image_series,
    sex_estimate_summaries,
    zscore_image_series,
)

NAMES_COVS = ['Intercept', 'baseAge', 'ageDiff', 'sexM', 'headsize', 'ageBYageDiff', 'ageBYsexM']
NAMES_COVS_PLOTS = [
    'Intercept',
    'Age (visit 1)',
    'Time difference',
    'Sex',
    'Head size',
    'Age (visit 1):Time difference',
    'Age (visit 1):Sex',
]


def main() -> None:
    voxel_ids = read_voxel_ids()
    _, brain_mask = read_nifti(BRAIN_MASK_PATH)
    _, mni152 = read_nifti(MNI152_PATH)
    gee_results_dir = GEEDIR / 'results_July_gee_interaction'
    pgee_results_dir = GEEDIR / 'results_July_pgee_interaction'
    gee_path = gee_results_dir / 'results_interaction_GEE_2635subjs.Rdata'
    pgee_path = pgee_results_dir / 'results_interaction_GEE_2635subjs.Rdata'

    se_ratios_gee, idx_temp_gee, _ = analyze_se_ratios(gee_path, len(NAMES_COVS), 8, NAMES_COVS, 'GEE')
    print('GEE separated voxel count:', len(idx_temp_gee))
    sex_estimate_summaries(gee_results_dir, NAMES_COVS, voxel_ids, idx_temp_gee)
    keep_gee = np.ones(se_ratios_gee.shape[1], dtype=bool)
    keep_gee[idx_temp_gee] = False
    for i, name in enumerate(NAMES_COVS):
        from GEE_UKB_May2021 import print_summary
        print_summary(f'GEE non-separated SE ratio: {name}', se_ratios_gee[i, keep_gee], [0, 0.5, 1])

    se_ratios_pgee, idx_temp_pgee, _ = analyze_se_ratios(pgee_path, len(NAMES_COVS), 9, NAMES_COVS, 'PGEE')
    keep_pgee = np.ones(se_ratios_pgee.shape[1], dtype=bool)
    keep_pgee[idx_temp_pgee] = False
    for i, name in enumerate(NAMES_COVS):
        from GEE_UKB_May2021 import print_summary
        print_summary(f'PGEE non-separated SE ratio: {name}', se_ratios_pgee[i, keep_pgee], [0, 0.5, 1])
    hat = extract_hat_indices(load_output_all(pgee_path))
    if hat is not None:
        print('PGEE high leverage indices:', np.where(hat > 18 / (2 * 2635))[0] + 1)

    plot_se_histogram(se_ratios_gee, se_ratios_pgee, gee_results_dir / 'plots' / 'SEratios_sex_histogram.pdf')
    compare_zscore_plots(gee_results_dir, pgee_results_dir, gee_results_dir / 'plots', NAMES_COVS, voxel_ids, idx_temp_gee)
    zscore_image_series(gee_results_dir, NAMES_COVS, NAMES_COVS_PLOTS, voxel_ids, brain_mask, mni152, fdr=False)
    zscore_image_series(pgee_results_dir, NAMES_COVS, NAMES_COVS_PLOTS, voxel_ids, brain_mask, mni152, fdr=False)
    scalar_image_series(gee_results_dir, 'phi', voxel_ids, brain_mask, mni152, 0, 2, 'phi')
    scalar_image_series(pgee_results_dir, 'phi', voxel_ids, brain_mask, mni152, 0, 2, 'phi')
    scalar_image_series(gee_results_dir, 'alpha', voxel_ids, brain_mask, mni152, -1, 1, 'alpha')
    scalar_image_series(pgee_results_dir, 'alpha', voxel_ids, brain_mask, mni152, -1, 1, 'alpha')
    lesion_volume_scatter_and_gee(interaction_with_age_diff=True)


if __name__ == '__main__':
    main()
