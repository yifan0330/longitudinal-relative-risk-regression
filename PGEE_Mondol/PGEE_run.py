#!/usr/bin/env python3
"""Voxel-wise PGEE runner ported from PGEE_run.R."""
from __future__ import annotations

import gc
import sys
from multiprocessing import Pool
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib  # kept as the Python equivalent of the original NIfTI dependency
import numpy as np
import pandas as pd
from scipy.io import loadmat
from data_io import load_array_data, save_pickle_data

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PGEE_source import geefirth

TEMP_DIR = PROJECT_ROOT / 'prelim/temp'
GEE_DIR = PROJECT_ROOT / 'PGEE_Mondol'
IMAGEDIR_VIS1 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis'
IMAGEDIR_VIS2 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis'
OVERLAP_FILE = PROJECT_ROOT.parent / 'bianca_1vis_2vis_overlap.txt'
N_CORES = 8


def _load_data_file(path: Path) -> dict:
    """Load Python npz/MAT files from a historical data filename."""
    data_error: Exception | None = None
    try:
        return load_array_data(path)
    except Exception as first_error:
        data_error = first_error
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load {path}; provide npz or MAT data with lesions_vis1 and lesions_vis2"
        ) from data_error or exc


def _read_ids() -> np.ndarray:
    subjs = pd.read_table(OVERLAP_FILE, header=None)[0].astype(str)
    return subjs.str.replace(r"_.*", "", regex=True).to_numpy()


def _read_visits() -> pd.DataFrame:
    df = pd.read_table(TEMP_DIR / "df_visits.dat", sep=r"\s+", engine="python")
    df = df[["eid_8107", "age_vis2", "avg_age"]]
    df["eid_8107"] = df["eid_8107"].astype(str)
    df["age_diff_vis2"] = df["age_vis2"] - df["avg_age"]
    return df


def _load_lesions(voxel_ids: np.ndarray | None = None):
    data = _load_data_file(TEMP_DIR / "lesions_atleast6.npz")
    lesions_vis1 = np.asarray(data["lesions_vis1"])
    lesions_vis2 = np.asarray(data["lesions_vis2"])
    if voxel_ids is not None:
        lesions_vis1 = lesions_vis1[voxel_ids, :]
        lesions_vis2 = lesions_vis2[voxel_ids, :]
    return lesions_vis1, lesions_vis2


def _fit_one(args):
    local_index, global_index, lesions1_row, lesions2_row, ids, df_visits = args
    n_subj = len(lesions1_row)
    panel = pd.DataFrame(
        {
            "y": np.r_[lesions1_row, lesions2_row],
            "vis": np.r_[np.repeat(1, n_subj), np.repeat(2, n_subj)],
            "eid_8107": np.r_[ids, ids].astype(str),
        }
    )
    panel = panel.merge(df_visits, on="eid_8107", how="left")
    panel["age_diff_vis2"] = panel["age_diff_vis2"] * np.tile([1, -1], n_subj)
    panel = panel.sort_values(["eid_8107", "vis"])
    x = panel[["avg_age", "age_diff_vis2"]]
    y = panel["y"].to_numpy(float)
    n_visits = panel["vis"].nunique()
    n_subjects = len(panel) // n_visits
    model = geefirth(
        y=y, x=x, id=np.repeat(np.arange(1, n_subjects + 1), n_visits), ar=False
    )
    return {
        "beta": model[0]["coefficients"].to_numpy(),
        "beta_se_sandwich": model[0]["std.err"].to_numpy(),
        "alpha": model[1],
        "voxel": int(local_index),
        "global_voxel": int(global_index),
    }


def _save_exact(path: Path, obj) -> None:
    save_pickle_data(obj, path)


def main(argv: list[str] | None = None) -> int:
    _ = argv or sys.argv[1:]
    ids = _read_ids()
    print("Everything loaded")
    df_visits = _read_visits()
    voxel_ids = (
        pd.read_table(GEE_DIR / "temp_atleast6" / "voxel_IDs_NAs.dat", header=None)[
            0
        ].to_numpy(dtype=int)
        - 1
    )
    lesions_vis1, lesions_vis2 = _load_lesions(voxel_ids)
    subset_size = 1000
    n_subsets = int(np.ceil(lesions_vis1.shape[0] / subset_size))
    del lesions_vis1, lesions_vis2
    gc.collect()

    print("Start running gee foreach fn")
    for j in range(1, n_subsets + 1):
        lesions_vis1, lesions_vis2 = _load_lesions(voxel_ids)
        start = subset_size * (j - 1)
        stop = min(subset_size * j, lesions_vis1.shape[0])
        subset_idx = np.arange(start, stop)
        print(j)
        tasks = [
            (
                i + 1,
                subset_idx[i] + 1,
                lesions_vis1[subset_idx[i], :],
                lesions_vis2[subset_idx[i], :],
                ids,
                df_visits,
            )
            for i in range(len(subset_idx))
        ]
        with Pool(processes=N_CORES) as pool:
            output = pool.map(_fit_one, tasks)
        _save_exact(GEE_DIR / "temp_atleast6" / f"PGEE_NAs_{j}.pkl", output)
        del output, lesions_vis1, lesions_vis2
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
