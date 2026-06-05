#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from data_io import load_array_data, load_pickle_data, save_pickle_data

TEMPDIR = PROJECT_ROOT / 'prelim/temp'
GEEDIR = PROJECT_ROOT / 'GEE_tests'
IMAGE_DIR_VIS1 = Path(
    str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis')
)
IMAGE_DIR_VIS2 = Path(
    str(PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis')
)
SUBJECT_FILE = PROJECT_ROOT.parent / 'bianca_1vis_2vis_overlap.txt'


def load_rdata(path: Path) -> dict:
    return load_array_data(path)


def save_output(obj, path: Path) -> None:
    save_pickle_data(obj, path)


def load_output(path: Path):
    return load_pickle_data(path)


def read_table(path: Path, header="infer") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=header, engine="python")


def load_ids() -> np.ndarray:
    df = read_table(SUBJECT_FILE, header=None)
    return df.iloc[:, 0].astype(str).str.replace(r"_.*", "", regex=True).to_numpy()


def load_visits() -> pd.DataFrame:
    df = read_table(TEMPDIR / "df_visits.dat", header=0)
    cols = list(df.columns)
    if len(cols) >= 3:
        cols[2] = "sexM"
        df.columns = cols
    keep = ["eid_8107", "age_vis2", "avg_age", "sexM"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df_visits.dat: {missing}")
    df = df.loc[:, keep].copy()
    df["eid_8107"] = df["eid_8107"].astype(str)
    df["age_diff_vis2"] = df["age_vis2"] - df["avg_age"]
    return df


def load_lesions() -> tuple[np.ndarray, np.ndarray]:
    data = load_array_data(
        TEMPDIR / "lesions_atleast6.npz", required=("lesions_vis1", "lesions_vis2")
    )
    if "lesions_vis1" not in data or "lesions_vis2" not in data:
        raise KeyError(
            "lesions_atleast6.npz must contain lesions_vis1 and lesions_vis2"
        )
    return np.asarray(data["lesions_vis1"], dtype=float), np.asarray(
        data["lesions_vis2"], dtype=float
    )


def subset_indices(n_rows: int, subset_size: int, j_1based: int) -> np.ndarray:
    start = subset_size * (j_1based - 1)
    stop = min(subset_size * j_1based, n_rows)
    return np.arange(start, stop, dtype=int)


def make_panel(
    local_i: int, subset_idx: np.ndarray, ids: np.ndarray, df_visits: pd.DataFrame
) -> pd.DataFrame:
    lesions_vis1, lesions_vis2 = load_lesions()
    lesions_vis1 = lesions_vis1[subset_idx, :]
    lesions_vis2 = lesions_vis2[subset_idx, :]
    n_subj = lesions_vis1.shape[1]
    panel = pd.DataFrame(
        {
            "y": np.r_[lesions_vis1[local_i, :], lesions_vis2[local_i, :]],
            "vis": np.r_[np.repeat(1, n_subj), np.repeat(2, n_subj)],
            "eid_8107": np.r_[ids, ids].astype(str),
        }
    )
    panel = panel.merge(df_visits, on="eid_8107", how="left")
    panel["age_diff_vis2"] = panel["age_diff_vis2"].to_numpy() * np.tile(
        [1, -1], n_subj
    )
    panel = panel.sort_values(["eid_8107", "vis"]).reset_index(drop=True)
    return panel


def design_matrix(panel: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(panel)),
            panel["avg_age"].to_numpy(dtype=float),
            panel["age_diff_vis2"].to_numpy(dtype=float),
            panel["sexM"].to_numpy(dtype=float),
        ]
    )


def run_parallel(fn, n_items: int, n_cores: int):
    if n_cores <= 1:
        return [fn(i) for i in range(n_items)]
    with ThreadPoolExecutor(max_workers=n_cores) as ex:
        return list(ex.map(fn, range(n_items)))


def fit_gee_statsmodels(panel: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Independence
    from statsmodels.genmod.families import Poisson

    n_visits = panel["vis"].nunique()
    n_subj = len(panel) // n_visits
    model = sm.GEE.from_formula(
        "y ~ avg_age + age_diff_vis2 + sexM",
        groups=np.repeat(np.arange(n_subj), n_visits),
        data=panel,
        family=Poisson(link=sm.families.links.Log()),
        cov_struct=Independence(),
    )
    result = model.fit(maxiter=10, cov_type="robust")
    return {
        "beta": result.params.to_numpy(),
        "beta_se_sandwich": result.bse.to_numpy(),
        "alpha": getattr(result.cov_struct, "dep_params", 0.0),
    }


def main() -> int:
    n_cores = 1
    ids = load_ids()
    print("Everything loaded")
    df_visits = load_visits()
    lesions_vis1, _ = load_lesions()
    subset_size = 1000
    n_subsets = math.ceil(lesions_vis1.shape[0] / subset_size)
    del lesions_vis1
    print("Start running gee foreach fn")

    for j in range(1, n_subsets + 1):
        lesions_vis1, _ = load_lesions()
        subset_idx = subset_indices(lesions_vis1.shape[0], subset_size, j)
        del lesions_vis1
        n_voxels = len(subset_idx)
        print(n_cores)
        print(j)

        def fit_one(i: int):
            try:
                panel = make_panel(i, subset_idx, ids, df_visits)
                out = fit_gee_statsmodels(panel)
                out["voxel"] = i + 1
                return out
            except Exception as exc:
                return exc

        output = run_parallel(fit_one, n_voxels, n_cores)
        save_output(output, GEEDIR / "temp_poisson_sexM" / f"GEE_subset_{j}.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
