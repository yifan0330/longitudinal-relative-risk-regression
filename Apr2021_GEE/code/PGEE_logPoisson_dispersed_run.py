#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

TEMPDIR = PROJECT_ROOT / 'prelim/temp'
GEEDIR = PROJECT_ROOT / 'Apr2021_GEE'
IMAGEDIR_VIS1 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw1vis'
IMAGEDIR_VIS2 = PROJECT_ROOT.parent / 'T2_lesions_MNI_2mm_subjsw2vis'


def _read_rdata(path: Path) -> dict:
    path = Path(path)
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
            f"Cannot read RData file {path}; install rdata or pyreadr"
        ) from exc


def _save_payload(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _as_array(obj) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    return np.asarray(obj)


def _load_lesions(cleaned: bool = True):
    name = (
        "lesions_atleast6_cleaned_Apr2021.RData"
        if cleaned
        else "lesions_atleast6_Apr2021.RData"
    )
    data = _read_rdata(TEMPDIR / name)
    return _as_array(data["lesions_vis1"]), _as_array(data["lesions_vis2"])


def _load_function(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, name)


def _subset_indices(n_rows: int, subset_size: int, j: int) -> np.ndarray:
    start = subset_size * (j - 1)
    stop = min(subset_size * j, n_rows)
    if start < 0 or start >= n_rows:
        raise ValueError(f"subset {j} is outside 1..{math.ceil(n_rows / subset_size)}")
    return np.arange(start, stop)


def _prepare_cleaned_visits() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(
        TEMPDIR / "df_visits_cleaned_Apr2021.dat", sep=r"\s+", engine="python"
    )
    ids = df["eid_8107"].to_numpy()
    cols = list(df.columns)
    cols[2] = "sexM"
    df.columns = cols
    print(df["sexM"].value_counts().sort_index())
    df["headsize"] = (df["X25000.2.0"] + df["X25000.3.0"]) / 2
    df = df[["eid_8107", "age_vis2", "age_vis3", "sexM", "headsize"]].copy()
    df["age_diff"] = df["age_vis3"] - df["age_vis2"]
    df["age_vis2"] = df["age_vis2"] - df["age_vis2"].mean()
    df["headsize"] = df["headsize"] - df["headsize"].mean()
    return df, ids


def _prepare_prior_visits() -> tuple[pd.DataFrame, np.ndarray]:
    subjects = pd.read_csv(
        str(PROJECT_ROOT.parent / 'bianca_1vis_2vis_overlap_filenames_Apr2021.txt'),
        header=None,
        sep=r"\s+",
        engine="python",
    )
    ids = subjects.iloc[:, 0].astype(str).str.replace(r"_.*", "", regex=True).to_numpy()
    df = pd.read_csv(TEMPDIR / "df_visits_Apr2021.dat", sep=r"\s+", engine="python")
    cols = list(df.columns)
    cols[2] = "sexM"
    df.columns = cols
    print(df["sexM"].value_counts().sort_index())
    df = df[["eid_8107", "age_vis2", "avg_age", "sexM"]].copy()
    df["age_diff_vis2"] = df["age_vis2"] - df["avg_age"]
    return df, ids


def _panel_for_voxel(
    i: int,
    lesions_vis1: np.ndarray,
    lesions_vis2: np.ndarray,
    subset_idx: np.ndarray,
    ids: np.ndarray,
    df_visits: pd.DataFrame,
    cleaned: bool,
):
    n_subj = lesions_vis1.shape[1]
    voxel = subset_idx[i]
    panel = pd.DataFrame(
        {
            "y": np.r_[lesions_vis1[voxel, :], lesions_vis2[voxel, :]],
            "vis": np.r_[np.ones(n_subj, dtype=int), np.full(n_subj, 2, dtype=int)],
            "eid_8107": np.r_[ids, ids],
        }
    )
    panel = panel.merge(df_visits, on="eid_8107", how="left")
    panel = panel.sort_values(["eid_8107", "vis"]).reset_index(drop=True)
    if cleaned:
        panel["age_diff"] = panel["age_diff"].to_numpy() * np.tile([0, 1], n_subj)
    else:
        panel["age_diff_vis2"] = panel["age_diff_vis2"].to_numpy() * np.tile(
            [1, -1], n_subj
        )
    return panel, int(voxel + 1)


CLEANED = True
SUBSET_SIZE = 500
OUTPUT_DIR = GEEDIR / "temp_pgee_exch"
FUNCTION_FILE = Path(__file__).with_name("pgee_logPoisson_dispersion_fn.py")
FUNCTION_NAME = "gee_penalty_run"
INTERACTION = False
PENALTY = True
USE_STATS_GEE_LOGISTIC = False
TOL = 1e-4
MAX_ITER = 10


_WORKER = {}


def _init_worker(subset_idx, ids, df_visits, lesions_vis1, lesions_vis2):
    _WORKER.update(
        subset_idx=subset_idx,
        ids=ids,
        df_visits=df_visits,
        lesions_vis1=lesions_vis1,
        lesions_vis2=lesions_vis2,
    )


def _fit_one(i):
    try:
        panel, voxel = _panel_for_voxel(
            i,
            _WORKER["lesions_vis1"],
            _WORKER["lesions_vis2"],
            _WORKER["subset_idx"],
            _WORKER["ids"],
            _WORKER["df_visits"],
            CLEANED,
        )
        n_visits = panel["vis"].nunique()
        n_subj = len(panel) // n_visits
        if CLEANED:
            cols = [
                np.ones(len(panel)),
                panel["age_vis2"],
                panel["age_diff"],
                panel["sexM"],
                panel["headsize"],
            ]
            if INTERACTION:
                cols.extend(
                    [
                        panel["age_vis2"] * panel["age_diff"],
                        panel["age_vis2"] * panel["sexM"],
                    ]
                )
            else:
                cols.append(panel["age_vis2"] * panel["sexM"])
        else:
            cols = [
                np.ones(len(panel)),
                panel["avg_age"],
                panel["age_diff_vis2"],
                panel["sexM"],
            ]
        X = np.column_stack([np.asarray(c, dtype=float) for c in cols])
        y = panel["y"].to_numpy(dtype=float)
        if USE_STATS_GEE_LOGISTIC:
            import statsmodels.api as sm

            model = sm.GEE(
                y,
                X,
                groups=np.repeat(np.arange(n_subj), n_visits),
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            result = model.fit(maxiter=25)
            dep = np.ravel(result.cov_struct.dep_params)
            out = {
                "beta": result.params,
                "beta_se_sandwich": result.bse,
                "alpha": float(dep[0]) if dep.size else np.nan,
                "phi": float(result.scale),
                "iterations": 0,
                "voxel": i + 1,
            }
            if INTERACTION:
                out["beta_se_sandwich_modified"] = result.bse
        else:
            fn = _load_function(FUNCTION_FILE, FUNCTION_NAME)
            model = fn(
                y=y,
                X=X,
                n_subj=n_subj,
                n_visits=n_visits,
                covariance="Exchangeable",
                tol=TOL,
                max_iter=MAX_ITER,
            )
            if isinstance(model, float) and np.isnan(model):
                return np.nan
            out = {
                "beta": np.asarray(model["beta"]),
                "beta_se_model": model["beta_se_model"],
                "beta_se_model_trace": model["beta_se_model_trace"],
                "beta_se_sandwich": model["beta_se_sandwich"],
                "alpha": model["alpha"],
                "phi": model["phi"],
                "iterations": model["iterations"],
                "voxel": i + 1,
            }
            if PENALTY and "H" in model:
                out["H"] = model["H"]
        if (i + 1) % 100 == 0:
            print(i + 1)
        return out
    except Exception as exc:
        return {"error": repr(exc), "voxel": i + 1}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run voxelwise GEE model for one subset."
    )
    parser.add_argument("n_cores", type=int)
    parser.add_argument("j", type=int)
    args = parser.parse_args(argv)
    print("Everything loaded")
    df_visits, ids = _prepare_cleaned_visits() if CLEANED else _prepare_prior_visits()
    print("Start running gee foreach fn")
    lesions_vis1, lesions_vis2 = _load_lesions(cleaned=CLEANED)
    subset_idx = _subset_indices(lesions_vis1.shape[0], SUBSET_SIZE, args.j)
    n_voxels = len(subset_idx)
    print(args.j)
    start = time.time()
    _init_worker(subset_idx, ids, df_visits, lesions_vis1, lesions_vis2)
    tasks = range(n_voxels)
    if args.n_cores > 1:
        output = [None] * n_voxels
        with ProcessPoolExecutor(
            max_workers=args.n_cores,
            initializer=_init_worker,
            initargs=(subset_idx, ids, df_visits, lesions_vis1, lesions_vis2),
        ) as ex:
            futs = {ex.submit(_fit_one, i): i for i in tasks}
            for fut in as_completed(futs):
                output[futs[fut]] = fut.result()
    else:
        output = [_fit_one(i) for i in tasks]
    print(time.time() - start)
    _save_payload({"output": output}, OUTPUT_DIR / f"GEE_subset_{args.j}.RData")


if __name__ == "__main__":
    main()
