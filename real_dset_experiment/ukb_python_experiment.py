#!/usr/bin/env python3
"""Fresh Python UKB voxel-model fits from raw lesion and covariate inputs.

This module intentionally reads only the raw inputs under ``real_dset_experiment/UKB``.
It does not read the historical CVRanalysis model-output directories.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# Multiprocessing already parallelizes across voxel chunks.  Keep BLAS single
# threaded inside each worker to avoid massive nested-thread oversubscription.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .paths import DEFAULT_ANATOMICAL, DEFAULT_PYTHON_RESULTS_DIR, DEFAULT_UKB_DIR

MODEL_NAMES = ("rr-gee", "rr-pgee", "or-gee", "or-pgee")
COEFFICIENT_NAMES = (
    "Intercept",
    "baseAge",
    "ageDiff",
    "baseCVR",
    "CVRdiff",
    "sexM",
    "headsize",
    "ageBYageDiff",
    "ageBYsexM",
)
PREDICTORS = COEFFICIENT_NAMES[1:]

_DESIGN: "UKBDesign | None" = None
_MODEL: str | None = None
RR_GEE_CHUNK_SIZE = 512
RR_GEE_MAX_ITER = 50
RR_GEE_TOL = 1e-6
RR_GEE_MAX_STEP = 2.0


@dataclass(frozen=True)
class UKBDesign:
    X: np.ndarray
    X_clusters: np.ndarray
    lesions1: np.ndarray
    lesions2: np.ndarray
    voxel_ids: np.ndarray

    @property
    def n_subjects(self) -> int:
        return self.X_clusters.shape[0]

    @property
    def n_voxels(self) -> int:
        return self.lesions1.shape[0]

    @property
    def n_coefficients(self) -> int:
        return self.X.shape[1]


def model_result_dir(results_root: Path, model: str) -> Path:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}")
    return results_root / model.replace("-", "_")


def model_is_poisson(model: str) -> bool:
    return model.startswith("rr-")


def model_is_penalized(model: str) -> bool:
    return model.endswith("-pgee")


def default_n_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 1)))


def load_voxel_ids(path: Path) -> np.ndarray:
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0 or np.any(voxel_ids < 1):
        raise ValueError(f"Voxel IDs must be nonempty positive one-based indices: {path}")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError(f"Voxel IDs must be unique: {path}")
    return voxel_ids


def load_ukb_design(ukb_dir: Path = DEFAULT_UKB_DIR, max_voxels: int | None = None) -> UKBDesign:
    with (ukb_dir / "CVR_9June2021.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "complete_df" not in payload:
        raise KeyError(f"{ukb_dir / 'CVR_9June2021.pkl'} must contain complete_df")
    complete_df = pd.DataFrame(payload["complete_df"]).copy()
    complete_df.columns = [str(column) for column in complete_df.columns]

    required_columns = (
        "eid_8107",
        "age_vis2",
        "age_vis3",
        "X31.0.0",
        "X25000.2.0",
        "X25000.3.0",
        "CVR_vis2",
        "CVR_vis3",
    )
    missing = [column for column in required_columns if column not in complete_df.columns]
    if missing:
        raise KeyError(f"Missing UKB covariate columns: {', '.join(missing)}")

    headsize = (
        complete_df["X25000.2.0"].to_numpy(float)
        + complete_df["X25000.3.0"].to_numpy(float)
    ) / 2.0
    base_age = complete_df["age_vis2"].to_numpy(float)
    base_cvr = complete_df["CVR_vis2"].to_numpy(float)
    sex_m = complete_df["X31.0.0"].to_numpy(float)
    age_diff = (
        complete_df["age_vis3"].to_numpy(float)
        - complete_df["age_vis2"].to_numpy(float)
    )
    cvr_diff = (
        complete_df["CVR_vis3"].to_numpy(float)
        - complete_df["CVR_vis2"].to_numpy(float)
    )

    base_age = base_age - np.nanmean(base_age)
    headsize = headsize - np.nanmean(headsize)
    base_cvr = base_cvr - np.nanmean(base_cvr)

    n_subjects = len(complete_df)
    second_visit = np.array([0.0, 1.0])
    X_clusters = np.empty((n_subjects, 2, len(COEFFICIENT_NAMES)), dtype=float)
    X_clusters[:, :, 0] = 1.0
    X_clusters[:, :, 1] = base_age[:, None]
    X_clusters[:, :, 2] = age_diff[:, None] * second_visit
    X_clusters[:, :, 3] = base_cvr[:, None]
    X_clusters[:, :, 4] = cvr_diff[:, None] * second_visit
    X_clusters[:, :, 5] = sex_m[:, None]
    X_clusters[:, :, 6] = headsize[:, None]
    X_clusters[:, :, 7] = base_age[:, None] * age_diff[:, None] * second_visit
    X_clusters[:, :, 8] = base_age[:, None] * sex_m[:, None]

    with np.load(ukb_dir / "lesions_atleast6_CVR.npz") as lesion_file:
        lesions1 = np.asarray(lesion_file["lesions_vis1"], dtype=float)
        lesions2 = np.asarray(lesion_file["lesions_vis2"], dtype=float)
    if lesions1.shape != lesions2.shape:
        raise ValueError("Visit 1 and visit 2 lesion matrices must have the same shape")
    if lesions1.shape[1] != n_subjects:
        raise ValueError(
            f"Lesion matrices have {lesions1.shape[1]} subjects, but complete_df has {n_subjects}"
        )

    voxel_ids = load_voxel_ids(ukb_dir / "voxel_IDs_CVR.dat")
    if voxel_ids.size != lesions1.shape[0]:
        raise ValueError("voxel_IDs_CVR.dat does not match lesion matrix row count")

    if max_voxels is not None:
        if max_voxels <= 0:
            raise ValueError("--max-voxels must be positive when provided")
        lesions1 = lesions1[:max_voxels]
        lesions2 = lesions2[:max_voxels]
        voxel_ids = voxel_ids[:max_voxels]

    return UKBDesign(
        X=X_clusters.reshape(-1, len(COEFFICIENT_NAMES)),
        X_clusters=X_clusters,
        lesions1=lesions1,
        lesions2=lesions2,
        voxel_ids=voxel_ids,
    )


def _safe_information_inverse(information: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(information, hermitian=True)


def _information(beta: np.ndarray, X: np.ndarray, poisson: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.clip(X @ beta, -30.0, 30.0)
    if poisson:
        mu = np.exp(eta)
        weights = np.clip(mu, 1e-10, None)
    else:
        mu = expit(eta)
        weights = np.clip(mu * (1.0 - mu), 1e-10, None)
    information = X.T @ (weights[:, None] * X)
    return information, mu, weights


def _objective_and_gradient(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    poisson: bool,
    firth: bool,
) -> tuple[float, np.ndarray]:
    eta = np.clip(X @ beta, -30.0, 30.0)
    if poisson:
        mu = np.exp(eta)
        objective = float(np.sum(mu - y * eta))
        gradient = X.T @ (mu - y)
    else:
        mu = expit(eta)
        objective = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        gradient = X.T @ (mu - y)

    if not firth:
        return objective, gradient

    information, mu, weights = _information(beta, X, poisson)
    jitter = 1e-8 * np.eye(information.shape[0])
    sign, logdet = np.linalg.slogdet(information + jitter)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf, np.full_like(beta, np.nan)

    information_inverse = _safe_information_inverse(information + jitter)
    leverage = np.einsum("ij,jk,ik->i", X, information_inverse, X) * weights
    leverage = np.clip(leverage, 0.0, 1.0)
    leverage_factor = 1.0 if poisson else (1.0 - 2.0 * mu)
    firth_gradient = 0.5 * (X.T @ (leverage * leverage_factor))
    return objective - 0.5 * float(logdet), gradient - firth_gradient


def _initial_beta(y: np.ndarray, poisson: bool, p: int) -> np.ndarray:
    mean_y = float(np.mean(y))
    if poisson:
        intercept = math.log(max(mean_y, 1e-6))
    else:
        intercept = float(logit(np.clip(mean_y, 1e-6, 1.0 - 1e-6)))
    beta = np.zeros(p, dtype=float)
    beta[0] = intercept
    return beta


def _fit_one(y: np.ndarray, design: UKBDesign, model: str) -> dict[str, Any]:
    poisson = model_is_poisson(model)
    firth = model_is_penalized(model)
    if np.any(~np.isfinite(y)):
        raise ValueError("Outcome contains non-finite values")

    beta0 = _initial_beta(y, poisson, design.n_coefficients)

    def func(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = _objective_and_gradient(beta, y, design.X, poisson, firth)
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            return np.inf, np.zeros_like(beta)
        return value, gradient

    fit = minimize(func, beta0, method="L-BFGS-B", jac=True, options={"maxiter": 100})
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"L-BFGS failed with non-finite coefficients: {fit.message}")

    beta = np.asarray(fit.x, dtype=float)
    information, mu, _weights = _information(beta, design.X, poisson)
    information_inverse = _safe_information_inverse(information)
    model_se = np.sqrt(np.clip(np.diag(information_inverse), 0.0, None))

    residual_clusters = (y - mu).reshape(design.n_subjects, 2)
    cluster_scores = np.einsum("svi,sv->si", design.X_clusters, residual_clusters)
    centered_scores = cluster_scores - cluster_scores.mean(axis=0)
    correction = (design.n_subjects / (design.n_subjects - 1)) * (
        (design.n_subjects * 2 - 1)
        / (design.n_subjects * 2 - design.n_coefficients)
    )
    meat = centered_scores.T @ centered_scores * correction
    sandwich = information_inverse @ meat @ information_inverse
    sandwich_se = np.sqrt(np.clip(np.diag(sandwich), 0.0, None))

    return {
        "beta": beta,
        "se": sandwich_se,
        "model_se": model_se,
        "zscore": beta / sandwich_se,
        "converged": bool(fit.success),
        "objective": float(fit.fun),
        "iterations": int(getattr(fit, "nit", 0)),
    }


def _fit_voxel(index: int) -> dict[str, Any]:
    if _DESIGN is None or _MODEL is None:
        raise RuntimeError("Worker was not initialised")
    y = np.column_stack((_DESIGN.lesions1[index], _DESIGN.lesions2[index])).reshape(-1)
    try:
        result = _fit_one(y, _DESIGN, _MODEL)
        result["voxel_index"] = index
        return result
    except Exception as exc:
        p = _DESIGN.n_coefficients
        return {
            "voxel_index": index,
            "beta": np.full(p, np.nan),
            "se": np.full(p, np.nan),
            "model_se": np.full(p, np.nan),
            "zscore": np.full(p, np.nan),
            "converged": False,
            "objective": np.nan,
            "iterations": 0,
            "error": repr(exc),
        }


def _safe_batched_inverse(matrices: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.pinv(matrix, hermitian=True) for matrix in matrices])


def _poisson_outcome_chunk(design: UKBDesign, start: int, stop: int) -> np.ndarray:
    return np.stack(
        (design.lesions1[start:stop].T, design.lesions2[start:stop].T),
        axis=1,
    ).reshape(design.n_subjects * 2, stop - start)


def _rr_gee_information(
    X: np.ndarray, mu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    information = np.einsum("nv,ni,nj->vij", mu, X, X, optimize=True)
    scale = np.trace(information, axis1=1, axis2=2) / X.shape[1]
    jitter = np.maximum(scale, 1.0) * 1e-8
    information = information + jitter[:, None, None] * np.eye(X.shape[1])
    return information, jitter


def _fit_rr_gee_chunk(bounds: tuple[int, int]) -> tuple[int, dict[str, np.ndarray]]:
    if _DESIGN is None:
        raise RuntimeError("Worker was not initialised")
    design = _DESIGN
    start, stop = bounds
    X = design.X
    Y = _poisson_outcome_chunk(design, start, stop)
    n_voxels = stop - start
    p = design.n_coefficients

    beta = np.zeros((n_voxels, p), dtype=float)
    beta[:, 0] = np.log(np.maximum(Y.mean(axis=0), 1e-6))
    converged = np.zeros(n_voxels, dtype=bool)
    failed = np.zeros(n_voxels, dtype=bool)
    iterations = np.full(n_voxels, RR_GEE_MAX_ITER, dtype=int)

    for iteration in range(1, RR_GEE_MAX_ITER + 1):
        eta = np.clip(X @ beta.T, -30.0, 30.0)
        mu = np.exp(eta)
        score = X.T @ (Y - mu)
        information, _ = _rr_gee_information(X, mu)
        try:
            step = np.linalg.solve(information, score.T[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            inv_information = _safe_batched_inverse(information)
            step = np.einsum("vij,vj->vi", inv_information, score.T, optimize=True)

        active = ~(converged | failed)
        finite_step = np.all(np.isfinite(step), axis=1)
        failed |= active & ~finite_step
        step[~active | ~finite_step] = 0.0
        step = np.clip(step, -RR_GEE_MAX_STEP, RR_GEE_MAX_STEP)
        beta += step

        step_size = np.max(np.abs(step), axis=1)
        newly_converged = active & finite_step & (step_size < RR_GEE_TOL)
        iterations[newly_converged] = iteration
        converged |= newly_converged
        if np.all(converged | failed):
            break

    eta = np.clip(X @ beta.T, -30.0, 30.0)
    mu = np.exp(eta)
    information, _ = _rr_gee_information(X, mu)
    information_inverse = _safe_batched_inverse(information)
    residual_clusters = (Y - mu).reshape(design.n_subjects, 2, n_voxels)
    cluster_scores = np.einsum(
        "svp,svm->smp", design.X_clusters, residual_clusters, optimize=True
    )
    centered_scores = cluster_scores - cluster_scores.mean(axis=0, keepdims=True)
    correction = (design.n_subjects / (design.n_subjects - 1)) * (
        (design.n_subjects * 2 - 1)
        / (design.n_subjects * 2 - design.n_coefficients)
    )
    meat = np.einsum("smp,smq->mpq", centered_scores, centered_scores, optimize=True)
    meat *= correction
    sandwich = np.einsum(
        "mij,mjk,mkl->mil",
        information_inverse,
        meat,
        information_inverse,
        optimize=True,
    )
    se = np.sqrt(np.clip(np.diagonal(sandwich, axis1=1, axis2=2), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = beta / se

    bad = failed | ~np.all(np.isfinite(beta), axis=1)
    failed |= bad
    beta[bad] = np.nan
    se[bad] = np.nan
    zscore[bad] = np.nan
    return start, {
        "beta": beta,
        "se": se,
        "zscore": zscore,
        "converged": converged & ~failed,
        "iterations": iterations,
        "failed": failed,
    }


def _fit_rr_gee_fast(
    design: UKBDesign, n_jobs: int, chunk_size: int = RR_GEE_CHUNK_SIZE
) -> dict[str, np.ndarray]:
    chunk_size = max(1, int(chunk_size))
    chunks = [
        (start, min(start + chunk_size, design.n_voxels))
        for start in range(0, design.n_voxels, chunk_size)
    ]
    arrays = {
        "beta": np.full((design.n_voxels, design.n_coefficients), np.nan),
        "se": np.full((design.n_voxels, design.n_coefficients), np.nan),
        "zscore": np.full((design.n_voxels, design.n_coefficients), np.nan),
        "converged": np.zeros(design.n_voxels, dtype=bool),
        "iterations": np.zeros(design.n_voxels, dtype=int),
        "failed": np.ones(design.n_voxels, dtype=bool),
    }

    if n_jobs == 1:
        _init_worker(design, "rr-gee")
        chunk_results = [_fit_rr_gee_chunk(chunk) for chunk in chunks]
    else:
        with Pool(n_jobs, initializer=_init_worker, initargs=(design, "rr-gee")) as pool:
            chunk_results = pool.map(_fit_rr_gee_chunk, chunks, chunksize=1)

    for start, chunk_arrays in chunk_results:
        stop = start + chunk_arrays["beta"].shape[0]
        for key, values in chunk_arrays.items():
            arrays[key][start:stop] = values
    return arrays


def _init_worker(design: UKBDesign, model: str) -> None:
    global _DESIGN, _MODEL
    _DESIGN = design
    _MODEL = model


def fit_model(
    model: str,
    *,
    ukb_dir: Path = DEFAULT_UKB_DIR,
    max_voxels: int | None = None,
    n_jobs: int | None = None,
    chunk_size: int = RR_GEE_CHUNK_SIZE,
) -> tuple[UKBDesign, dict[str, np.ndarray]]:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}")
    design = load_ukb_design(ukb_dir, max_voxels=max_voxels)
    n_jobs = default_n_jobs() if n_jobs is None else max(1, int(n_jobs))
    if model == "rr-gee":
        return design, _fit_rr_gee_fast(design, n_jobs, chunk_size=chunk_size)

    if n_jobs == 1:
        _init_worker(design, model)
        results = [_fit_voxel(index) for index in range(design.n_voxels)]
    else:
        chunksize = max(1, design.n_voxels // (n_jobs * 20))
        with Pool(n_jobs, initializer=_init_worker, initargs=(design, model)) as pool:
            results = pool.map(_fit_voxel, range(design.n_voxels), chunksize=chunksize)

    beta = np.vstack([item["beta"] for item in results])
    se = np.vstack([item["se"] for item in results])
    zscore = np.vstack([item["zscore"] for item in results])
    converged = np.asarray([item["converged"] for item in results], dtype=bool)
    iterations = np.asarray([item["iterations"] for item in results], dtype=int)
    failed = np.asarray(["error" in item for item in results], dtype=bool)
    return design, {
        "beta": beta,
        "se": se,
        "zscore": zscore,
        "converged": converged,
        "iterations": iterations,
        "failed": failed,
    }


def _write_nifti_map(
    values: np.ndarray,
    voxel_ids: np.ndarray,
    template: nib.Nifti1Image,
    output: Path,
) -> None:
    if voxel_ids.max() > np.prod(template.shape):
        raise ValueError("Voxel IDs exceed the anatomical image grid")
    data = np.full(template.shape, np.nan, dtype=np.float32)
    coordinates = np.unravel_index(voxel_ids - 1, template.shape, order="F")
    data[coordinates] = np.asarray(values, dtype=float)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, template.affine, template.header), str(output))


def write_model_outputs(
    model: str,
    output_dir: Path,
    design: UKBDesign,
    arrays: dict[str, np.ndarray],
    anatomical: Path = DEFAULT_ANATOMICAL,
) -> None:
    template = nib.load(str(anatomical))
    output_dir.mkdir(parents=True, exist_ok=True)

    for coefficient_index, coefficient_name in enumerate(COEFFICIENT_NAMES):
        for prefix, key in (("estimate", "beta"), ("se", "se"), ("zscore", "zscore")):
            _write_nifti_map(
                arrays[key][:, coefficient_index],
                design.voxel_ids,
                template,
                output_dir / f"{prefix}_{coefficient_name}_GEE.nii.gz",
            )

    np.savez_compressed(
        output_dir / "fit_summary.npz",
        converged=arrays["converged"],
        iterations=arrays["iterations"],
        failed=arrays["failed"],
        voxel_ids=design.voxel_ids,
        coefficient_names=np.asarray(COEFFICIENT_NAMES, dtype=object),
        model=np.asarray(model),
    )


def required_zscore_maps(result_dir: Path) -> list[Path]:
    return [result_dir / f"zscore_{predictor}_GEE.nii.gz" for predictor in PREDICTORS]


def ensure_model_outputs(
    model: str,
    *,
    ukb_dir: Path = DEFAULT_UKB_DIR,
    results_root: Path = DEFAULT_PYTHON_RESULTS_DIR,
    result_dir: Path | None = None,
    anatomical: Path = DEFAULT_ANATOMICAL,
    n_jobs: int | None = None,
    max_voxels: int | None = None,
    chunk_size: int = RR_GEE_CHUNK_SIZE,
    force_rerun: bool = True,
) -> Path:
    output_dir = result_dir or model_result_dir(results_root, model)
    maps_exist = all(path.is_file() for path in required_zscore_maps(output_dir))
    if maps_exist and not force_rerun:
        return output_dir

    optimizer = "chunked vectorized IRLS/Newton" if model == "rr-gee" else "L-BFGS"
    print(
        f"Fitting {model.upper()} from raw UKB inputs with {optimizer} "
        f"({default_n_jobs() if n_jobs is None else n_jobs} worker(s), "
        "BLAS threads capped at 1)"
    )
    design, arrays = fit_model(
        model,
        ukb_dir=ukb_dir,
        max_voxels=max_voxels,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    write_model_outputs(model, output_dir, design, arrays, anatomical=anatomical)
    finite_z = int(np.count_nonzero(np.isfinite(arrays["zscore"][:, 1:])))
    total_z = arrays["zscore"][:, 1:].size
    print(
        f"Saved {model.upper()} Python results to {output_dir}; "
        f"finite predictor z-scores: {finite_z:,}/{total_z:,}"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun UKB voxel experiments from raw inputs using Python L-BFGS fits."
    )
    parser.add_argument("--ukb-dir", type=Path, default=DEFAULT_UKB_DIR)
    parser.add_argument("--python-results-dir", type=Path, default=DEFAULT_PYTHON_RESULTS_DIR)
    parser.add_argument("--anatomical", type=Path, default=DEFAULT_ANATOMICAL)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*MODEL_NAMES, "all"),
        default=["all"],
        help="Models to rerun; 'all' expands to RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE.",
    )
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs())
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=RR_GEE_CHUNK_SIZE,
        help="RR-GEE voxel chunk size for the vectorized IRLS/Newton solver.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing UKB/python_results maps instead of rerunning fits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.max_voxels is not None and args.max_voxels <= 0:
        raise ValueError("--max-voxels must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    for label, path in (
        ("UKB input directory", args.ukb_dir),
        ("anatomical image", args.anatomical),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    models = MODEL_NAMES if "all" in args.models else tuple(args.models)
    for model in models:
        ensure_model_outputs(
            model,
            ukb_dir=args.ukb_dir,
            results_root=args.python_results_dir,
            anatomical=args.anatomical,
            n_jobs=args.n_jobs,
            max_voxels=args.max_voxels,
            chunk_size=args.chunk_size,
            force_rerun=not args.use_cache,
        )


if __name__ == "__main__":
    main()
