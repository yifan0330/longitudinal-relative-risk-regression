from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

RDATA_SUFFIXES = {".rdata", ".rda", ".rds"}


def python_data_path(path: Path | str, suffix: str) -> Path:
    path = Path(path)
    if path.suffix.lower() in RDATA_SUFFIXES:
        return path.with_suffix(suffix)
    return path


def _candidate_paths(path: Path | str, suffixes: Iterable[str]) -> list[Path]:
    path = Path(path)
    if path.suffix.lower() in RDATA_SUFFIXES:
        candidates = [path.with_suffix(suffix) for suffix in suffixes]
        candidates.append(path)
    else:
        candidates = [path]
        candidates.extend(path.with_suffix(suffix) for suffix in suffixes)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def load_array_data(path: Path | str, required: Iterable[str] = ()) -> dict:
    candidates = _candidate_paths(path, (".npz", ".pkl", ".pickle"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".npz":
            with np.load(candidate, allow_pickle=False) as data:
                output = {name: data[name] for name in data.files}
        elif suffix in {".pkl", ".pickle"}:
            with candidate.open("rb") as fh:
                output = pickle.load(fh)
        elif suffix in RDATA_SUFFIXES:
            try:
                import pyreadr  # type: ignore
            except ImportError as exc:
                preferred = python_data_path(candidate, ".npz")
                raise ImportError(
                    f"{candidate} is an R data file. Convert it to {preferred} "
                    "or install pyreadr for one-off compatibility reads."
                ) from exc
            output = dict(pyreadr.read_r(str(candidate)))
        else:
            continue

        missing = [name for name in required if name not in output]
        if missing:
            raise KeyError(f"{candidate} is missing required objects: {missing}")
        return output

    raise FileNotFoundError(
        "Could not find any supported Python data file among: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_pickle_data(path: Path | str):
    candidates = _candidate_paths(path, (".pkl", ".pickle", ".RData", ".Rdata"))
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        if suffix in {".pkl", ".pickle"} or suffix in RDATA_SUFFIXES:
            try:
                with candidate.open("rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                last_error = exc
                continue
    if last_error is not None:
        raise RuntimeError(
            "Found candidate files, but none were readable as Python pickle data: "
            + ", ".join(str(candidate) for candidate in candidates if candidate.exists())
        ) from last_error
    raise FileNotFoundError(
        "Could not find any supported pickle data file among: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def save_pickle_data(obj, path: Path | str) -> Path:
    output_path = python_data_path(path, ".pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
