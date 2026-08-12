"""Map reconstruction utilities for the UKB analysis mask."""

from __future__ import annotations

import numpy as np


def values_to_map(
    values: np.ndarray, voxel_ids: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    """Place mask values into a 3D array using Fortran-order voxel IDs."""
    values = np.asarray(values, dtype=float).reshape(-1)
    voxel_ids = np.asarray(voxel_ids, dtype=int).reshape(-1)
    if values.size != voxel_ids.size:
        raise ValueError("values and voxel_ids must have the same length")
    if voxel_ids.size and (voxel_ids.min() < 1 or voxel_ids.max() > np.prod(shape)):
        raise ValueError("Voxel IDs exceed the requested map shape")
    data = np.full(shape, np.nan, dtype=float)
    if voxel_ids.size:
        coordinates = np.unravel_index(voxel_ids - 1, shape, order="F")
        data[coordinates] = values
    return data
