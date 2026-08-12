"""Reusable UK Biobank input and NIfTI map loading helpers.

The UKB analysis mask is represented by one-based voxel IDs in Fortran order.
Keeping that convention in one module prevents each figure and table workflow
from implementing its own indexing and alignment checks.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_voxel_ids(path: Path, max_voxels: int | None = None) -> np.ndarray:
    """Load unique positive one-based voxel IDs from ``path``."""
    voxel_ids = np.loadtxt(path, dtype=int).reshape(-1)
    if voxel_ids.size == 0 or np.any(voxel_ids < 1):
        raise ValueError(f"Voxel IDs must be nonempty positive one-based indices: {path}")
    if np.unique(voxel_ids).size != voxel_ids.size:
        raise ValueError(f"Voxel IDs must be unique: {path}")
    if max_voxels is not None:
        if max_voxels <= 0:
            raise ValueError("max_voxels must be positive")
        voxel_ids = voxel_ids[:max_voxels]
    return voxel_ids


def load_aligned_map(
    path: Path,
    anatomical: nib.Nifti1Image,
    *,
    label: str = "map",
) -> np.ndarray:
    """Load a 3D NIfTI map after checking shape and affine alignment."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    image = nib.load(path)
    if image.shape != anatomical.shape or not np.allclose(
        image.affine, anatomical.affine
    ):
        raise ValueError(f"{label} is not aligned with anatomical image: {path}")
    data = np.asarray(image.get_fdata(), dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D {label}, got {data.shape}: {path}")
    return data


def values_at_voxels(
    data: np.ndarray, voxel_ids: np.ndarray, *, source: str = "map"
) -> np.ndarray:
    """Extract values using the UKB one-based Fortran-order voxel convention."""
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D {source}, got {data.shape}")
    if voxel_ids.size == 0:
        return np.empty(0, dtype=float)
    if voxel_ids.min() < 1 or voxel_ids.max() > data.size:
        raise ValueError(f"Voxel IDs exceed the image grid in {source}")
    return data.ravel(order="F")[voxel_ids - 1]


def load_empirical_visits(
    ukb_dir: Path, voxel_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load matching visit lesion matrices for the selected voxel count."""
    with np.load(ukb_dir / "lesions_atleast6_CVR.npz") as lesion_data:
        visit_1 = np.asarray(lesion_data["lesions_vis1"], dtype=float)
        visit_2 = np.asarray(lesion_data["lesions_vis2"], dtype=float)
    if visit_1.shape != visit_2.shape:
        raise ValueError("Visit lesion matrices must have matching shapes")
    if visit_1.shape[0] < voxel_count:
        raise ValueError("Lesion matrices contain fewer voxels than the analysis mask")
    return visit_1[:voxel_count], visit_2[:voxel_count]
