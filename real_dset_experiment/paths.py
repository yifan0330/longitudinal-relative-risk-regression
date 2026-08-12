"""Shared filesystem configuration for the real-data experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Canonical input, cache, and external-reference locations."""

    package_dir: Path
    ukb_dir: Path
    python_results_dir: Path
    anatomical: Path

    @classmethod
    def from_package_dir(cls, package_dir: Path) -> "ExperimentPaths":
        package_dir = package_dir.resolve()
        workspace_root = package_dir.parents[1]
        ukb_dir = package_dir / "UKB"
        return cls(
            package_dir=package_dir,
            ukb_dir=ukb_dir,
            python_results_dir=ukb_dir / "python_results",
            anatomical=(
                workspace_root / "CBMR" / "ALE" / "template"
                / "MNI152_T1_2mm_brain.nii.gz"
            ),
        )

    def model_result_dir(self, model: str) -> Path:
        """Return the existing cache directory name for a model."""
        return self.python_results_dir / model.replace("-", "_")


DEFAULT_PATHS = ExperimentPaths.from_package_dir(Path(__file__).parent)
DEFAULT_UKB_DIR = DEFAULT_PATHS.ukb_dir
DEFAULT_PYTHON_RESULTS_DIR = DEFAULT_PATHS.python_results_dir
DEFAULT_ANATOMICAL = DEFAULT_PATHS.anatomical
