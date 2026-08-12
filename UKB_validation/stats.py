"""Small statistical transformations shared by UKB outputs."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values, treating failed tests as p=1."""
    p_values = np.asarray(p_values, dtype=float)
    finite_p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    order = np.argsort(finite_p_values)
    ranked = finite_p_values[order]
    ranks = np.arange(1, ranked.size + 1)
    adjusted_ranked = np.minimum.accumulate(
        (ranked * ranked.size / ranks)[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def odds_ratio_to_relative_risk(
    odds_ratio: np.ndarray, baseline_risk: np.ndarray
) -> np.ndarray:
    """Convert odds ratios to relative risks at a given baseline risk."""
    return odds_ratio / (1.0 - baseline_risk + baseline_risk * odds_ratio)
