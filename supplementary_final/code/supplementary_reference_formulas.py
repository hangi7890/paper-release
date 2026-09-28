"""Reference implementations for the frozen Supplementary definitions.

These functions document calculations applied to stored final results. They do
not train a model, adapt a checkpoint, or regenerate predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def source_normalize(values: np.ndarray, source_mean: float, source_std: float) -> np.ndarray:
    if not np.isfinite(source_std) or source_std <= 0:
        raise ValueError("source_std must be finite and positive")
    return (np.asarray(values, dtype=float) - source_mean) / source_std


def supervised_window_count(adaptation_days: int) -> int:
    """Hourly 24-to-24 windows after a separate 24-hour input-history block."""
    if adaptation_days == 0:
        return 0
    if adaptation_days < 0:
        raise ValueError("adaptation_days must be nonnegative")
    return 24 * adaptation_days - 23


def effective_adaptation_batch(sample_count: int, candidate_batch: int = 64) -> int:
    if sample_count == 0:
        return 0
    return min(candidate_batch, sample_count)


def event_flags(row: pd.Series, eps: float = EPS) -> dict[str, bool]:
    f1 = bool(row.delta_mae_source > eps)
    f2 = bool(row.delta_rmse_source > eps)
    f3_pd = bool(row.gap_mae_previous_day > eps)
    f3_w = bool(row.gap_mae_weekly_same_hour > eps)
    return {
        "F1": f1,
        "F2": f2,
        "F3_PD": f3_pd,
        "F3_W": f3_w,
        "F3_any": f3_pd or f3_w,
        "F3_both": f3_pd and f3_w,
        "MAE_improved_RMSE_worsened": bool(
            row.delta_mae_source < -eps and row.delta_rmse_source > eps
        ),
    }


def f4_seed_direction_mixed(delta_mae_by_seed: np.ndarray, eps: float = EPS) -> bool:
    values = np.asarray(delta_mae_by_seed, dtype=float)
    return bool(np.any(values < -eps) and np.any(values > eps))


def site_equal_mean(
    frame: pd.DataFrame,
    value: str,
    site: str = "site_id",
    building: str = "building_id",
) -> float:
    """Equal weight to buildings within site, then equal weight to sites."""
    building_values = frame.groupby([site, building], as_index=False)[value].mean()
    site_values = building_values.groupby(site, as_index=False)[value].mean()
    return float(site_values[value].mean())


def site_cluster_percentile_ci(
    site_effects: np.ndarray,
    resamples: int = 10_000,
    seed: int = 20260925,
) -> tuple[float, float]:
    values = np.asarray(site_effects, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper)
