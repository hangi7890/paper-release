"""Shared C0 feature utilities for BDG2 stage 6 v3.

C0 uses 24 past electricity values and calendar features that are deterministic
at forecast creation time. It never reads future observed electricity or weather.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


C0_FEATURE_NAMES = (
    "past_electricity_normalized",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_weekend",
)


def target_calendar_features(metadata: pd.DataFrame) -> np.ndarray:
    """Return known calendar features for each of 24 forecast target hours."""
    rows = []
    for target_start in pd.to_datetime(metadata["target_start"]):
        target_times = pd.date_range(target_start, periods=24, freq="h")
        hour = target_times.hour.to_numpy(dtype=float)
        weekday = target_times.dayofweek.to_numpy(dtype=float)
        rows.append(
            np.column_stack(
                [
                    np.sin(2 * np.pi * hour / 24.0),
                    np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * weekday / 7.0),
                    np.cos(2 * np.pi * weekday / 7.0),
                    (weekday >= 5).astype(float),
                ]
            )
        )
    return np.stack(rows).astype(np.float32) if rows else np.empty((0, 24, 5), np.float32)


def make_c0_features(
    past_power: np.ndarray,
    metadata: pd.DataFrame,
    source_mean: float,
    source_std: float,
) -> np.ndarray:
    """Combine normalized past power with forecast-target calendar features."""
    past = np.asarray(past_power, dtype=float)
    if past.ndim != 2 or past.shape[1] != 24:
        raise AssertionError(f"Expected past power shape (n,24), got {past.shape}")
    if len(past) != len(metadata):
        raise AssertionError("Past-power rows and metadata rows differ")
    if not np.isfinite(source_mean) or not np.isfinite(source_std) or source_std <= 0:
        raise AssertionError("Invalid source normalization")
    normalized = ((past - source_mean) / source_std).astype(np.float32)[..., None]
    calendar = target_calendar_features(metadata)
    features = np.concatenate([normalized, calendar], axis=2)
    if features.shape != (len(past), 24, len(C0_FEATURE_NAMES)):
        raise AssertionError(f"Unexpected C0 shape: {features.shape}")
    if not np.isfinite(features).all():
        raise AssertionError("C0 contains non-finite values")
    return features
