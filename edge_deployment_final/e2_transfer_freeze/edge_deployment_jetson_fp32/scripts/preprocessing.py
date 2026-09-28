"""NumPy-only C0 preprocessing used by the standalone deployment bundle."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np


FEATURE_ORDER = (
    "past_electricity_normalized",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_weekend",
)


def parse_naive_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace(" ", "T"))
    if timestamp.tzinfo is not None:
        raise ValueError("C0 timestamps are timezone-naive; timezone conversion is outside E1")
    return timestamp


def build_c0_tensor(
    past_electricity: np.ndarray,
    forecast_starts: Iterable[str | datetime],
    mean: float,
    population_std: float,
) -> np.ndarray:
    past = np.asarray(past_electricity, dtype=np.float64)
    starts = [parse_naive_timestamp(value) for value in forecast_starts]
    if past.ndim != 2 or past.shape[1] != 24:
        raise ValueError(f"Expected past electricity shape (n, 24), got {past.shape}")
    if len(starts) != past.shape[0]:
        raise ValueError("forecast_starts count does not match past electricity rows")
    if not np.isfinite(past).all():
        raise ValueError("past electricity contains NaN or Inf")
    if not np.isfinite(mean) or not np.isfinite(population_std) or population_std <= 0:
        raise ValueError("invalid Source normalization")

    features = np.empty((past.shape[0], 24, 6), dtype=np.float32)
    features[:, :, 0] = ((past - mean) / population_std).astype(np.float32)
    for row, start in enumerate(starts):
        target_times = [start + timedelta(hours=hour) for hour in range(24)]
        hours = np.asarray([value.hour for value in target_times], dtype=np.float64)
        weekdays = np.asarray([value.weekday() for value in target_times], dtype=np.float64)
        features[row, :, 1] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
        features[row, :, 2] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
        features[row, :, 3] = np.sin(2.0 * np.pi * weekdays / 7.0).astype(np.float32)
        features[row, :, 4] = np.cos(2.0 * np.pi * weekdays / 7.0).astype(np.float32)
        features[row, :, 5] = (weekdays >= 5).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("generated C0 tensor contains NaN or Inf")
    return features


def inverse_transform(
    normalized_output: np.ndarray, mean: float, population_std: float
) -> np.ndarray:
    values = np.asarray(normalized_output, dtype=np.float64)
    physical = values * population_std + mean
    if not np.isfinite(physical).all():
        raise ValueError("inverse transformed output contains NaN or Inf")
    return physical

