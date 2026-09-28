"""Shared data utilities for the frozen BDG2 multi-building v2 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parent.parent))).resolve()
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
COHORT = REPORTS / "bdg2_multibuilding_cohort_manifest_v2.csv"
TEMPORAL = REPORTS / "bdg2_multibuilding_temporal_manifest_v2.csv"
MODEL_DIR = ROOT / "models" / "bdg2_multibuilding_v2"
RUN_DIR = ROOT / "outputs" / "bdg2_multibuilding_v2"
FIGURE_DIR = REPORTS / "figures" / "bdg2_multibuilding_v2"

SOURCE = "source_train_proposal"
DEVELOPMENT = "development_validation_proposal"
FINAL = "final_test_proposal"
ADAPTATION_DAYS = (0, 1, 3, 7, 14, 28)
SEEDS = (20260924, 20260925, 20260926)


def ensure_dirs() -> None:
    for path in (REPORTS, MODEL_DIR, RUN_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def iso(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat(sep=" ")


def selected_buildings() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT)
    selected = cohort.loc[cohort.selected_for_multibuilding_v2].copy()
    if len(selected) != 32:
        raise AssertionError(f"Expected 32 selected buildings, got {len(selected)}")
    if selected.alternative_group.eq(FINAL).any():
        raise AssertionError("A final-test building is selected")
    counts = selected.groupby("alternative_group").size().to_dict()
    if counts != {DEVELOPMENT: 12, SOURCE: 20}:
        raise AssertionError(f"Unexpected selected counts: {counts}")
    return selected.sort_values(["alternative_group", "site_id", "building_id"]).reset_index(drop=True)


def load_selected_power(selected: pd.DataFrame | None = None) -> pd.DataFrame:
    if selected is None:
        selected = selected_buildings()
    if selected.alternative_group.eq(FINAL).any():
        raise AssertionError("Final-test electricity access blocked")
    ids = selected.building_id.tolist()
    header = pd.read_csv(RAW, nrows=0).columns
    missing = sorted(set(ids).difference(header))
    if missing:
        raise AssertionError(f"Missing raw columns: {missing}")
    raw = pd.read_csv(RAW, usecols=["timestamp", *ids])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    if raw.timestamp.duplicated().any():
        raise AssertionError("Raw timestamps are duplicated")
    return raw


def series_for(raw: pd.DataFrame, building_id: str) -> pd.Series:
    return pd.Series(
        pd.to_numeric(raw[building_id], errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(raw.timestamp),
        name=building_id,
    )


def build_windows(
    raw: pd.DataFrame, building_id: str, target_starts: Sequence[pd.Timestamp]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    series = series_for(raw, building_id)
    xs, ys, metadata = [], [], []
    for target_start in pd.DatetimeIndex(target_starts):
        input_times = pd.date_range(target_start - pd.Timedelta(hours=24), periods=24, freq="h")
        target_times = pd.date_range(target_start, periods=24, freq="h")
        x = series.reindex(input_times).to_numpy(float)
        y = series.reindex(target_times).to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        xs.append(x)
        ys.append(y)
        metadata.append(
            {
                "building_id": building_id,
                "forecast_origin": iso(target_start),
                "input_start": iso(input_times[0]),
                "input_end": iso(input_times[-1]),
                "target_start": iso(target_times[0]),
                "target_end": iso(target_times[-1]),
            }
        )
    return (
        np.vstack(xs) if xs else np.empty((0, 24), dtype=float),
        np.vstack(ys) if ys else np.empty((0, 24), dtype=float),
        pd.DataFrame(metadata),
    )


def temporal_manifest() -> pd.DataFrame:
    return pd.read_csv(TEMPORAL)


def source_windows(raw: pd.DataFrame, building_id: str):
    row = temporal_manifest().query("building_id == @building_id and alternative_group == @SOURCE")
    if len(row) != 1:
        raise AssertionError(f"{building_id}: expected one source temporal row")
    row = row.iloc[0]
    train_starts = pd.date_range(
        pd.Timestamp(row.train_target_start),
        pd.Timestamp(row.train_target_end) - pd.Timedelta(hours=23),
        freq="h",
    )
    val_starts = pd.date_range(
        pd.Timestamp(row.validation_target_start),
        pd.Timestamp(row.validation_target_end) - pd.Timedelta(hours=23),
        freq="h",
    )
    return {
        "train": build_windows(raw, building_id, train_starts),
        "validation": build_windows(raw, building_id, val_starts),
    }


def development_row(building_id: str, days: int = 0) -> pd.Series:
    rows = temporal_manifest().query(
        "building_id == @building_id and alternative_group == @DEVELOPMENT and adaptation_days == @days"
    )
    if len(rows) != 1:
        raise AssertionError(f"{building_id}: expected one development temporal row for {days}d")
    return rows.iloc[0]


def daily_test_windows(raw: pd.DataFrame, building_id: str):
    row = development_row(building_id, 0)
    starts = pd.date_range(
        pd.Timestamp(row.test_target_start),
        pd.Timestamp(row.test_target_end) - pd.Timedelta(hours=23),
        freq="24h",
    )
    return build_windows(raw, building_id, starts)


def adaptation_windows(raw: pd.DataFrame, building_id: str, days: int):
    if days == 0:
        return np.empty((0, 24)), np.empty((0, 24)), pd.DataFrame()
    row = development_row(building_id, days)
    starts = pd.date_range(
        pd.Timestamp(row.adaptation_target_start),
        pd.Timestamp(row.adaptation_target_end) - pd.Timedelta(hours=23),
        freq="h",
    )
    return build_windows(raw, building_id, starts)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not valid.any():
        return {"n_values": 0, "mae": math.nan, "rmse": math.nan, "metric_status": "unavailable"}
    err = pred[valid] - true[valid]
    return {
        "n_values": int(valid.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "metric_status": "ok",
    }


def prediction_rows(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    seed: int | None = None,
    adaptation_days: int | None = None,
) -> pd.DataFrame:
    rows = []
    for i, meta in metadata.reset_index(drop=True).iterrows():
        target_start = pd.Timestamp(meta.target_start)
        for horizon in range(24):
            rows.append(
                {
                    "building_id": meta.building_id,
                    "model": model,
                    "seed": seed,
                    "adaptation_days": adaptation_days,
                    "forecast_origin": meta.forecast_origin,
                    "input_start": meta.input_start,
                    "input_end": meta.input_end,
                    "target_timestamp": iso(target_start + pd.Timedelta(hours=horizon)),
                    "horizon_hour": horizon + 1,
                    "y_true": float(y_true[i, horizon]),
                    "y_pred": float(y_pred[i, horizon]) if np.isfinite(y_pred[i, horizon]) else math.nan,
                    "prediction_available": bool(np.isfinite(y_pred[i, horizon])),
                    "evaluation_mode": "daily_nonoverlap",
                }
            )
    return pd.DataFrame(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
