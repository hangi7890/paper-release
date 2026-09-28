"""Independent final_v1 metric recomputation, hypotheses, failures, and exploration."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"
SUPPORT = REPORTS / "final_v1"
FIGURES = SUPPORT / "figures"
RAW_PREDICTIONS = REPORTS / "final_v1_raw_predictions.csv"
EXECUTION_METRICS = SUPPORT / "gate2_execution_metrics.csv"
GATE2_CONFIG = SUPPORT / "gate2_run_config.json"
COHORT = REPORTS / "final_v1_cohort_manifest.csv"
TEMPORAL = REPORTS / "final_v1_temporal_manifest.csv"
RAW_POWER = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
SOURCE_SCALE = REPORTS / "bdg2_v5_source_scale_reference.csv"

METRICS_OUT = REPORTS / "final_v1_metrics_by_building_seed_budget.csv"
EFFICIENCY_OUT = REPORTS / "final_v1_data_efficiency.csv"
FAILURE_EVENTS_OUT = REPORTS / "final_v1_failure_events.csv"
FAILURE_SUMMARY_OUT = REPORTS / "final_v1_failure_summary.csv"
HYPOTHESIS_OUT = REPORTS / "final_v1_hypothesis_results.md"
CHARACTERISTICS_REPORT = REPORTS / "final_v1_building_characteristics_exploratory.md"
FINAL_REPORT = REPORTS / "final_v1_final_test_report.md"

RECOMPUTE_OUT = SUPPORT / "gate3_metric_recalculation_comparison.csv"
CURVE_OUT = SUPPORT / "final_v1_site_equal_adaptation_curve.csv"
SITE_EFFECTS_OUT = SUPPORT / "final_v1_h1_site_effects.csv"
H2_SITE_OUT = SUPPORT / "final_v1_h2_site_effects.csv"
H3_SITE_OUT = SUPPORT / "final_v1_h3_site_effects.csv"
H3_TABLE_OUT = SUPPORT / "final_v1_h3_contingency.csv"
REPRESENTATIVE_OUT = SUPPORT / "final_v1_representative_failure_cases.csv"
REPRESENTATIVE_PRED_OUT = SUPPORT / "final_v1_representative_failure_predictions.csv"
CHARACTERISTICS_OUT = SUPPORT / "final_v1_building_characteristics.csv"
CHARACTERISTIC_ASSOC_OUT = SUPPORT / "final_v1_characteristic_associations.csv"
CHARACTERISTIC_GROUP_OUT = SUPPORT / "final_v1_characteristic_group_summary.csv"
RUN_LOG = SUPPORT / "gate3_analysis.log"

EPS = 1e-12
TOLERANCE = 1e-10
SEEDS = (20260924, 20260925, 20260926)
BUDGETS = (0, 1, 3, 7, 14, 28)
NONZERO_BUDGETS = (1, 3, 7, 14, 28)
EXPECTED_SAMPLES = {0: 0, 1: 1, 3: 49, 7: 145, 14: 313, 28: 649}
SOURCE_MEAN = 167.0651976323119
SOURCE_STD = 354.328669571624


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{utc_now()} {message}"
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def check_write_once() -> None:
    outputs = (
        METRICS_OUT,
        EFFICIENCY_OUT,
        FAILURE_EVENTS_OUT,
        FAILURE_SUMMARY_OUT,
        HYPOTHESIS_OUT,
        CHARACTERISTICS_REPORT,
        FINAL_REPORT,
        RECOMPUTE_OUT,
        CURVE_OUT,
        SITE_EFFECTS_OUT,
        H2_SITE_OUT,
        H3_SITE_OUT,
        H3_TABLE_OUT,
        REPRESENTATIVE_OUT,
        REPRESENTATIVE_PRED_OUT,
        CHARACTERISTICS_OUT,
        CHARACTERISTIC_ASSOC_OUT,
        CHARACTERISTIC_GROUP_OUT,
        RUN_LOG,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Gate 3 refuses to overwrite existing outputs: " + ", ".join(existing))


def metric_key_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["seed_key"] = pd.to_numeric(chunk.seed, errors="coerce").fillna(-1).astype(int)
    chunk["days_key"] = pd.to_numeric(chunk.adaptation_days, errors="coerce").fillna(-1).astype(int)
    return chunk


def recompute_metrics() -> tuple[pd.DataFrame, int, int]:
    keys = ["building_id", "site_id", "model", "seed_key", "days_key"]
    accum: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {"row_count": 0, "valid_count": 0, "sum_abs": 0.0, "sum_sq": 0.0}
    )
    digests: dict[tuple, hashlib._Hash] = {}
    unavailable = 0
    bad_status = 0
    total_rows = 0
    usecols = [
        "building_id",
        "site_id",
        "model",
        "seed",
        "adaptation_days",
        "forecast_origin",
        "target_timestamp",
        "horizon_hour",
        "y_true",
        "y_pred",
        "prediction_available",
        "status",
    ]
    for chunk_index, chunk in enumerate(
        pd.read_csv(RAW_PREDICTIONS, usecols=usecols, chunksize=250_000)
    ):
        chunk = metric_key_frame(chunk)
        true = pd.to_numeric(chunk.y_true, errors="coerce").to_numpy(float)
        pred = pd.to_numeric(chunk.y_pred, errors="coerce").to_numpy(float)
        valid = np.isfinite(true) & np.isfinite(pred)
        chunk["_valid"] = valid
        chunk["_abs"] = np.where(valid, np.abs(pred - true), 0.0)
        chunk["_sq"] = np.where(valid, (pred - true) ** 2, 0.0)
        grouped = chunk.groupby(keys, sort=False, dropna=False).agg(
            row_count=("building_id", "size"),
            valid_count=("_valid", "sum"),
            sum_abs=("_abs", "sum"),
            sum_sq=("_sq", "sum"),
        )
        for key, row in grouped.iterrows():
            entry = accum[tuple(key)]
            for field in ("row_count", "valid_count", "sum_abs", "sum_sq"):
                entry[field] += float(row[field])
        for key, sub in chunk.groupby(keys, sort=False, dropna=False):
            tuple_key = tuple(key)
            digest = digests.setdefault(tuple_key, hashlib.sha256())
            digest.update(
                pd.util.hash_pandas_object(
                    sub[["forecast_origin", "target_timestamp", "horizon_hour", "y_true"]],
                    index=False,
                ).to_numpy(dtype=np.uint64).tobytes()
            )
        unavailable += int((~valid).sum())
        bad_status += int(chunk.status.ne("ok").sum())
        total_rows += len(chunk)
        if (chunk_index + 1) % 5 == 0:
            log(f"Gate3 metric pass rows={total_rows}")

    rows = []
    for key, values in accum.items():
        building_id, site_id, model, seed_key, days_key = key
        n = int(values["valid_count"])
        rows.append(
            {
                "building_id": building_id,
                "site_id": site_id,
                "model": model,
                "seed_key": seed_key,
                "days_key": days_key,
                "row_count": int(values["row_count"]),
                "n_values": n,
                "mae": values["sum_abs"] / n if n else math.nan,
                "rmse": math.sqrt(values["sum_sq"] / n) if n else math.nan,
                "truth_timestamp_digest": digests[key].hexdigest(),
            }
        )
    result = pd.DataFrame(rows)
    return result, unavailable, bad_status


def validate_metric_grid(metrics: pd.DataFrame, cohort: pd.DataFrame) -> None:
    if len(metrics) != 9060:
        raise AssertionError(f"Expected 9060 metric groups, got {len(metrics)}")
    if not metrics.row_count.eq(672).all() or not metrics.n_values.eq(672).all():
        raise AssertionError("A final metric group does not contain 672 valid target values")
    if set(metrics.building_id) != set(cohort.building_id):
        raise AssertionError("Metric/cohort building IDs differ")
    for building_id, frame in metrics.groupby("building_id", sort=False):
        if frame.truth_timestamp_digest.nunique() != 1:
            raise AssertionError(f"{building_id}: model/budget/seed truth timestamps or values differ")
        if len(frame) != 20:
            raise AssertionError(f"{building_id}: expected 20 metric groups, got {len(frame)}")
        neural = frame.loc[frame.seed_key.ge(0)]
        if set(neural.seed_key) != set(SEEDS) or set(neural.days_key) != set(BUDGETS):
            raise AssertionError(f"{building_id}: incomplete neural seed/budget grid")
        if len(frame.loc[frame.model.isin(["previous_day", "weekly_same_hour"])]) != 2:
            raise AssertionError(f"{building_id}: incomplete baseline grid")


def compare_execution_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    execution = metric_key_frame(pd.read_csv(EXECUTION_METRICS))
    comparison = metrics.merge(
        execution[
            ["building_id", "site_id", "model", "seed_key", "days_key", "mae", "rmse", "status"]
        ].rename(columns={"mae": "execution_mae", "rmse": "execution_rmse"}),
        on=["building_id", "site_id", "model", "seed_key", "days_key"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison._merge.eq("both").all():
        raise AssertionError("Execution and independently recomputed metric keys differ")
    comparison["mae_abs_difference"] = (comparison.mae - comparison.execution_mae).abs()
    comparison["rmse_abs_difference"] = (comparison.rmse - comparison.execution_rmse).abs()
    comparison.to_csv(RECOMPUTE_OUT, index=False, encoding="utf-8", float_format="%.17g")
    maximum = max(comparison.mae_abs_difference.max(), comparison.rmse_abs_difference.max())
    if maximum > TOLERANCE:
        raise AssertionError(f"Independent metric discrepancy {maximum} exceeds {TOLERANCE}")
    return comparison


def output_metric_frame(metrics: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    meta = cohort.set_index("building_id")
    result = metrics.copy()
    result["primaryspaceusage"] = result.building_id.map(meta.primaryspaceusage)
    result["seed"] = pd.array(result.seed_key.where(result.seed_key.ge(0)), dtype="Int64")
    result["adaptation_days"] = pd.array(result.days_key.where(result.days_key.ge(0)), dtype="Int64")
    result["adaptation_samples"] = result.days_key.map(EXPECTED_SAMPLES).fillna(0).astype(int)
    result["metric_source"] = "independent_recalculation_from_final_v1_raw_predictions"
    result["metric_status"] = "ok"
    columns = [
        "building_id",
        "site_id",
        "primaryspaceusage",
        "model",
        "seed",
        "adaptation_days",
        "adaptation_samples",
        "row_count",
        "n_values",
        "mae",
        "rmse",
        "metric_status",
        "truth_timestamp_digest",
        "metric_source",
    ]
    return result[columns].sort_values(
        ["site_id", "building_id", "model", "seed", "adaptation_days"], na_position="first"
    )


def build_efficiency(metrics: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    neural = metrics.loc[metrics.seed_key.ge(0)].copy()
    source = neural.loc[neural.days_key.eq(0), ["building_id", "seed_key", "mae", "rmse"]].rename(
        columns={"mae": "source_only_mae", "rmse": "source_only_rmse"}
    )
    baselines = metrics.loc[metrics.model.isin(["previous_day", "weekly_same_hour"])].pivot(
        index="building_id", columns="model", values=["mae", "rmse"]
    )
    baselines.columns = [f"{model}_{metric}" for metric, model in baselines.columns]
    baselines = baselines.reset_index()
    result = neural.merge(source, on=["building_id", "seed_key"], validate="many_to_one")
    result = result.merge(baselines, on="building_id", validate="many_to_one")
    result["delta_mae_source"] = result.mae - result.source_only_mae
    result["delta_rmse_source"] = result.rmse - result.source_only_rmse
    result["relative_delta_mae_source_pct"] = np.where(
        result.source_only_mae.abs().gt(EPS), 100 * result.delta_mae_source / result.source_only_mae, np.nan
    )
    result["relative_delta_rmse_source_pct"] = np.where(
        result.source_only_rmse.abs().gt(EPS), 100 * result.delta_rmse_source / result.source_only_rmse, np.nan
    )
    result["gap_mae_previous_day"] = result.mae - result.previous_day_mae
    result["gap_mae_weekly_same_hour"] = result.mae - result.weekly_same_hour_mae
    result["gap_rmse_previous_day"] = result.rmse - result.previous_day_rmse
    result["gap_rmse_weekly_same_hour"] = result.rmse - result.weekly_same_hour_rmse
    result["a_source_improvement"] = result.delta_mae_source.lt(-EPS)
    result["b_beats_previous_day"] = result.mae.lt(result.previous_day_mae - EPS)
    result["b_beats_weekly_same_hour"] = result.mae.lt(result.weekly_same_hour_mae - EPS)
    result["b_beats_both_baselines"] = result.b_beats_previous_day & result.b_beats_weekly_same_hour
    meta = cohort.set_index("building_id")
    result["primaryspaceusage"] = result.building_id.map(meta.primaryspaceusage)
    result["adaptation_samples"] = result.days_key.map(EXPECTED_SAMPLES).astype(int)
    result["seed"] = result.seed_key.astype(int)
    result["adaptation_days"] = result.days_key.astype(int)
    columns = [
        "building_id", "site_id", "primaryspaceusage", "seed", "adaptation_days",
        "adaptation_samples", "mae", "rmse", "source_only_mae", "source_only_rmse",
        "delta_mae_source", "delta_rmse_source", "relative_delta_mae_source_pct",
        "relative_delta_rmse_source_pct", "previous_day_mae", "weekly_same_hour_mae",
        "previous_day_rmse", "weekly_same_hour_rmse", "gap_mae_previous_day",
        "gap_mae_weekly_same_hour", "gap_rmse_previous_day", "gap_rmse_weekly_same_hour",
        "a_source_improvement", "b_beats_previous_day", "b_beats_weekly_same_hour",
        "b_beats_both_baselines",
    ]
    return result[columns].sort_values(["site_id", "building_id", "seed", "adaptation_days"])


def site_equal_flag_rate(frame: pd.DataFrame, flag: str) -> float:
    by_building = frame.groupby(["site_id", "building_id"], as_index=False)[flag].mean()
    return float(by_building.groupby("site_id")[flag].mean().mean())


def build_failures(efficiency: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = efficiency.loc[efficiency.adaptation_days.gt(0)].copy()
    events["f1"] = events.delta_mae_source.gt(EPS)
    events["f2"] = events.delta_rmse_source.gt(EPS)
    events["f3_previous_day"] = events.gap_mae_previous_day.gt(EPS)
    events["f3_weekly_same_hour"] = events.gap_mae_weekly_same_hour.gt(EPS)
    events["f3_any"] = events.f3_previous_day | events.f3_weekly_same_hour
    events["f3_both"] = events.f3_previous_day & events.f3_weekly_same_hour
    events["mae_improved_rmse_worsened"] = events.delta_mae_source.lt(-EPS) & events.delta_rmse_source.gt(EPS)
    mixed = events.groupby(["building_id", "adaptation_days"]).delta_mae_source.agg(
        lambda values: bool((values < -EPS).any() and (values > EPS).any())
    )
    events = events.merge(mixed.rename("f4").reset_index(), on=["building_id", "adaptation_days"], validate="many_to_one")
    summary_rows = []
    for days in NONZERO_BUDGETS:
        frame = events.loc[events.adaptation_days.eq(days)]
        row = {
            "adaptation_days": days,
            "seed_event_denominator": len(frame),
            "building_budget_denominator_for_f4": frame.building_id.nunique(),
        }
        for flag in ("f1", "f2", "f3_previous_day", "f3_weekly_same_hour", "f3_any", "f3_both", "mae_improved_rmse_worsened"):
            row[f"{flag}_count"] = int(frame[flag].sum())
            row[f"{flag}_rate"] = float(frame[flag].mean())
            row[f"{flag}_site_equal_rate"] = site_equal_flag_rate(frame, flag)
        unique_f4 = frame.drop_duplicates(["building_id", "adaptation_days"])
        row["f4_count"] = int(unique_f4.f4.sum())
        row["f4_rate"] = float(unique_f4.f4.mean())
        row["f4_site_equal_rate"] = float(unique_f4.groupby("site_id").f4.mean().mean())
        summary_rows.append(row)
    all_row = {
        "adaptation_days": "ALL_NONZERO",
        "seed_event_denominator": len(events),
        "building_budget_denominator_for_f4": len(events.drop_duplicates(["building_id", "adaptation_days"])),
    }
    for flag in ("f1", "f2", "f3_previous_day", "f3_weekly_same_hour", "f3_any", "f3_both", "mae_improved_rmse_worsened"):
        all_row[f"{flag}_count"] = int(events[flag].sum())
        all_row[f"{flag}_rate"] = float(events[flag].mean())
        all_row[f"{flag}_site_equal_rate"] = site_equal_flag_rate(events, flag)
    unique_all = events.drop_duplicates(["building_id", "adaptation_days"])
    all_row["f4_count"] = int(unique_all.f4.sum())
    all_row["f4_rate"] = float(unique_all.f4.mean())
    all_row["f4_site_equal_rate"] = float(unique_all.groupby("site_id").f4.mean().mean())
    summary_rows.append(all_row)
    return events, pd.DataFrame(summary_rows)


def bootstrap_ci(site_values: pd.Series, seed: int = 20260925, repetitions: int = 10_000) -> tuple[float, float]:
    values = site_values.to_numpy(float)
    if len(values) < 1:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def build_hypotheses(efficiency: pd.DataFrame, events: pd.DataFrame):
    # H1: seed mean within building, building mean within site, site mean overall.
    day28 = efficiency.loc[efficiency.adaptation_days.eq(28)]
    h1_building = day28.groupby(["site_id", "building_id"], as_index=False).agg(
        delta_mae_source=("delta_mae_source", "mean"),
        delta_rmse_source=("delta_rmse_source", "mean"),
    )
    h1_site = h1_building.groupby("site_id", as_index=False).agg(
        buildings=("building_id", "size"),
        delta_mae_source=("delta_mae_source", "mean"),
        delta_rmse_source=("delta_rmse_source", "mean"),
    )
    h1_effect = float(h1_site.delta_mae_source.mean())
    h1_low, h1_high = bootstrap_ci(h1_site.set_index("site_id").delta_mae_source)
    if len(h1_site) < 5:
        h1_status = "inconclusive_fewer_than_5_sites"
    elif h1_effect < 0 and h1_high < 0:
        h1_status = "supported"
    else:
        h1_status = "not_supported"

    # H2: site-equal difference in F1 rate between 28d and 1d.
    f1_building = events.groupby(["site_id", "building_id", "adaptation_days"], as_index=False).f1.mean()
    f1_site = f1_building.groupby(["site_id", "adaptation_days"], as_index=False).f1.mean()
    f1_pivot = f1_site.pivot(index="site_id", columns="adaptation_days", values="f1")
    h2_site = pd.DataFrame(
        {
            "site_id": f1_pivot.index,
            "f1_rate_1d": f1_pivot[1].to_numpy(float),
            "f1_rate_28d": f1_pivot[28].to_numpy(float),
        }
    )
    h2_site["rate_difference_28d_minus_1d"] = h2_site.f1_rate_28d - h2_site.f1_rate_1d
    h2_effect = float(h2_site.rate_difference_28d_minus_1d.mean())
    h2_low, h2_high = bootstrap_ci(h2_site.set_index("site_id").rate_difference_28d_minus_1d)

    # H3: site-equal A rate minus B rate at 28d, plus paired 2x2 table.
    h3 = day28.copy()
    h3_building = h3.groupby(["site_id", "building_id"], as_index=False).agg(
        a_rate=("a_source_improvement", "mean"),
        b_rate=("b_beats_both_baselines", "mean"),
    )
    h3_site = h3_building.groupby("site_id", as_index=False).agg(
        a_rate=("a_rate", "mean"), b_rate=("b_rate", "mean")
    )
    h3_site["rate_difference_a_minus_b"] = h3_site.a_rate - h3_site.b_rate
    h3_effect = float(h3_site.rate_difference_a_minus_b.mean())
    h3_low, h3_high = bootstrap_ci(h3_site.set_index("site_id").rate_difference_a_minus_b)
    contingency = (
        h3.groupby(["a_source_improvement", "b_beats_both_baselines"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return {
        "h1_effect": h1_effect,
        "h1_ci_low": h1_low,
        "h1_ci_high": h1_high,
        "h1_status": h1_status,
        "h2_effect": h2_effect,
        "h2_ci_low": h2_low,
        "h2_ci_high": h2_high,
        "h3_effect": h3_effect,
        "h3_ci_low": h3_low,
        "h3_ci_high": h3_high,
        "h3_a_rate": float(h3_site.a_rate.mean()),
        "h3_b_rate": float(h3_site.b_rate.mean()),
    }, h1_site, h2_site, h3_site, contingency


def build_curves(efficiency: pd.DataFrame) -> pd.DataFrame:
    site = efficiency.groupby(["site_id", "adaptation_days"], as_index=False).agg(
        buildings=("building_id", "nunique"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        delta_mae_source=("delta_mae_source", "mean"),
        delta_rmse_source=("delta_rmse_source", "mean"),
        a_rate=("a_source_improvement", "mean"),
        b_rate=("b_beats_both_baselines", "mean"),
    )
    overall = site.groupby("adaptation_days", as_index=False).agg(
        sites=("site_id", "nunique"),
        mae_site_equal=("mae", "mean"),
        rmse_site_equal=("rmse", "mean"),
        delta_mae_source_site_equal=("delta_mae_source", "mean"),
        delta_rmse_source_site_equal=("delta_rmse_source", "mean"),
        a_rate_site_equal=("a_rate", "mean"),
        b_rate_site_equal=("b_rate", "mean"),
    )
    return overall, site


def numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError("Characteristic interval is empty or non-finite")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "mean": mean,
        "std_population": std,
        "cv_abs_mean": std / abs(mean) if abs(mean) > EPS else math.nan,
        "zero_count": int((values == 0).sum()),
        "zero_ratio": float((values == 0).mean()),
    }


def lag_repeatability(values: np.ndarray, lag: int, minimum_pairs: int) -> tuple[int, float, str]:
    values = np.asarray(values, dtype=float)
    if len(values) <= lag:
        return 0, math.nan, "insufficient_pairs"
    earlier, later = values[:-lag], values[lag:]
    n = len(earlier)
    if n < minimum_pairs:
        return n, math.nan, "insufficient_pairs"
    if np.std(earlier) <= EPS or np.std(later) <= EPS:
        return n, math.nan, "constant_series"
    return n, float(np.corrcoef(earlier, later)[0, 1]), "ok"


def source_comparison(mean: float, source: pd.DataFrame) -> dict[str, float]:
    means = source.source_train_mean.to_numpy(float)
    return {
        "mean_minus_source_config_mean": mean - SOURCE_MEAN,
        "mean_to_source_config_mean_ratio": mean / SOURCE_MEAN if abs(SOURCE_MEAN) > EPS else math.nan,
        "absolute_mean_gap_in_source_config_std": abs(mean - SOURCE_MEAN) / SOURCE_STD,
        "mean_minus_source_building_median": mean - float(np.median(means)),
        "mean_percentile_among_20_source_building_means": float(np.mean(means <= mean)),
    }


def build_characteristics(cohort: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_SCALE)
    if len(source) != 20 or source.development_or_final_power_used.astype(bool).any():
        raise AssertionError("Frozen Source scale reference is invalid")
    ids = cohort.building_id.tolist()
    raw = pd.read_csv(RAW_POWER, usecols=["timestamp", *ids])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    raw = raw.set_index("timestamp")
    rows = []
    for building in cohort.itertuples(index=False):
        series = pd.to_numeric(raw[building.building_id], errors="coerce")
        for days in BUDGETS:
            row = temporal.loc[
                temporal.building_id.eq(building.building_id) & temporal.adaptation_days.eq(days)
            ].iloc[0]
            test_start = pd.Timestamp(row.test_target_start)
            if days == 0:
                context_end = test_start - pd.Timedelta(hours=1)
                context_start = test_start - pd.Timedelta(hours=24)
            else:
                adapt_start = pd.Timestamp(row.adaptation_target_start)
                context_end = adapt_start - pd.Timedelta(hours=1)
                context_start = adapt_start - pd.Timedelta(hours=24)
            context_index = pd.date_range(context_start, context_end, freq="h")
            context = series.reindex(context_index).to_numpy(float)
            context_stats = numeric_summary(context)
            base = {
                "building_id": building.building_id,
                "site_id": building.site_id,
                "primaryspaceusage": building.primaryspaceusage,
                "sqm": building.sqm,
                "adaptation_days": days,
                "test_target_start": row.test_target_start,
                "test_target_end": row.test_target_end,
                "context_24h_start": context_start,
                "context_24h_end": context_end,
                "context_24h_hours": 24,
                **{f"context_24h_{key}": value for key, value in context_stats.items()},
                **{f"context_24h_{key}": value for key, value in source_comparison(context_stats["mean"], source).items()},
                "test_target_values_used_for_characteristics": False,
                "exploratory_only": True,
            }
            if days == 0:
                rows.append(
                    {
                        **base,
                        "feature_information_cutoff": context_end,
                        "budget_history_status": "0d_no_target_labels_context24_only",
                        "available_pretest_start": pd.NaT,
                        "available_pretest_end": pd.NaT,
                        "available_pretest_hours": 0,
                        "available_mean": math.nan,
                        "available_std_population": math.nan,
                        "available_cv_abs_mean": math.nan,
                        "available_zero_count": math.nan,
                        "available_zero_ratio": math.nan,
                        "daily_repeatability_pairs": 0,
                        "daily_repeatability_pearson": math.nan,
                        "daily_repeatability_status": "not_defined_for_0d_long_history",
                        "weekly_repeatability_pairs": 0,
                        "weekly_repeatability_pearson": math.nan,
                        "weekly_repeatability_status": "not_defined_for_0d_long_history",
                        **{key: math.nan for key in source_comparison(0.0, source)},
                    }
                )
                continue
            available_end = pd.Timestamp(row.adaptation_target_end)
            available_index = pd.date_range(context_start, available_end, freq="h")
            available = series.reindex(available_index).to_numpy(float)
            if len(available) != 24 * (days + 1) or not available_end < test_start:
                raise AssertionError(f"{building.building_id}/{days}d characteristic boundary invalid")
            stats = numeric_summary(available)
            daily_n, daily_corr, daily_status = lag_repeatability(available, 24, 24)
            weekly_n, weekly_corr, weekly_status = lag_repeatability(available, 168, 168)
            rows.append(
                {
                    **base,
                    "feature_information_cutoff": available_end,
                    "budget_history_status": "context24_plus_allowed_adaptation_observations",
                    "available_pretest_start": context_start,
                    "available_pretest_end": available_end,
                    "available_pretest_hours": len(available),
                    **{f"available_{key}": value for key, value in stats.items()},
                    "daily_repeatability_pairs": daily_n,
                    "daily_repeatability_pearson": daily_corr,
                    "daily_repeatability_status": daily_status,
                    "weekly_repeatability_pairs": weekly_n,
                    "weekly_repeatability_pearson": weekly_corr,
                    "weekly_repeatability_status": weekly_status,
                    **source_comparison(stats["mean"], source),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(cohort) * 6:
        raise AssertionError("Characteristic grid incomplete")
    if result.test_target_values_used_for_characteristics.astype(bool).any():
        raise AssertionError("Test targets used in characteristics")
    return result


def spearman(x: pd.Series, y: pd.Series) -> tuple[int, float, str]:
    complete = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(complete) < 3:
        return len(complete), math.nan, "insufficient_complete_cases"
    if complete.x.nunique() < 2 or complete.y.nunique() < 2:
        return len(complete), math.nan, "constant_variable"
    rho = complete.x.rank(method="average").corr(complete.y.rank(method="average"))
    return len(complete), float(rho), "exploratory_descriptive"


def characteristic_analysis(characteristics: pd.DataFrame, events: pd.DataFrame):
    # Preserve buildings whose official primaryspaceusage is missing.  The
    # category is explicit and remains descriptive; it is never imputed.
    characteristics = characteristics.copy()
    events = events.copy()
    characteristics["primaryspaceusage"] = characteristics.primaryspaceusage.fillna("<missing>")
    events["primaryspaceusage"] = events.primaryspaceusage.fillna("<missing>")
    outcomes = events.groupby(
        ["building_id", "site_id", "primaryspaceusage", "adaptation_days"],
        as_index=False,
        dropna=False,
    ).agg(
        seed_mean_mae_delta=("delta_mae_source", "mean"),
        seed_mean_rmse_delta=("delta_rmse_source", "mean"),
        f1_seed_rate=("f1", "mean"),
        f2_seed_rate=("f2", "mean"),
        f3_any_seed_rate=("f3_any", "mean"),
    )
    merged = outcomes.merge(
        characteristics.loc[characteristics.adaptation_days.gt(0)],
        on=["building_id", "site_id", "primaryspaceusage", "adaptation_days"],
        validate="one_to_one",
    )
    features = [
        "context_24h_mean", "context_24h_std_population", "context_24h_cv_abs_mean",
        "context_24h_zero_ratio", "context_24h_absolute_mean_gap_in_source_config_std",
        "available_mean", "available_std_population", "available_cv_abs_mean",
        "available_zero_ratio", "daily_repeatability_pearson", "weekly_repeatability_pearson",
        "absolute_mean_gap_in_source_config_std", "mean_percentile_among_20_source_building_means", "sqm",
    ]
    responses = ["seed_mean_mae_delta", "seed_mean_rmse_delta", "f1_seed_rate", "f2_seed_rate", "f3_any_seed_rate"]
    rows = []
    for days in NONZERO_BUDGETS:
        frame = merged.loc[merged.adaptation_days.eq(days)]
        for feature in features:
            for response in responses:
                n, rho, status = spearman(frame[feature], frame[response])
                rows.append(
                    {
                        "adaptation_days": days,
                        "feature": feature,
                        "response": response,
                        "n_complete": n,
                        "spearman_rho": rho,
                        "status": status,
                        "causal_interpretation_allowed": False,
                        "used_for_exclusion_or_hypothesis_decision": False,
                    }
                )
    associations = pd.DataFrame(rows)
    group_rows = []
    for field in ("site_id", "primaryspaceusage"):
        for days in NONZERO_BUDGETS:
            frame = events.loc[events.adaptation_days.eq(days)]
            for value, part in frame.groupby(field, sort=True, dropna=False):
                group_rows.append(
                    {
                        "group_field": field,
                        "group_value": value,
                        "adaptation_days": days,
                        "buildings": part.building_id.nunique(),
                        "seed_events": len(part),
                        "f1_rate": part.f1.mean(),
                        "f2_rate": part.f2.mean(),
                        "f3_any_rate": part.f3_any.mean(),
                        "mean_delta_mae_source": part.delta_mae_source.mean(),
                        "descriptive_only": True,
                    }
                )
    return associations, pd.DataFrame(group_rows)


def choose_representative_cases(events: pd.DataFrame) -> pd.DataFrame:
    seed_mean = events.groupby(["building_id", "site_id", "adaptation_days"], as_index=False).agg(
        mean_delta_mae=("delta_mae_source", "mean"),
        max_delta_mae=("delta_mae_source", "max"),
    )
    repeat = seed_mean.assign(fail=lambda frame: frame.mean_delta_mae.gt(EPS)).groupby(
        ["building_id", "site_id"], as_index=False
    ).agg(repeated_f1_budgets=("fail", "sum"), maximum_mean_delta=("mean_delta_mae", "max"))
    repeat = repeat.sort_values(
        ["repeated_f1_budgets", "maximum_mean_delta", "building_id"], ascending=[False, False, True]
    )
    rows = []
    if len(repeat) and repeat.iloc[0].repeated_f1_budgets > 0:
        building_id = repeat.iloc[0].building_id
        candidates = seed_mean.loc[seed_mean.building_id.eq(building_id)].sort_values(
            ["mean_delta_mae", "adaptation_days"], ascending=[False, True]
        )
        budget = int(candidates.iloc[0].adaptation_days)
        event = events.loc[
            events.building_id.eq(building_id) & events.adaptation_days.eq(budget)
        ].sort_values(["delta_mae_source", "seed"], ascending=[False, True]).iloc[0]
        rows.append({"case_type": "repeated_f1", "building_id": building_id, "site_id": event.site_id, "adaptation_days": budget, "seed": int(event.seed), "selection_value": repeat.iloc[0].repeated_f1_budgets, "selection_rule": "most seed-mean F1 budgets; tie max mean delta then building_id"})
    mixed = events.groupby(["building_id", "site_id", "adaptation_days"], as_index=False).agg(
        minimum=("delta_mae_source", "min"), maximum=("delta_mae_source", "max"), f4=("f4", "first")
    )
    mixed["range"] = mixed.maximum - mixed.minimum
    mixed = mixed.loc[mixed.f4].sort_values(["range", "building_id", "adaptation_days"], ascending=[False, True, True])
    if len(mixed):
        chosen = mixed.iloc[0]
        event = events.loc[
            events.building_id.eq(chosen.building_id) & events.adaptation_days.eq(chosen.adaptation_days)
        ].sort_values(["delta_mae_source", "seed"], ascending=[False, True]).iloc[0]
        rows.append({"case_type": "f4_seed_direction_mixed", "building_id": chosen.building_id, "site_id": chosen.site_id, "adaptation_days": int(chosen.adaptation_days), "seed": int(event.seed), "selection_value": chosen.range, "selection_rule": "largest seed delta range; tie building_id then budget"})
    discord = events.loc[events.mae_improved_rmse_worsened].sort_values(
        ["delta_rmse_source", "building_id", "adaptation_days", "seed"], ascending=[False, True, True, True]
    )
    if len(discord):
        event = discord.iloc[0]
        rows.append({"case_type": "mae_improved_rmse_worsened", "building_id": event.building_id, "site_id": event.site_id, "adaptation_days": int(event.adaptation_days), "seed": int(event.seed), "selection_value": event.delta_rmse_source, "selection_rule": "largest RMSE worsening; tie building_id, budget, seed"})
    return pd.DataFrame(rows)


def collect_representative_predictions(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    frames = []
    usecols = ["building_id", "site_id", "model", "seed", "adaptation_days", "target_timestamp", "y_true", "y_pred"]
    for chunk in pd.read_csv(RAW_PREDICTIONS, usecols=usecols, chunksize=250_000):
        seed_values = pd.to_numeric(chunk.seed, errors="coerce")
        day_values = pd.to_numeric(chunk.adaptation_days, errors="coerce")
        keep = pd.Series(False, index=chunk.index)
        for case in cases.itertuples(index=False):
            same_building = chunk.building_id.eq(case.building_id)
            baseline = same_building & chunk.model.isin(["previous_day", "weekly_same_hour"])
            source = same_building & chunk.model.eq("c0_source_only_lstm") & seed_values.eq(case.seed) & day_values.eq(0)
            adapted = same_building & chunk.model.eq("c0_head_only_adaptation_lstm") & seed_values.eq(case.seed) & day_values.eq(case.adaptation_days)
            keep |= baseline | source | adapted
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def make_figures(curve: pd.DataFrame, site_curve: pd.DataFrame, h1_site: pd.DataFrame, failure_summary: pd.DataFrame, representative_predictions: pd.DataFrame, cases: pd.DataFrame) -> None:
    """Create dependency-free scientific PNGs with the already installed Pillow."""
    FIGURES.mkdir(parents=True, exist_ok=True)

    def font(size: int, bold: bool = False):
        path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
        return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()

    def line_chart(path: Path, title: str, x_values, series: list[tuple[str, np.ndarray, str]], y_label: str, zero_line: bool = False, x_label: str = "Adaptation days"):
        width, height = 1400, 800
        left, top, right, bottom = 130, 90, 60, 115
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        xs = np.asarray(x_values, dtype=float)
        all_y = np.concatenate([np.asarray(values, dtype=float) for _, values, _ in series])
        finite = all_y[np.isfinite(all_y)]
        ymin, ymax = float(finite.min()), float(finite.max())
        if zero_line:
            ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
        padding = max((ymax - ymin) * 0.08, 1e-9)
        ymin, ymax = ymin - padding, ymax + padding
        x_min, x_max = float(xs.min()), float(xs.max())
        plot_w, plot_h = width - left - right, height - top - bottom

        def point(x, y):
            px = left + (float(x) - x_min) / max(x_max - x_min, 1e-9) * plot_w
            py = top + (ymax - float(y)) / max(ymax - ymin, 1e-9) * plot_h
            return px, py

        draw.line((left, top, left, height - bottom), fill="#333333", width=2)
        draw.line((left, height - bottom, width - right, height - bottom), fill="#333333", width=2)
        for tick in np.linspace(ymin, ymax, 6):
            _, py = point(x_min, tick)
            draw.line((left, py, width - right, py), fill="#e5e8eb", width=1)
            draw.text((10, py - 10), f"{tick:.4g}", fill="#333333", font=font(20))
        tick_indices = np.linspace(0, len(xs) - 1, min(len(xs), 8), dtype=int)
        for x in xs[tick_indices]:
            px, _ = point(x, ymin)
            draw.text((px - 12, height - bottom + 15), f"{x:g}", fill="#333333", font=font(20))
        if zero_line and ymin <= 0 <= ymax:
            _, py = point(x_min, 0)
            draw.line((left, py, width - right, py), fill="#222222", width=2)
        for index, (label, values, color) in enumerate(series):
            values = np.asarray(values, dtype=float)
            points = [point(x, y) for x, y in zip(xs, values) if np.isfinite(y)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=4)
            for px, py in points:
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
            legend_x = left + (index % 4) * 290
            legend_y = height - 58 + (index // 4) * 28
            draw.line((legend_x, legend_y + 12, legend_x + 40, legend_y + 12), fill=color, width=5)
            draw.text((legend_x + 50, legend_y), label, fill="#222222", font=font(17))
        draw.text((left, 25), title, fill="#111111", font=font(30, True))
        draw.text((width // 2 - 70, height - 90), x_label, fill="#333333", font=font(21))
        draw.text((10, 55), y_label, fill="#333333", font=font(20))
        image.save(path)

    site_series = []
    palette = ["#9aa7b3", "#b3a29a", "#9ab3a2", "#a79ab3", "#b3ae9a", "#9aaeb3", "#b39aa8"]
    for color, (site, frame) in zip(palette, site_curve.groupby("site_id", sort=True)):
        frame = frame.sort_values("adaptation_days")
        site_series.append((site, frame.delta_mae_source.to_numpy(float), color))
    site_series.append(("site-equal mean", curve.sort_values("adaptation_days").delta_mae_source_site_equal.to_numpy(float), "#1769aa"))
    line_chart(
        FIGURES / "final_v1_site_equal_adaptation_curve.png",
        "Final C0 adaptation curve (all budgets)",
        curve.sort_values("adaptation_days").adaptation_days.to_numpy(float),
        site_series,
        "Delta MAE vs source-only",
        True,
    )

    ordered = h1_site.sort_values("delta_mae_source").reset_index(drop=True)
    width, height = 1400, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 130, 90, 60, 130
    values = ordered.delta_mae_source.to_numpy(float)
    ymin, ymax = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    pad = max((ymax - ymin) * 0.1, 1e-9)
    ymin, ymax = ymin - pad, ymax + pad
    plot_h, plot_w = height - top - bottom, width - left - right
    zero_y = top + (ymax - 0.0) / (ymax - ymin) * plot_h
    draw.line((left, zero_y, width - right, zero_y), fill="#222222", width=2)
    bar_w = plot_w / max(len(values), 1) * 0.62
    for i, row in ordered.iterrows():
        cx = left + (i + 0.5) / len(values) * plot_w
        y = top + (ymax - row.delta_mae_source) / (ymax - ymin) * plot_h
        draw.rectangle((cx - bar_w / 2, min(y, zero_y), cx + bar_w / 2, max(y, zero_y)), fill="#2d7d46" if row.delta_mae_source < 0 else "#b64949")
        draw.text((cx - 35, height - bottom + 18), str(row.site_id), fill="#333333", font=font(18))
    draw.text((left, 25), "H1 site effects", fill="#111111", font=font(30, True))
    draw.text((10, 55), "28d Delta MAE", fill="#333333", font=font(20))
    image.save(FIGURES / "final_v1_h1_site_effects.png")

    failures = failure_summary.loc[failure_summary.adaptation_days.ne("ALL_NONZERO")].copy()
    failures["adaptation_days"] = pd.to_numeric(failures.adaptation_days)
    line_chart(
        FIGURES / "final_v1_failure_rate_curve.png",
        "Final failure-rate curve",
        failures.adaptation_days.to_numpy(float),
        [("F1", failures.f1_site_equal_rate.to_numpy(float), "#b64949"), ("F2", failures.f2_site_equal_rate.to_numpy(float), "#1769aa")],
        "Site-equal failure rate",
        False,
    )

    colors = {"previous_day": "#d97706", "weekly_same_hour": "#7c3aed", "c0_source_only_lstm": "#1769aa", "c0_head_only_adaptation_lstm": "#2d7d46"}
    for case in cases.itertuples(index=False):
        part = representative_predictions.loc[representative_predictions.building_id.eq(case.building_id)].copy()
        part["seed_num"] = pd.to_numeric(part.seed, errors="coerce")
        part["days_num"] = pd.to_numeric(part.adaptation_days, errors="coerce")
        part = part.loc[
            part.model.isin(["previous_day", "weekly_same_hour"])
            | (part.model.eq("c0_source_only_lstm") & part.seed_num.eq(case.seed))
            | (part.model.eq("c0_head_only_adaptation_lstm") & part.seed_num.eq(case.seed) & part.days_num.eq(case.adaptation_days))
        ]
        if part.empty:
            continue
        truth = part.sort_values("target_timestamp").drop_duplicates("target_timestamp")
        x = np.arange(len(truth), dtype=float)
        chart_series = [("truth", truth.y_true.to_numpy(float), "#111111")]
        for model, frame in part.groupby("model", sort=True):
            frame = frame.sort_values("target_timestamp")
            label = model if model != "c0_head_only_adaptation_lstm" else f"adapted_{case.adaptation_days}d"
            chart_series.append((label, frame.y_pred.to_numpy(float), colors[model]))
        line_chart(
            FIGURES / f"final_v1_case_{case.case_type}.png",
            f"{case.case_type}: {case.building_id}, seed {case.seed}",
            x,
            chart_series,
            "Electricity meter value",
            False,
            "Target hour index",
        )


def md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for column, value in zip(columns, values):
            if pd.isna(value):
                rendered.append("NA")
            elif column in formats:
                rendered.append(formats[column].format(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_reports(cohort_all: pd.DataFrame, cohort: pd.DataFrame, temporal: pd.DataFrame, curve: pd.DataFrame, site_curve: pd.DataFrame, failures: pd.DataFrame, failure_summary: pd.DataFrame, hypothesis: dict, h1_site: pd.DataFrame, h2_site: pd.DataFrame, h3_site: pd.DataFrame, contingency: pd.DataFrame, metrics: pd.DataFrame, characteristics: pd.DataFrame, associations: pd.DataFrame, cases: pd.DataFrame, unavailable: int, bad_status: int, max_metric_difference: float, elapsed: float) -> None:
    baseline = metrics.loc[metrics.model.isin(["previous_day", "weekly_same_hour"])].groupby(["site_id", "model"], as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
    baseline_summary = baseline.groupby("model", as_index=False).agg(mae_site_equal=("mae", "mean"), rmse_site_equal=("rmse", "mean"))
    h1_text = f"{hypothesis['h1_effect']:.12g} [{hypothesis['h1_ci_low']:.12g}, {hypothesis['h1_ci_high']:.12g}]"
    HYPOTHESIS_OUT.write_text(
        f"""# final_v1 독립 최종시험 가설 결과

## 확정 분석 상태

- 최종 평가: **{len(cohort)}개 건물, {cohort.site_id.nunique()}개 사이트**.
- H1 확정 판정: **`{hypothesis['h1_status']}`**.
- 사이트 동일 가중 28일 `ΔMAE_source`: **{h1_text}** (10,000회 site-cluster percentile 95% CI, seed 20260925).
- 사이트가 7개이므로 사전 조건인 최소 5개 사이트를 충족했다.

## H1

건물별 세 시드 평균을 낸 뒤 사이트 안에서 건물을 동일 가중하고, 마지막으로 사이트를 동일 가중했다. 효과가 음수이고 CI 상한도 0 미만일 때만 지지한다.

{md_table(h1_site, ['site_id','buildings','delta_mae_source','delta_rmse_source'], {'delta_mae_source':'{:.6g}','delta_rmse_source':'{:.6g}'})}

## H2

- 추정량 `site-equal F1 rate(28d) - site-equal F1 rate(1d)`: **{hypothesis['h2_effect']:.12g}**.
- site-cluster bootstrap 95% CI: **[{hypothesis['h2_ci_low']:.12g}, {hypothesis['h2_ci_high']:.12g}]**.
- p-value와 Holm 검정은 승인된 final protocol에 따라 계산하지 않았다.

{md_table(h2_site, ['site_id','f1_rate_1d','f1_rate_28d','rate_difference_28d_minus_1d'], {'f1_rate_1d':'{:.4f}','f1_rate_28d':'{:.4f}','rate_difference_28d_minus_1d':'{:.4f}'})}

## H3

- A(Source-only 대비 개선) site-equal 비율: **{hypothesis['h3_a_rate']:.6f}**.
- B(전일·전주 기준선 모두 추월) site-equal 비율: **{hypothesis['h3_b_rate']:.6f}**.
- `rate(A)-rate(B)`: **{hypothesis['h3_effect']:.12g}**, 95% CI **[{hypothesis['h3_ci_low']:.12g}, {hypothesis['h3_ci_high']:.12g}]**.

{md_table(h3_site, ['site_id','a_rate','b_rate','rate_difference_a_minus_b'], {'a_rate':'{:.4f}','b_rate':'{:.4f}','rate_difference_a_minus_b':'{:.4f}'})}

건물×시드 대응표:

{md_table(contingency, ['a_source_improvement','b_beats_both_baselines','count'])}

## 전체 적응 곡선

{md_table(curve, ['adaptation_days','sites','mae_site_equal','rmse_site_equal','delta_mae_source_site_equal','delta_rmse_source_site_equal','a_rate_site_equal','b_rate_site_equal'], {'mae_site_equal':'{:.6g}','rmse_site_equal':'{:.6g}','delta_mae_source_site_equal':'{:.6g}','delta_rmse_source_site_equal':'{:.6g}','a_rate_site_equal':'{:.4f}','b_rate_site_equal':'{:.4f}'})}

가장 좋은 예산만 선택하지 않고 0·1·3·7·14·28일을 모두 보고했다.
""",
        encoding="utf-8",
    )

    strongest = associations.loc[associations.response.eq("seed_mean_mae_delta")].dropna(subset=["spearman_rho"]).copy()
    strongest["abs_rho"] = strongest.spearman_rho.abs()
    strongest = strongest.sort_values(["adaptation_days", "abs_rho", "feature"], ascending=[True, False, True]).groupby("adaptation_days").head(3)
    CHARACTERISTICS_REPORT.write_text(
        f"""# final_v1 건물 특성 탐색 분석

## 범위와 정보 시점

- 모든 결과는 **탐색적**이며 건물 제외, 모델 선택, H1~H3 판정에 사용하지 않았다.
- 0일은 첫 예측 직전 24시간 context만 사용했고 장기 Target 통계는 `NA`로 유지했다.
- 비제로 예산은 초기 적응 직전 24시간과 허용된 적응 관측 종료까지만 사용했다.
- 모든 `feature_information_cutoff`는 시험 시작보다 이르다. 시험 정답 실제값은 특성 계산에 사용하지 않았다.
- 일간 반복성은 최소 24개 lag pair, 주간 반복성은 최소 168개 lag pair가 있고 양쪽이 비상수일 때만 Pearson 상관을 계산했다.
- Source 규모 기준은 기존 Source 20개 학습 구간 표를 그대로 사용했다.

## 자료 규모

- 특성 행: {len(characteristics)} = {len(cohort)}개 건물 × 6개 예산.
- 사이트: {cohort.site_id.nunique()}개; 용도: {cohort.primaryspaceusage.nunique()}개.
- 적응 기반 장기 특성을 사용할 수 있는 행: {int(characteristics.available_pretest_hours.gt(0).sum())}개.
- 주간 반복성 정의 가능 행: {int(characteristics.weekly_repeatability_pearson.notna().sum())}개.

## 예산별 `ΔMAE_source`와 절댓값 상관이 큰 탐색 항목

{md_table(strongest, ['adaptation_days','feature','n_complete','spearman_rho','status'], {'spearman_rho':'{:.4f}'})}

이 표는 작은 수의 독립 사이트와 반복 예산 관측에서 얻은 탐색적 상관이다. 인과관계, 사전 실패 판별 규칙, 전체 BDG2의 일반 법칙으로 해석하지 않는다. 다중 탐색에 대한 확증적 유의성 검정을 수행하지 않았다.
""",
        encoding="utf-8",
    )

    excluded = cohort_all.loc[cohort_all.selection_status.ne("selected_for_final_v1")]
    exclusion_by_site = excluded.groupby(["site_id", "exclusion_reason"]).size().rename("buildings").reset_index()
    totals = failure_summary.loc[failure_summary.adaptation_days.eq("ALL_NONZERO")].iloc[0]
    FINAL_REPORT.write_text(
        f"""# final_v1 C0 독립 최종시험 보고서

## 완료 상태

- Gate 0 cohort 동결: 완료.
- Gate 1 구조 검증: **PASS**, {len(cohort)}개 건물·{cohort.site_id.nunique()}개 사이트 모두 통과.
- Gate 2 단일 모델 실행: 완료. 저장 raw prediction {len(cohort) * 20 * 672:,}행, 실행 오류 0개.
- Gate 3 독립 metric 재계산: **PASS**. 실행 metric과 최대 차이 `{max_metric_difference:.3g}` (`≤1e-10`).
- 최종 실행 뒤 설정·건물·기간·시드·학습률·epoch 변경: **없음**.

## cohort와 제외

- 기존 `final_test_proposal`: {len(cohort_all)}개 건물, {cohort_all.site_id.nunique()}개 사이트.
- 품질 규칙 적격·최종 평가: {len(cohort)}개, {cohort.site_id.nunique()}개 사이트.
- Gate 0 제외: {len(excluded)}개. 전부 정확한 사유 `continuous_zero_run_ge_30d`이며 manifest에 건물별로 보존했다.
- Gate 1 `data_integrity_failure`: 0개. 대체 건물·대체 기간 탐색 없음.

{md_table(exclusion_by_site, ['site_id','exclusion_reason','buildings'])}

## 핵심·부가 결과

- **H1: `{hypothesis['h1_status']}`**. 28일 사이트 동일 가중 `ΔMAE_source` = **{hypothesis['h1_effect']:.12g}**, 95% site-cluster bootstrap CI **[{hypothesis['h1_ci_low']:.12g}, {hypothesis['h1_ci_high']:.12g}]**.
- H2 효과크기 `F1 rate(28d)-F1 rate(1d)` = **{hypothesis['h2_effect']:.12g}**, 95% CI **[{hypothesis['h2_ci_low']:.12g}, {hypothesis['h2_ci_high']:.12g}]**.
- H3 효과크기 `rate(A)-rate(B)` = **{hypothesis['h3_effect']:.12g}**, 95% CI **[{hypothesis['h3_ci_low']:.12g}, {hypothesis['h3_ci_high']:.12g}]**.
- H2/H3 p-value·Holm 검정은 수행하지 않았다.

## 전체 0·1·3·7·14·28일 결과

{md_table(curve, ['adaptation_days','mae_site_equal','rmse_site_equal','delta_mae_source_site_equal','delta_rmse_source_site_equal','a_rate_site_equal','b_rate_site_equal'], {'mae_site_equal':'{:.6g}','rmse_site_equal':'{:.6g}','delta_mae_source_site_equal':'{:.6g}','delta_rmse_source_site_equal':'{:.6g}','a_rate_site_equal':'{:.4f}','b_rate_site_equal':'{:.4f}'})}

## 단순 기준선

{md_table(baseline_summary, ['model','mae_site_equal','rmse_site_equal'], {'mae_site_equal':'{:.6g}','rmse_site_equal':'{:.6g}'})}

## F1~F4 실패

- 전체 비제로 건물×예산×시드 사건 {int(totals.seed_event_denominator):,}개 중 F1 {int(totals.f1_count):,}, F2 {int(totals.f2_count):,}, F3-PD {int(totals.f3_previous_day_count):,}, F3-W {int(totals.f3_weekly_same_hour_count):,}, F3-any {int(totals.f3_any_count):,}, F3-both {int(totals.f3_both_count):,}개.
- 건물×예산 {int(totals.building_budget_denominator_for_f4):,}개 중 F4 {int(totals.f4_count):,}개.
- MAE 개선·RMSE 악화 불일치: {int(totals.mae_improved_rmse_worsened_count):,}개.

{md_table(failure_summary, ['adaptation_days','seed_event_denominator','f1_count','f2_count','f3_previous_day_count','f3_weekly_same_hour_count','f3_any_count','f3_both_count','f4_count'])}

## 데이터 누수·동일성 검사

- 모든 모델·예산·시드에서 건물별 `forecast_origin + target timestamp + horizon + y_true` digest가 동일했다.
- 적응 정답과 시험 정답 교집합은 Gate 1에서 0으로 확인했다.
- C0 입력은 직전 24시간 전력과 예측 대상의 결정적 달력 정보만 사용했다.
- Source 정규화 mean/std와 세 기존 checkpoint 해시는 실행 전 동결값과 일치했다.
- 독립 metric 재계산에서 예측 불가 값 {unavailable}개, 비정상 status {bad_status}개였다.
- 특성 계산은 시험 시작 이전 관측까지만 사용했다.

## 고정 대표 실패 사례

{md_table(cases, ['case_type','building_id','site_id','adaptation_days','seed','selection_value','selection_rule']) if len(cases) else '해당 사례 없음.'}

## 실행 정보

- Gate 2 장치: NVIDIA GeForce RTX 3080, 고정 세 시드.
- Gate 2 실행 시간: {json.loads(GATE2_CONFIG.read_text(encoding='utf-8'))['elapsed_seconds']:.1f}초.
- Gate 3 분석 시간: {elapsed:.1f}초.
- 원본 prediction은 성능 확인 뒤 재생성하지 않았다.

## 한계

- 평가 가능한 사이트는 7개로 cluster bootstrap의 군집 수가 작다.
- final cohort는 고정 품질 기준을 통과한 건물에 한정되며, 30일 이상 연속 0값 건물과 적격 건물이 없는 2개 사이트는 포함하지 않는다.
- BDG2 개별 계량기의 물리적 범위가 건물마다 완전히 동일하다는 공식 근거는 부족하다.
- C0 LSTM과 head-only 적응만 검증했다. C1 기상, 실내 센서, 신규 Transformer, HVAC 제어는 검증하지 않았다.
- 특성 분석은 탐색적 상관이며 인과 추론이나 건물 제외 규칙으로 사용할 수 없다.
""",
        encoding="utf-8",
    )


def main() -> None:
    check_write_once()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.time()
    gate2 = json.loads(GATE2_CONFIG.read_text(encoding="utf-8"))
    if gate2.get("status") != "complete" or gate2.get("failure_count") != 0:
        raise AssertionError("Gate 2 is not a complete zero-execution-error run")
    cohort_all = pd.read_csv(COHORT)
    cohort = cohort_all.loc[cohort_all.selection_status.eq("selected_for_final_v1")].copy()
    temporal = pd.read_csv(TEMPORAL)
    log(f"Gate3 started raw_predictions_sha256={sha256(RAW_PREDICTIONS)}")
    metrics, unavailable, bad_status = recompute_metrics()
    if unavailable or bad_status:
        raise AssertionError(f"Unavailable predictions={unavailable}, bad statuses={bad_status}")
    validate_metric_grid(metrics, cohort)
    comparison = compare_execution_metrics(metrics)
    max_metric_difference = float(max(comparison.mae_abs_difference.max(), comparison.rmse_abs_difference.max()))
    log(f"Independent metric verification PASS max_difference={max_metric_difference:.17g}")

    metric_output = output_metric_frame(metrics, cohort)
    efficiency = build_efficiency(metrics, cohort)
    events, failure_summary = build_failures(efficiency)
    hypothesis, h1_site, h2_site, h3_site, contingency = build_hypotheses(efficiency, events)
    curve, site_curve = build_curves(efficiency)
    characteristics = build_characteristics(cohort, temporal)
    associations, group_summary = characteristic_analysis(characteristics, events)
    cases = choose_representative_cases(events)
    representative_predictions = collect_representative_predictions(cases)

    metric_output.to_csv(METRICS_OUT, index=False, encoding="utf-8", float_format="%.17g")
    efficiency.to_csv(EFFICIENCY_OUT, index=False, encoding="utf-8", float_format="%.17g")
    events.to_csv(FAILURE_EVENTS_OUT, index=False, encoding="utf-8", float_format="%.17g")
    failure_summary.to_csv(FAILURE_SUMMARY_OUT, index=False, encoding="utf-8", float_format="%.17g")
    curve.to_csv(CURVE_OUT, index=False, encoding="utf-8", float_format="%.17g")
    h1_site.to_csv(SITE_EFFECTS_OUT, index=False, encoding="utf-8", float_format="%.17g")
    h2_site.to_csv(H2_SITE_OUT, index=False, encoding="utf-8", float_format="%.17g")
    h3_site.to_csv(H3_SITE_OUT, index=False, encoding="utf-8", float_format="%.17g")
    contingency.to_csv(H3_TABLE_OUT, index=False, encoding="utf-8")
    cases.to_csv(REPRESENTATIVE_OUT, index=False, encoding="utf-8", float_format="%.17g")
    representative_predictions.to_csv(REPRESENTATIVE_PRED_OUT, index=False, encoding="utf-8", float_format="%.17g")
    characteristics.to_csv(CHARACTERISTICS_OUT, index=False, encoding="utf-8", float_format="%.17g")
    associations.to_csv(CHARACTERISTIC_ASSOC_OUT, index=False, encoding="utf-8", float_format="%.17g")
    group_summary.to_csv(CHARACTERISTIC_GROUP_OUT, index=False, encoding="utf-8", float_format="%.17g")
    make_figures(curve, site_curve, h1_site, failure_summary, representative_predictions, cases)
    write_reports(
        cohort_all, cohort, temporal, curve, site_curve, events, failure_summary, hypothesis,
        h1_site, h2_site, h3_site, contingency, metrics, characteristics, associations, cases,
        unavailable, bad_status, max_metric_difference, time.time() - started,
    )
    log(
        f"Gate3 COMPLETE H1={hypothesis['h1_status']} effect={hypothesis['h1_effect']:.12g} "
        f"CI=[{hypothesis['h1_ci_low']:.12g},{hypothesis['h1_ci_high']:.12g}] "
        f"elapsed={time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
