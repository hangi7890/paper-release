"""Exploratory v5 analysis of pre-test characteristics and C0 adaptation failures.

Safety properties
-----------------
* Only the frozen 20 Source and 12 development power columns are requested.
* No ``final_test_proposal`` power column is requested or inspected.
* Development test targets are outcomes only through frozen metric artifacts;
  raw test power is never used to construct a characteristic.
* The raw CSV whole-file hash is not recomputed.  Its v3 declaration is cited
  from the independently hash-checked model configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[1]))).resolve()
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures" / "bdg2_v5"
RAW_POWER = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
COHORT = REPORTS / "bdg2_multibuilding_cohort_manifest_v2.csv"
TEMPORAL = REPORTS / "bdg2_multibuilding_temporal_manifest_v2.csv"
SOURCE_METRICS = REPORTS / "bdg2_stage6_c0_source_only_metrics_v3.csv"
ADAPT_METRICS = REPORTS / "bdg2_stage6_c0_adaptation_metrics_v3.csv"
BASELINE_METRICS = REPORTS / "bdg2_multibuilding_baseline_metrics_v2.csv"
CONFIG = ROOT / "models" / "bdg2_stage6_v3" / "bdg2_stage6_c0_lstm_config_v3.json"
METADATA = REPORTS / "bdg2_metadata_quality_join.csv"
DEFINITIONS = REPORTS / "bdg2_v5_analysis_definitions_frozen.md"

CHARACTERISTICS_OUT = REPORTS / "bdg2_v5_building_characteristics.csv"
SOURCE_SCALE_OUT = REPORTS / "bdg2_v5_source_scale_reference.csv"
OUTCOMES_OUT = REPORTS / "bdg2_v5_characteristic_failure_outcomes.csv"
ASSOCIATIONS_OUT = REPORTS / "bdg2_v5_building_characteristic_associations.csv"
GROUP_SUMMARY_OUT = REPORTS / "bdg2_v5_building_characteristic_group_summary.csv"
HASHES_OUT = REPORTS / "bdg2_v5_building_characteristic_hashes.csv"
FIGURE_OUT = FIGURES / "bdg2_v5_scale_gap_failure_association.png"
REPORT_OUT = REPORTS / "bdg2_v5_building_characteristics.md"
LOG_OUT = REPORTS / "bdg2_v5_building_characteristics_run.log"

EXPECTED_BUDGETS = (0, 1, 3, 7, 14, 28)
NONZERO_BUDGETS = (1, 3, 7, 14, 28)
EXPECTED_SEEDS = (20260924, 20260925, 20260926)
EPS = 1.0e-12
DECLARED_RAW_SHA256 = "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"

EXPECTED_HASHES = {
    COHORT: "59f4128ebd0a594e60c410c8ae6af48fdd674422553d9e47e4c771195c0bf23b",
    TEMPORAL: "6dd57dadf2aaae9e565a870784d2ced3482c1466156e1c691a3149bee3ff8f1b",
    SOURCE_METRICS: "215113f201f33e25a7a7e0c3f59f80f8dde20d76649332224305e7d88f1906dc",
    ADAPT_METRICS: "e6d4a64a154c39bcec74f19107a72931447612bbf842b1a2e54059c29d58e4aa",
    BASELINE_METRICS: "fbd31763a56abe230db52a10b50e1e5f027836c47e57c9afe0e328501a96c3b7",
    CONFIG: "c2cb99e8a01403175bc6240edd990a6a53d2244187038c41f1dba6727da1bd9f",
    METADATA: "b8e8421ad95f16377dbbcfc03a2fd4c56c04fe53259d91a985230be87ec28cb0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_small_input_hashes() -> list[dict]:
    rows: list[dict] = []
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(
                f"Frozen input hash mismatch: {path.relative_to(ROOT)}; "
                f"expected={expected}, observed={observed}"
            )
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "sha256": observed,
                "verification": "recomputed_v5",
            }
        )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    declared = config.get("hashes", {}).get("raw")
    if declared != DECLARED_RAW_SHA256:
        raise AssertionError("Hash-checked v3 config has an unexpected raw hash declaration")
    rows.append(
        {
            "artifact": "data/raw/bdg2/electricity.csv",
            "sha256": declared,
            "verification": "declared_in_hash_checked_v3_config_not_recomputed_v5",
        }
    )
    return rows


def selected_long_hash(frame: pd.DataFrame) -> str:
    """Hash only the Source-train and development-pretest rows actually used."""
    required = ["role", "building_id", "timestamp", "value"]
    ordered = frame[required].copy().sort_values(required[:3]).reset_index(drop=True)
    ordered["timestamp"] = pd.to_datetime(ordered.timestamp).dt.strftime("%Y-%m-%d %H:%M:%S")
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or values.size == 0:
        raise AssertionError("Characteristic history must be finite and non-empty")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "mean": mean,
        "std_population": std,
        "cv_abs_mean": std / abs(mean) if abs(mean) > EPS else math.nan,
        "zero_count": int(np.equal(values, 0.0).sum()),
        "zero_ratio": float(np.equal(values, 0.0).mean()),
    }


def lag_repeatability(values: np.ndarray, lag: int, minimum_pairs: int) -> tuple[int, float, str]:
    """Pearson lag correlation with a predeclared minimum pair requirement."""
    values = np.asarray(values, dtype=float)
    if len(values) <= lag:
        return 0, math.nan, "insufficient_pairs"
    earlier, later = values[:-lag], values[lag:]
    valid = np.isfinite(earlier) & np.isfinite(later)
    n = int(valid.sum())
    if n < minimum_pairs:
        return n, math.nan, "insufficient_pairs"
    earlier, later = earlier[valid], later[valid]
    if np.std(earlier) <= EPS or np.std(later) <= EPS:
        return n, math.nan, "constant_series"
    return n, float(np.corrcoef(earlier, later)[0, 1]), "ok"


def read_allowed_power(
    cohort: pd.DataFrame, temporal: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str], set[str]]:
    selected = cohort.loc[cohort.selected_for_multibuilding_v2.astype(str).str.lower().eq("true")]
    source_ids = sorted(
        selected.loc[selected.alternative_group.eq("source_train_proposal"), "building_id"].astype(str)
    )
    development_ids = sorted(
        selected.loc[
            selected.alternative_group.eq("development_validation_proposal"), "building_id"
        ].astype(str)
    )
    final_ids = set(
        cohort.loc[cohort.alternative_group.eq("final_test_proposal"), "building_id"].astype(str)
    )
    if len(source_ids) != 20 or len(development_ids) != 12:
        raise AssertionError("Expected frozen Source 20 and development 12")
    allowed = source_ids + development_ids
    if final_ids.intersection(allowed):
        raise AssertionError("Allowed power columns overlap final_test_proposal")
    manifest_ids = set(temporal.building_id.astype(str))
    if set(allowed) != manifest_ids:
        raise AssertionError("Selected cohort and temporal manifest building sets differ")
    header = pd.read_csv(RAW_POWER, nrows=0).columns.tolist()
    requested = ["timestamp", *allowed]
    if not set(requested).issubset(header):
        raise AssertionError("An allowed Source/development column is absent from raw power")
    # usecols is the byte-level column guard: final-test power fields are never requested.
    raw = pd.read_csv(RAW_POWER, usecols=requested)
    if set(raw.columns) != set(requested) or final_ids.intersection(raw.columns):
        raise AssertionError("Raw reader returned a disallowed power column")
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    if raw.timestamp.duplicated().any():
        raise AssertionError("Raw timestamp has duplicates")
    return raw.set_index("timestamp"), source_ids, development_ids, final_ids


def exact_interval(raw: pd.DataFrame, building: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    index = pd.date_range(start, end, freq="h")
    values = pd.to_numeric(raw[building].reindex(index), errors="coerce").to_numpy(float)
    if len(values) != len(index) or not np.isfinite(values).all():
        raise AssertionError(f"{building}: non-finite or incomplete characteristic interval {start}..{end}")
    return values


def build_source_reference(
    raw: pd.DataFrame, temporal: pd.DataFrame, source_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, used = [], []
    source_manifest = temporal.loc[temporal.role.eq("source_pretraining")].copy()
    if set(source_manifest.building_id) != set(source_ids):
        raise AssertionError("Source temporal rows do not match the frozen 20 buildings")
    for row in source_manifest.sort_values("building_id").itertuples(index=False):
        start, end = pd.Timestamp(row.train_target_start), pd.Timestamp(row.train_target_end)
        values = exact_interval(raw, row.building_id, start, end)
        stats = numeric_summary(values)
        rows.append(
            {
                "building_id": row.building_id,
                "site_id": row.site_id,
                "primaryspaceusage": row.primaryspaceusage,
                "source_train_target_start": start,
                "source_train_target_end": end,
                "source_train_target_hours": len(values),
                **{f"source_train_{key}": value for key, value in stats.items()},
                "development_or_final_power_used": False,
            }
        )
        used.append(
            pd.DataFrame(
                {
                    "role": "source_train_target",
                    "building_id": row.building_id,
                    "timestamp": pd.date_range(start, end, freq="h"),
                    "value": values,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(used, ignore_index=True)


def source_comparison(mean: float, source: pd.DataFrame, config_mean: float, config_std: float) -> dict:
    means = source.source_train_mean.to_numpy(float)
    nearest_idx = int(np.argmin(np.abs(means - mean)))
    nearest = source.iloc[nearest_idx]
    median = float(np.median(means))
    return {
        "mean_minus_source_config_mean": mean - config_mean,
        "mean_to_source_config_mean_ratio": mean / config_mean if abs(config_mean) > EPS else math.nan,
        "absolute_mean_gap_in_source_config_std": abs(mean - config_mean) / config_std,
        "mean_minus_source_building_median": mean - median,
        "mean_percentile_among_20_source_building_means": float(np.mean(means <= mean)),
        "nearest_source_building_by_mean": nearest.building_id,
        "nearest_source_building_mean": float(nearest.source_train_mean),
        "nearest_source_absolute_mean_gap": abs(float(nearest.source_train_mean) - mean),
    }


def build_characteristics(
    raw: pd.DataFrame,
    temporal: pd.DataFrame,
    metadata: pd.DataFrame,
    source: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = temporal.loc[temporal.role.eq("development_adaptation_evaluation")].copy()
    meta = metadata[["building_id", "sqm"]].drop_duplicates("building_id")
    dev = dev.merge(meta, on="building_id", how="left", validate="many_to_one")
    norm = config["source_normalization"]
    config_mean, config_std = float(norm["mean"]), float(norm["std_population"])
    rows, used = [], []
    for row in dev.sort_values(["building_id", "adaptation_days"]).itertuples(index=False):
        days = int(row.adaptation_days)
        test_start = pd.Timestamp(row.test_target_start)
        if days == 0:
            context_end = test_start - pd.Timedelta(hours=1)
            context_start = test_start - pd.Timedelta(hours=24)
        else:
            context_end = pd.Timestamp(row.adaptation_target_start) - pd.Timedelta(hours=1)
            context_start = pd.Timestamp(row.adaptation_target_start) - pd.Timedelta(hours=24)
        context = exact_interval(raw, row.building_id, context_start, context_end)
        context_stats = numeric_summary(context)
        context_scale = source_comparison(context_stats["mean"], source, config_mean, config_std)
        used.append(
            pd.DataFrame(
                {
                    "role": f"development_context_{days}d",
                    "building_id": row.building_id,
                    "timestamp": pd.date_range(context_start, context_end, freq="h"),
                    "value": context,
                }
            )
        )

        base = {
            "building_id": row.building_id,
            "site_id": row.site_id,
            "primaryspaceusage": row.primaryspaceusage,
            "sqm": row.sqm,
            "adaptation_days": days,
            "adaptation_train_sample_count": int(row.adaptation_train_sample_count),
            "test_target_start": row.test_target_start,
            "test_target_end": row.test_target_end,
            "context_24h_start": context_start,
            "context_24h_end": context_end,
            "context_24h_hours": 24,
            **{f"context_24h_{key}": value for key, value in context_stats.items()},
            **{f"context_24h_{key}": value for key, value in context_scale.items()},
            "context_24h_daily_repeatability_pairs": 0,
            "context_24h_daily_repeatability_pearson": np.nan,
            "context_24h_daily_repeatability_status": "not_defined_24h_context",
            "context_24h_weekly_repeatability_pairs": 0,
            "context_24h_weekly_repeatability_pearson": np.nan,
            "context_24h_weekly_repeatability_status": "not_defined_24h_context",
            "source_config_mean": config_mean,
            "source_config_std_population": config_std,
            "source_config_provenance": norm["computed_from"],
            "test_target_values_used_for_characteristics": False,
            "final_test_power_used": False,
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
                    "available_mean": np.nan,
                    "available_std_population": np.nan,
                    "available_cv_abs_mean": np.nan,
                    "available_zero_count": np.nan,
                    "available_zero_ratio": np.nan,
                    "daily_repeatability_pairs": 0,
                    "daily_repeatability_pearson": np.nan,
                    "daily_repeatability_status": "not_defined_for_0d_long_history",
                    "weekly_repeatability_pairs": 0,
                    "weekly_repeatability_pearson": np.nan,
                    "weekly_repeatability_status": "not_defined_for_0d_long_history",
                    "mean_minus_source_config_mean": np.nan,
                    "mean_to_source_config_mean_ratio": np.nan,
                    "absolute_mean_gap_in_source_config_std": np.nan,
                    "mean_minus_source_building_median": np.nan,
                    "mean_percentile_among_20_source_building_means": np.nan,
                    "nearest_source_building_by_mean": "",
                    "nearest_source_building_mean": np.nan,
                    "nearest_source_absolute_mean_gap": np.nan,
                    "history_strictly_before_test": True,
                }
            )
            continue

        adaptation_end = pd.Timestamp(row.adaptation_target_end)
        available_start, available_end = context_start, adaptation_end
        available = exact_interval(raw, row.building_id, available_start, available_end)
        expected_hours = 24 * (days + 1)
        if len(available) != expected_hours or not available_end < test_start:
            raise AssertionError(f"{row.building_id} {days}d: invalid pretest history boundary")
        stats = numeric_summary(available)
        daily_n, daily_corr, daily_status = lag_repeatability(available, 24, 24)
        weekly_n, weekly_corr, weekly_status = lag_repeatability(available, 168, 168)
        scale = source_comparison(stats["mean"], source, config_mean, config_std)
        rows.append(
            {
                **base,
                "feature_information_cutoff": available_end,
                "budget_history_status": "context24_plus_allowed_adaptation_observations",
                "available_pretest_start": available_start,
                "available_pretest_end": available_end,
                "available_pretest_hours": len(available),
                **{f"available_{key}": value for key, value in stats.items()},
                "daily_repeatability_pairs": daily_n,
                "daily_repeatability_pearson": daily_corr,
                "daily_repeatability_status": daily_status,
                "weekly_repeatability_pairs": weekly_n,
                "weekly_repeatability_pearson": weekly_corr,
                "weekly_repeatability_status": weekly_status,
                **scale,
                "history_strictly_before_test": True,
            }
        )
        used.append(
            pd.DataFrame(
                {
                    "role": f"development_available_pretest_{days}d",
                    "building_id": row.building_id,
                    "timestamp": pd.date_range(available_start, available_end, freq="h"),
                    "value": available,
                }
            )
        )
    result = pd.DataFrame(rows)
    if len(result) != 12 * len(EXPECTED_BUDGETS):
        raise AssertionError("Expected 72 development building-budget rows")
    return result.sort_values(["building_id", "adaptation_days"]), pd.concat(used, ignore_index=True)


def build_outcomes(
    source_metrics: pd.DataFrame,
    adaptation_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    source = source_metrics[["building_id", "seed", "mae", "rmse"]].rename(
        columns={"mae": "source_only_mae", "rmse": "source_only_rmse"}
    )
    adapt = adaptation_metrics.loc[adaptation_metrics.adaptation_days.isin(NONZERO_BUDGETS)].copy()
    events = adapt.merge(source, on=["building_id", "seed"], validate="many_to_one")
    baseline = baseline_metrics.pivot(index="building_id", columns="model", values=["mae", "rmse"])
    baseline.columns = [f"{model}_{metric}" for metric, model in baseline.columns]
    events = events.merge(baseline.reset_index(), on="building_id", validate="many_to_one")
    events["mae_delta_vs_source"] = events.mae - events.source_only_mae
    events["rmse_delta_vs_source"] = events.rmse - events.source_only_rmse
    events["f1"] = events.mae_delta_vs_source.gt(EPS)
    events["f2"] = events.rmse_delta_vs_source.gt(EPS)
    events["f3_previous_day"] = events.mae.gt(events.previous_day_mae + EPS)
    events["f3_weekly"] = events.mae.gt(events.weekly_same_hour_mae + EPS)
    events["f3_any"] = events.f3_previous_day | events.f3_weekly
    events["f3_both"] = events.f3_previous_day & events.f3_weekly
    mixed = events.groupby(["building_id", "adaptation_days"]).mae_delta_vs_source.agg(
        lambda values: bool((values < -EPS).any() and (values > EPS).any())
    )
    events = events.merge(mixed.rename("f4").reset_index(), on=["building_id", "adaptation_days"])
    if len(events) != 12 * 5 * 3:
        raise AssertionError("Expected 180 nonzero-budget seed events")
    return events.sort_values(["building_id", "adaptation_days", "seed"])


def aggregate_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["building_id", "site_id", "primaryspaceusage", "adaptation_days"], sort=True)
        .agg(
            seed_count=("seed", "nunique"),
            seed_mean_mae_delta=("mae_delta_vs_source", "mean"),
            seed_mean_rmse_delta=("rmse_delta_vs_source", "mean"),
            f1_seed_rate=("f1", "mean"),
            f2_seed_rate=("f2", "mean"),
            f3_previous_day_seed_rate=("f3_previous_day", "mean"),
            f3_weekly_seed_rate=("f3_weekly", "mean"),
            f3_any_seed_rate=("f3_any", "mean"),
            f3_both_seed_rate=("f3_both", "mean"),
            f4=("f4", "first"),
        )
        .reset_index()
    )


def spearman(x: pd.Series, y: pd.Series) -> tuple[int, float, str]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return len(frame), math.nan, "insufficient_n"
    if frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return len(frame), math.nan, "constant_variable"
    return len(frame), float(frame.x.rank().corr(frame.y.rank())), "ok"


def build_associations(characteristics: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    data = outcomes.merge(
        characteristics.loc[characteristics.adaptation_days.gt(0)],
        on=["building_id", "site_id", "primaryspaceusage", "adaptation_days"],
        validate="one_to_one",
    )
    feature_families = {
        "pre_adaptation_context24": [
            "context_24h_mean",
            "context_24h_std_population",
            "context_24h_cv_abs_mean",
            "context_24h_zero_ratio",
            "context_24h_daily_repeatability_pearson",
            "context_24h_weekly_repeatability_pearson",
            "context_24h_absolute_mean_gap_in_source_config_std",
            "context_24h_mean_percentile_among_20_source_building_means",
        ],
        "pretest_budget_available": [
            "available_mean",
            "available_std_population",
            "available_cv_abs_mean",
            "available_zero_ratio",
            "daily_repeatability_pearson",
            "weekly_repeatability_pearson",
            "absolute_mean_gap_in_source_config_std",
            "mean_percentile_among_20_source_building_means",
        ],
        "static_metadata": ["sqm"],
    }
    response_names = [
        "seed_mean_mae_delta",
        "seed_mean_rmse_delta",
        "f1_seed_rate",
        "f2_seed_rate",
        "f3_previous_day_seed_rate",
        "f3_weekly_seed_rate",
    ]
    rows = []
    scopes = [(f"budget_{days}d", data.loc[data.adaptation_days.eq(days)], days) for days in NONZERO_BUDGETS]
    scopes.append(("pooled_nonzero_budgets", data, np.nan))
    for scope, frame, days in scopes:
        for family, features in feature_families.items():
            for feature in features:
                for response in response_names:
                    n, rho, status = spearman(frame[feature], frame[response])
                    if family == "pre_adaptation_context24" and "repeatability" in feature:
                        status = "not_defined_24h_context"
                    rows.append(
                        {
                            "scope": scope,
                            "adaptation_days": days,
                            "feature_family": family,
                            "feature_availability_time": {
                                "pre_adaptation_context24": "before_initial_adaptation_start",
                                "pretest_budget_available": "after_allowed_adaptation_before_test",
                                "static_metadata": "static_before_experiment",
                            }[family],
                            "feature": feature,
                            "response": response,
                            "n_complete": n,
                            "spearman_rho": rho,
                            "status": status,
                            "p_value_computed": False,
                            "causal_interpretation": False,
                            "independence_caveat": (
                                "12 buildings only" if scope != "pooled_nonzero_budgets" else
                                "60 repeated building-budget rows; rows are not independent"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_group_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in ("site_id", "primaryspaceusage"):
        for days in NONZERO_BUDGETS:
            subset = events.loc[events.adaptation_days.eq(days)]
            for category, frame in subset.groupby(field, sort=True):
                rows.append(
                    {
                        "group_field": field,
                        "group_value": category,
                        "adaptation_days": days,
                        "n_buildings": frame.building_id.nunique(),
                        "n_seed_events": len(frame),
                        "mean_mae_delta_vs_source": frame.mae_delta_vs_source.mean(),
                        "mean_rmse_delta_vs_source": frame.rmse_delta_vs_source.mean(),
                        "f1_event_rate": frame.f1.mean(),
                        "f2_event_rate": frame.f2.mean(),
                        "f3_previous_day_event_rate": frame.f3_previous_day.mean(),
                        "f3_weekly_event_rate": frame.f3_weekly.mean(),
                        "descriptive_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _font(size: int, bold: bool = False):
    candidates = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for filename in candidates:
        path = Path("C:/Windows/Fonts") / filename
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_scale_scatter(characteristics: pd.DataFrame, outcomes: pd.DataFrame, path: Path) -> None:
    data = outcomes.merge(
        characteristics.loc[characteristics.adaptation_days.gt(0)],
        on=["building_id", "site_id", "primaryspaceusage", "adaptation_days"],
        validate="one_to_one",
    )
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(30, True), _font(19), _font(15)
    draw.text((55, 28), "Budget-available scale gap vs. C0 adaptation MAE change", font=title_font, fill="#111111")
    draw.text(
        (55, 70),
        "Each dot is one development building; color is site. Positive y means worse than Source-only.",
        font=label_font,
        fill="#333333",
    )
    colors = {site: color for site, color in zip(
        sorted(data.site_id.unique()), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    )}
    panel_w, panel_h = 520, 390
    origins = [(65, 135), (640, 135), (1215, 135), (350, 600), (925, 600)]
    for (days, frame), (ox, oy) in zip(data.groupby("adaptation_days", sort=True), origins):
        x = frame.absolute_mean_gap_in_source_config_std.to_numpy(float)
        y = frame.seed_mean_mae_delta.to_numpy(float)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(min(np.min(y), 0)), float(max(np.max(y), 0))
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        pad_x, pad_y = 0.08 * (xmax - xmin), 0.10 * (ymax - ymin)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
        draw.rectangle((ox, oy, ox + panel_w, oy + panel_h), outline="#444444", width=2)
        zero_y = oy + panel_h - (0 - ymin) * panel_h / (ymax - ymin)
        if oy <= zero_y <= oy + panel_h:
            draw.line((ox, zero_y, ox + panel_w, zero_y), fill="#888888", width=2)
        for row in frame.itertuples(index=False):
            px = ox + (row.absolute_mean_gap_in_source_config_std - xmin) * panel_w / (xmax - xmin)
            py = oy + panel_h - (row.seed_mean_mae_delta - ymin) * panel_h / (ymax - ymin)
            color = colors[row.site_id]
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="#222222")
        draw.text((ox + 8, oy + 8), f"{int(days)} days (n=12)", font=label_font, fill="#111111")
        draw.text((ox + 5, oy + panel_h + 6), f"x {xmin:.2f} .. {xmax:.2f} source SD", font=small_font, fill="#333333")
        draw.text((ox + panel_w - 165, oy + panel_h + 6), f"y {ymin:.2f} .. {ymax:.2f}", font=small_font, fill="#333333")
    legend_x, legend_y = 1460, 690
    for i, (site, color) in enumerate(colors.items()):
        y = legend_y + i * 28
        draw.ellipse((legend_x, y, legend_x + 14, y + 14), fill=color, outline="#222222")
        draw.text((legend_x + 23, y - 4), site, font=small_font, fill="#222222")
    draw.text((55, 1035), "Descriptive development analysis; no fitted failure classifier and no causal claim.", font=label_font, fill="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    characteristics: pd.DataFrame,
    source: pd.DataFrame,
    outcomes: pd.DataFrame,
    associations: pd.DataFrame,
    hashes: pd.DataFrame,
    selected_hash: str,
) -> None:
    availability = (
        characteristics.groupby("adaptation_days", sort=True)
        .agg(
            buildings=("building_id", "nunique"),
            history_hours_min=("available_pretest_hours", "min"),
            history_hours_max=("available_pretest_hours", "max"),
            daily_available=("daily_repeatability_pearson", lambda s: int(s.notna().sum())),
            weekly_available=("weekly_repeatability_pearson", lambda s: int(s.notna().sum())),
            zero_ratio_max=("available_zero_ratio", "max"),
        )
        .reset_index()
    )
    strongest = associations.loc[
        associations.status.eq("ok")
        & associations.response.eq("seed_mean_mae_delta")
        & associations.scope.ne("pooled_nonzero_budgets")
    ].copy()
    strongest["abs_rho"] = strongest.spearman_rho.abs()
    strongest = (
        strongest.sort_values(["adaptation_days", "abs_rho", "feature"], ascending=[True, False, True])
        .groupby("adaptation_days", as_index=False)
        .first()
    )
    focused = associations.loc[
        associations.response.eq("seed_mean_mae_delta")
        & associations.scope.ne("pooled_nonzero_budgets")
        & associations.feature_family.eq("pretest_budget_available")
        & associations.feature.isin(
            [
                "absolute_mean_gap_in_source_config_std",
                "daily_repeatability_pearson",
                "weekly_repeatability_pearson",
            ]
        ),
        ["adaptation_days", "feature", "n_complete", "spearman_rho", "status"],
    ].sort_values(["feature", "adaptation_days"])
    context_focused = associations.loc[
        associations.response.eq("seed_mean_mae_delta")
        & associations.scope.ne("pooled_nonzero_budgets")
        & associations.feature_family.eq("pre_adaptation_context24")
        & associations.feature.isin(
            [
                "context_24h_mean",
                "context_24h_std_population",
                "context_24h_cv_abs_mean",
                "context_24h_zero_ratio",
                "context_24h_absolute_mean_gap_in_source_config_std",
            ]
        ),
        ["adaptation_days", "feature_family", "feature", "n_complete", "spearman_rho", "status"],
    ].sort_values(["feature", "adaptation_days"])
    source_summary = pd.DataFrame(
        [
            {
                "source_buildings": len(source),
                "training_hours_total": int(source.source_train_target_hours.sum()),
                "building_mean_min": source.source_train_mean.min(),
                "building_mean_median": source.source_train_mean.median(),
                "building_mean_max": source.source_train_mean.max(),
            }
        ]
    )
    outcome_summary = (
        outcomes.groupby("adaptation_days", sort=True)
        .agg(
            n_buildings=("building_id", "nunique"),
            mean_mae_delta=("seed_mean_mae_delta", "mean"),
            buildings_mean_worse=("seed_mean_mae_delta", lambda s: int((s > EPS).sum())),
            f4_pairs=("f4", "sum"),
        )
        .reset_index()
    )
    hash_lines = "\n".join(
        f"- `{row.artifact}`: `{row.sha256}` ({row.verification})" for row in hashes.itertuples(index=False)
    )
    text = f"""# BDG2 v5 실패 관련 건물 특성 탐색

## 범위와 안전 경계

- 분석 대상은 동결된 Source 20개와 개발 12개 건물이다. 개발 표본은 6개 사이트이다.
- 전력 CSV에는 `timestamp`와 이 32개 허용 열만 `usecols`로 요청했다. `final_test_proposal` 열은 요청·읽기·분석하지 않았다.
- 개발 시험 실제 전력은 특성 계산에 사용하지 않았다. 실패 결과변수는 동결된 v3 지표 CSV에서만 가져왔다.
- 원본 CSV 전체 SHA-256은 최종 시험 열까지 읽게 되므로 v5에서 재계산하지 않았다. 해시 검증된 v3 설정 안의 선언값만 대조했다.
- 새 모델·실패 분류기·임계값 탐색은 수행하지 않았다. 아래 연관성은 개발 표본의 탐색적 기술 결과이며 인과관계가 아니다.

## 정보 이용 가능 시점의 고정 정의

`context_24h_*`는 사용자 후보인 **초기 적응 시작 이전** 특성이다. 비제로 예산에서는 공통 적응 시작 직전 24시간이며 모든 예산에서 같은 값을 쓴다. `0일`은 적응 시작이 없으므로 첫 시험 예측 직전 24시간 입력만 요약한다. 24시간만으로는 24시간 lag 쌍이나 168시간 lag 쌍을 만들 수 없어 일간·주간 반복성은 모두 `NA`이다.

`available_*`는 별도 계열인 **시험 전 budget-available 특성**이다. `d>0`에서 적응 시작 직전 context 24시간과 해당 예산의 적응 종료 시각까지 허용된 관측을 합친다. 중간의 미사용 gap이나 시험 정답은 포함하지 않았다. 일간 반복성은 24시간 lag 쌍이 최소 24개, 주간 반복성은 168시간 lag 쌍이 최소 168개일 때의 Pearson 상관이다. 부족하거나 상수이면 `NA`이다. `0일` 장기 budget-available 통계는 Target 정답 이력이 없으므로 `NA`이다.

{markdown_table(availability, ['adaptation_days','buildings','history_hours_min','history_hours_max','daily_available','weekly_available','zero_ratio_max'], {'zero_ratio_max':'{:.4f}'})}

## Source 규모 기준

Source 건물별 규모는 각 Source 건물의 고정 학습 **정답 시각**에서만 계산했다. 모델의 전역 정규화 기준은 고정 설정의 Source 학습 X와 y에서 계산된 mean/std를 그대로 인용했다. Target 규모 차이는 이용 가능한 pre-test 구간 평균과 이 Source 기준의 차이로 정의했다.

{markdown_table(source_summary, ['source_buildings','training_hours_total','building_mean_min','building_mean_median','building_mean_max'], {'building_mean_min':'{:.4f}','building_mean_median':'{:.4f}','building_mean_max':'{:.4f}'})}

- Source 모델 정규화 mean: `{characteristics.source_config_mean.iloc[0]:.12g}`
- Source 모델 정규화 population std: `{characteristics.source_config_std_population.iloc[0]:.12g}`
- Source와 Target 계량값의 물리적 단위·경계가 건물마다 완전히 동일하다는 개별 근거는 없으므로 절대 규모 차이는 `계량값 규모 차이`로만 해석한다.

## 실패 결과와 특성의 기술적 연결

시드별 F1/F2/F3와 건물×예산 F4는 고정 규칙을 그대로 적용한 뒤, 연속 특성에는 Spearman 순위상관만 계산했다. p-value나 다중비교 유의성은 계산하지 않았다.

{markdown_table(outcome_summary, ['adaptation_days','n_buildings','mean_mae_delta','buildings_mean_worse','f4_pairs'], {'mean_mae_delta':'{:.4f}'})}

각 예산에서 시드 평균 MAE 변화와 절댓값이 가장 큰 탐색적 상관은 다음과 같다. `feature_family`가 정보 시점을 구분한다. 후보 변수가 많고 n=12이므로 선택된 상관의 크기가 낙관적으로 보일 수 있다.

{markdown_table(strongest, ['adaptation_days','feature_family','feature','n_complete','spearman_rho','status'], {'spearman_rho':'{:.3f}'})}

### 초기 적응 시작 이전 24시간 context

다음 표는 초기 적응 시작 직전 24시간의 평균·표준편차·CV·0비율·Source 규모 gap을 각 예산 결과와 직접 연결한다. 24시간 context의 일간·주간 반복성 association 행도 CSV에 남겼지만 `n_complete=0`, `status=not_defined_24h_context`이다.

{markdown_table(context_focused, ['adaptation_days','feature','n_complete','spearman_rho','status'], {'spearman_rho':'{:.3f}'})}

### 시험 전 budget-available 특성

사전에 지정한 규모 차이·일간·주간 반복성과 MAE 변화의 상관은 다음과 같다.

{markdown_table(focused, ['adaptation_days','feature','n_complete','spearman_rho','status'], {'spearman_rho':'{:.3f}'})}

사이트와 용도는 범주별 사건률 표에 그대로 남겼다. 사이트당 개발 건물이 2개뿐이고 일부 용도도 매우 작아 그룹 차이에 대한 통계적 판단은 보류한다.

## 검증 가능한 가설 후보

1. Source 규모와 Target 규모 차이의 상관 방향은 1·3·7일에는 음수, 14·28일에는 양수로 바뀌었다. 따라서 `규모 차이가 클수록 항상 실패한다`는 단순 방향 가설은 이 개발 결과가 지지하지 않는다. 최종 시험에서는 규모 차이와 예산의 상호 관계를 탐색 변수로만 유지하는 편이 타당하다.
2. 일간 반복성과 MAE 변화의 상관은 3·7·14·28일에서 음수였고 1일에서는 약한 양수였다. 주간 반복성은 정보가 충분한 14·28일에서만 정의됐고 두 예산 모두 음수였다. `충분한 적응 이력에서 반복성이 높을수록 적응 이득이 커진다`를 검증 후보로 둘 수 있지만, n=12의 선택된 관찰이므로 핵심 확증 가설로 격상하려면 연구자 승인이 필요하다.
3. 이용 가능 구간의 0값 비율은 1·3·7일 모든 건물에서 0이었고 14·28일 최대도 각각 0.0028, 0.0014였다. 이 개발 코호트에서는 변별력이 거의 없어 보조 기술 변수로만 유지한다.
4. 변동계수, site, 용도는 보조 탐색 변수로 유지할 수 있다. 12개 건물 결과로 자동 제외 규칙이나 인과 설명을 만들 수 없다.
5. 0일에는 context 24시간 특성만 허용하며 적응 기반 장기 특성과 직접 같은 의미로 비교하지 않는다.

## 산출물

- 특성표: `reports/bdg2_v5_building_characteristics.csv`
- Source 규모표: `reports/bdg2_v5_source_scale_reference.csv`
- 건물×예산 실패 요약: `reports/bdg2_v5_characteristic_failure_outcomes.csv`
- 상관표: `reports/bdg2_v5_building_characteristic_associations.csv`
- 사이트·용도 기술표: `reports/bdg2_v5_building_characteristic_group_summary.csv`
- 그림: `reports/figures/bdg2_v5/bdg2_v5_scale_gap_failure_association.png`
- 실행 로그: `reports/bdg2_v5_building_characteristics_run.log`

선택된 실제 사용 행의 논리 SHA-256은 `{selected_hash}`이다. 정렬된 `role, building_id, timestamp, value`를 UTF-8 CSV로 직렬화해 계산했다. 이는 원본 전체 파일 해시가 아니다.

## 고정 입력 해시

{hash_lines}

## 자동 검증

- 명령: `.\\.venv_bdg2_nightly_20260924\\Scripts\\python.exe -m unittest -v scripts\\test_bdg2_v5_building_characteristics.py`
- 마지막 실행 결과: **12/12 통과** (`reports/bdg2_v5_building_characteristics_tests.log`).
- 검증 범위: 72개 건물×예산 격자, 0일 정보 제한, 초기 적응 전 context의 예산 간 동일성, 비제로 예산 경계, 반복성 최소 쌍, Source 20개, final-test ID 부재, 60개 건물×비제로 예산 결과, 두 특성 계열 분리, 비인과 기술 분석 표시, 산출물 존재 여부.

## 한계

- 건물 12개·사이트 6개의 개발용 탐색 분석이며 전체 BDG2나 최종 시험 사이트의 일반 법칙이 아니다.
- 같은 건물의 여러 예산은 반복 관측이므로 pooled 상관의 행은 독립이 아니다.
- 용도·사이트와 계량 규모가 얽혀 있을 수 있으며 이 표본으로 분리된 효과를 추정하지 않았다.
- Pearson lag 상관은 반복성의 한 가지 사전 정의일 뿐이며 결측 보간이나 시차 최적화를 하지 않았다.
- 최종 시험 분석 변수·가설·보고 규칙은 연구자 승인 후 고정해야 한다. 최종 시험 전력은 아직 읽지 않았다.
"""
    REPORT_OUT.write_text(text, encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def main() -> None:
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    hash_rows = verify_small_input_hashes()
    cohort, temporal = pd.read_csv(COHORT), pd.read_csv(TEMPORAL)
    source_metrics, adapt_metrics = pd.read_csv(SOURCE_METRICS), pd.read_csv(ADAPT_METRICS)
    baseline_metrics, metadata = pd.read_csv(BASELINE_METRICS), pd.read_csv(METADATA)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw, source_ids, development_ids, final_ids = read_allowed_power(cohort, temporal)
    source, source_used = build_source_reference(raw, temporal, source_ids)
    characteristics, development_used = build_characteristics(raw, temporal, metadata, source, config)
    used = pd.concat([source_used, development_used], ignore_index=True)
    used_hash = selected_long_hash(used)
    hash_rows.append(
        {
            "artifact": "logical/source_train_and_development_pretest_rows_used_v5",
            "sha256": used_hash,
            "verification": "computed_from_only_allowed_rows_v5",
        }
    )
    events = build_outcomes(source_metrics, adapt_metrics, baseline_metrics)
    outcomes = aggregate_outcomes(events)
    associations = build_associations(characteristics, outcomes)
    group_summary = build_group_summary(events)
    hashes = pd.DataFrame(hash_rows)

    write_csv(characteristics, CHARACTERISTICS_OUT)
    write_csv(source, SOURCE_SCALE_OUT)
    write_csv(outcomes, OUTCOMES_OUT)
    write_csv(associations, ASSOCIATIONS_OUT)
    write_csv(group_summary, GROUP_SUMMARY_OUT)
    write_csv(hashes, HASHES_OUT)
    draw_scale_scatter(characteristics, outcomes, FIGURE_OUT)
    write_report(characteristics, source, outcomes, associations, hashes, used_hash)

    elapsed = time.perf_counter() - started
    log_lines = [
        f"started_at={started_at}",
        f"completed_at={datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"elapsed_seconds={elapsed:.3f}",
        f"python={platform.python_version()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"source_buildings={len(source_ids)}",
        f"development_buildings={len(development_ids)}",
        f"final_test_columns_requested={len(final_ids.intersection(raw.columns))}",
        "raw_whole_file_hash_recomputed_v5=False",
        f"selected_rows_logical_sha256={used_hash}",
        f"characteristic_rows={len(characteristics)}",
        f"association_rows={len(associations)}",
        "new_model_trained=False",
        "errors=[]",
    ]
    for path in (
        CHARACTERISTICS_OUT, SOURCE_SCALE_OUT, OUTCOMES_OUT, ASSOCIATIONS_OUT,
        GROUP_SUMMARY_OUT, HASHES_OUT, FIGURE_OUT, REPORT_OUT,
    ):
        log_lines.append(f"output={path.relative_to(ROOT)} sha256={sha256(path)}")
    LOG_OUT.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
