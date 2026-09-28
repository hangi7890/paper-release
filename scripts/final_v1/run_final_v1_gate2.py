"""Single write-once final_v1 C0 evaluation on the frozen final cohort.

This script deliberately refuses a second run.  It loads the already selected
source checkpoints and source-only normalization, evaluates both frozen naive
baselines, and applies the unchanged output-head-only adaptation procedure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from bdg2_multibuilding_common_v2 import build_windows, metric_row  # noqa: E402
from run_bdg2_stage6_c0_lstm_v3 import (  # noqa: E402
    C0LSTM,
    Config,
    adapt_head,
    predict,
)


REPORTS = ROOT / "reports"
SUPPORT = REPORTS / "final_v1"
MODELS_OUT = ROOT / "models" / "final_v1"
RAW = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
COHORT = REPORTS / "final_v1_cohort_manifest.csv"
TEMPORAL = REPORTS / "final_v1_temporal_manifest.csv"
PROTOCOL = REPORTS / "final_v1_protocol_manifest.md"
V5_PROTOCOL = REPORTS / "bdg2_v5_final_analysis_protocol_draft.md"
SOURCE_CONFIG = ROOT / "models" / "bdg2_stage6_v3" / "bdg2_stage6_c0_lstm_config_v3.json"

FINAL_PREDICTIONS = REPORTS / "final_v1_raw_predictions.csv"
PARTIAL_PREDICTIONS = SUPPORT / "raw_predictions.partial.csv"
EXECUTION_METRICS = SUPPORT / "gate2_execution_metrics.csv"
FAILURES = SUPPORT / "gate2_failures.csv"
RUN_CONFIG = SUPPORT / "gate2_run_config.json"
RUN_LOG = SUPPORT / "gate2_execution.log"
START_LOCK = SUPPORT / "GATE2_STARTED.lock"
COMPLETE_MARKER = SUPPORT / "GATE2_COMPLETE.json"

SEEDS = (20260924, 20260925, 20260926)
ADAPTATION_DAYS = (0, 1, 3, 7, 14, 28)
EXPECTED_SAMPLES = {0: 0, 1: 1, 3: 49, 7: 145, 14: 313, 28: 649}
SOURCE_MEAN = 167.0651976323119
SOURCE_STD = 354.328669571624

EXPECTED_HASHES = {
    RAW: "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418",
    COHORT: "8216847545f7814c12f3e2edb31523ac11da0ae19cc142985f736bbd7185d3ec",
    TEMPORAL: "1cb76d4cc7cef687fc747b2e0015112d85aec4e0f9e76271b28284f54421bc08",
    PROTOCOL: "3e51a1f6243c324e9bbf9f085ba363416ae5575fbe2dec023595166376e3eb44",
    V5_PROTOCOL: "359db1e34ba8fdc8d472267ca544b99005e4bcb92bacc77beedd4dd14942dee3",
    SOURCE_CONFIG: "c2cb99e8a01403175bc6240edd990a6a53d2244187038c41f1dba6727da1bd9f",
}
CHECKPOINT_HASHES = {
    20260924: "88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b",
    20260925: "202a610bc6f4c89c32335b674e80f372c03699c3df33c22e30bfae80b8f951f5",
    20260926: "708e3321db56f8828a5c59b8cfdbbf02c4b31feff9d1659d1be55d7a61e20d89",
}

PREDICTION_COLUMNS = [
    "building_id",
    "site_id",
    "model",
    "seed",
    "adaptation_days",
    "forecast_origin",
    "input_start",
    "input_end",
    "target_timestamp",
    "horizon_hour",
    "y_true",
    "y_pred",
    "prediction_available",
    "evaluation_mode",
    "status",
    "error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_log(message: str) -> None:
    line = f"{utc_now()} {message}"
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def checkpoint_path(seed: int) -> Path:
    return ROOT / "models" / "bdg2_stage6_v3" / f"bdg2_stage6_c0_lstm_seed_{seed}_v3.pt"


def preflight() -> tuple[pd.DataFrame, pd.DataFrame, dict, Config, torch.device]:
    for output in (
        FINAL_PREDICTIONS,
        PARTIAL_PREDICTIONS,
        EXECUTION_METRICS,
        FAILURES,
        RUN_CONFIG,
        RUN_LOG,
        START_LOCK,
        COMPLETE_MARKER,
    ):
        if output.exists():
            raise FileExistsError(f"Write-once Gate 2 refuses existing artifact: {output}")
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"Frozen input changed: {path} {actual} != {expected}")
    for seed, expected in CHECKPOINT_HASHES.items():
        actual = sha256(checkpoint_path(seed))
        if actual != expected:
            raise AssertionError(f"Source checkpoint changed for seed {seed}: {actual}")

    cohort = pd.read_csv(COHORT)
    selected = cohort.loc[cohort.selection_status.eq("selected_for_final_v1")].copy()
    selected = selected.sort_values(["site_id", "building_id"]).reset_index(drop=True)
    if len(selected) != 453 or selected.site_id.nunique() != 7:
        raise AssertionError("Gate 0 cohort count changed")
    temporal = pd.read_csv(TEMPORAL)
    if len(temporal) != 453 * 6 or not temporal.structure_status.eq("PASS").all():
        raise AssertionError("Gate 1 temporal structure is not fully PASS")
    if set(selected.building_id) != set(temporal.building_id):
        raise AssertionError("Gate 0/Gate 1 building IDs differ")

    source_cfg = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if source_cfg["status"] != "complete" or tuple(source_cfg["seeds"]) != SEEDS:
        raise AssertionError("Source C0 run is not the frozen completed three-seed run")
    if source_cfg["source_normalization"]["mean"] != SOURCE_MEAN:
        raise AssertionError("Source normalization mean changed")
    if source_cfg["source_normalization"]["std_population"] != SOURCE_STD:
        raise AssertionError("Source normalization std changed")
    config = Config(**source_cfg["config"])
    expected_config = Config()
    if config != expected_config:
        raise AssertionError("Frozen C0 architecture or optimizer settings changed")
    if tuple(source_cfg["feature_names"]) != (
        "past_electricity_normalized",
        "target_hour_sin",
        "target_hour_cos",
        "target_weekday_sin",
        "target_weekday_cos",
        "target_weekend",
    ):
        raise AssertionError("C0 feature definition changed")

    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 10 * 1024**3:
        raise OSError(f"Less than 10 GiB free before Gate 2: {free_bytes}")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen final neural run requires the available RTX 3080 CUDA device")
    return selected, temporal, source_cfg, config, torch.device("cuda")


def final_temporal_row(temporal: pd.DataFrame, building_id: str, days: int) -> pd.Series:
    rows = temporal.loc[
        temporal.building_id.eq(building_id) & temporal.adaptation_days.eq(days)
    ]
    if len(rows) != 1:
        raise AssertionError(f"{building_id}/{days}d: expected one frozen temporal row")
    return rows.iloc[0]


def daily_test_windows(raw: pd.DataFrame, temporal: pd.DataFrame, building_id: str):
    row = final_temporal_row(temporal, building_id, 0)
    starts = pd.date_range(
        pd.Timestamp(row.test_target_start),
        pd.Timestamp(row.test_target_end) - pd.Timedelta(hours=23),
        freq="24h",
    )
    x, y, meta = build_windows(raw, building_id, starts)
    if x.shape != (28, 24) or y.shape != (28, 24) or len(meta) != 28:
        raise AssertionError(f"{building_id}: invalid frozen daily shapes {x.shape}/{y.shape}")
    return x, y, meta


def adaptation_windows(
    raw: pd.DataFrame, temporal: pd.DataFrame, building_id: str, days: int
):
    if days == 0:
        return np.empty((0, 24)), np.empty((0, 24)), pd.DataFrame()
    row = final_temporal_row(temporal, building_id, days)
    starts = pd.date_range(
        pd.Timestamp(row.adaptation_target_start),
        pd.Timestamp(row.adaptation_target_end) - pd.Timedelta(hours=23),
        freq="h",
    )
    x, y, meta = build_windows(raw, building_id, starts)
    expected = EXPECTED_SAMPLES[days]
    if x.shape != (expected, 24) or y.shape != (expected, 24) or len(meta) != expected:
        raise AssertionError(f"{building_id}/{days}d: {len(meta)} samples != {expected}")
    return x, y, meta


def weekly_prediction(raw: pd.DataFrame, building_id: str, metadata: pd.DataFrame) -> np.ndarray:
    series = pd.Series(
        pd.to_numeric(raw[building_id], errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(raw.timestamp),
    )
    rows = []
    for start in pd.to_datetime(metadata.target_start):
        target_times = pd.date_range(start, periods=24, freq="h")
        values = series.reindex(target_times - pd.Timedelta(days=7)).to_numpy(float)
        rows.append(values)
    result = np.vstack(rows)
    if result.shape != (28, 24) or not np.isfinite(result).all():
        raise AssertionError(f"{building_id}: weekly_same_hour unavailable")
    return result


def prediction_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_id: str,
    model: str,
    seed: int | None,
    days: int | None,
    status: str = "ok",
    error: str = "",
) -> pd.DataFrame:
    meta = metadata.reset_index(drop=True)
    if y_true.shape != (28, 24) or y_pred.shape != (28, 24):
        raise AssertionError(f"Unexpected prediction shapes {y_true.shape}/{y_pred.shape}")
    target_starts = pd.to_datetime(meta.target_start)
    target_timestamp = np.concatenate(
        [pd.date_range(start, periods=24, freq="h").to_numpy() for start in target_starts]
    )
    pred_flat = np.asarray(y_pred, dtype=float).reshape(-1)
    return pd.DataFrame(
        {
            "building_id": np.repeat(meta.building_id.to_numpy(), 24),
            "site_id": site_id,
            "model": model,
            "seed": "" if seed is None else seed,
            "adaptation_days": "" if days is None else days,
            "forecast_origin": np.repeat(meta.forecast_origin.to_numpy(), 24),
            "input_start": np.repeat(meta.input_start.to_numpy(), 24),
            "input_end": np.repeat(meta.input_end.to_numpy(), 24),
            "target_timestamp": pd.DatetimeIndex(target_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hour": np.tile(np.arange(1, 25), len(meta)),
            "y_true": np.asarray(y_true, dtype=float).reshape(-1),
            "y_pred": pred_flat,
            "prediction_available": np.isfinite(pred_flat),
            "evaluation_mode": "daily_nonoverlap",
            "status": status,
            "error": error,
        },
        columns=PREDICTION_COLUMNS,
    )


def write_prediction_chunk(frame: pd.DataFrame, first: bool) -> None:
    frame.to_csv(
        PARTIAL_PREDICTIONS,
        mode="w" if first else "a",
        header=first,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
        quoting=csv.QUOTE_MINIMAL,
    )


def execution_metric(
    building: pd.Series,
    model: str,
    seed: int | None,
    days: int | None,
    adaptation_samples: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    status: str,
    error: str = "",
    adaptation_seed: int | None = None,
    adaptation_final_loss: float = math.nan,
    adaptation_batch_size: int = 0,
    oom_fallback_count: int = 0,
) -> dict:
    metrics = metric_row(y_true, y_pred)
    return {
        "building_id": building.building_id,
        "site_id": building.site_id,
        "primaryspaceusage": building.primaryspaceusage,
        "model": model,
        "seed": "" if seed is None else seed,
        "adaptation_days": "" if days is None else days,
        "adaptation_samples": adaptation_samples,
        "adaptation_seed": "" if adaptation_seed is None else adaptation_seed,
        "adaptation_epochs": 0 if not days else 50,
        "adaptation_learning_rate": 0 if not days else 5e-4,
        "adaptation_batch_size": adaptation_batch_size,
        "adaptation_final_train_mse_normalized": adaptation_final_loss,
        "oom_fallback_count": oom_fallback_count,
        "test_windows": 28,
        "n_values": metrics["n_values"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "status": status,
        "error": error,
    }


def load_raw(selected: pd.DataFrame) -> pd.DataFrame:
    ids = selected.building_id.tolist()
    header = pd.read_csv(RAW, nrows=0).columns.tolist()
    missing = sorted(set(ids).difference(header))
    if missing:
        raise AssertionError(f"Final columns missing: {missing}")
    raw = pd.read_csv(RAW, usecols=["timestamp", *ids])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    if raw.timestamp.duplicated().any() or not raw.timestamp.is_monotonic_increasing:
        raise AssertionError("Raw timestamps are duplicated or unsorted")
    return raw


def main() -> None:
    SUPPORT.mkdir(parents=True, exist_ok=True)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    selected, temporal, source_cfg, config, device = preflight()
    started = time.time()
    lock_payload = {
        "started_utc": utc_now(),
        "authorization": "user explicitly authorized single final_v1 execution",
        "cohort_sha256": EXPECTED_HASHES[COHORT],
        "temporal_sha256": EXPECTED_HASHES[TEMPORAL],
        "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "v5_protocol_sha256": EXPECTED_HASHES[V5_PROTOCOL],
        "source_config_sha256": EXPECTED_HASHES[SOURCE_CONFIG],
        "seeds": list(SEEDS),
        "budgets": list(ADAPTATION_DAYS),
    }
    with START_LOCK.open("x", encoding="utf-8") as handle:
        json.dump(lock_payload, handle, ensure_ascii=False, indent=2)
    append_log("Gate2 write-once lock created after frozen-input preflight")

    run_config = {
        **lock_payload,
        "status": "running",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "source_normalization": {"mean": SOURCE_MEAN, "std_population": SOURCE_STD},
        "config": source_cfg["config"],
        "source_checkpoints": {
            str(seed): {"path": str(checkpoint_path(seed)), "sha256": CHECKPOINT_HASHES[seed]}
            for seed in SEEDS
        },
        "adaptation_seed_rule": "base_seed + (site_then_building_sorted_index+1)*10000 + days*100",
        "building_order": selected.building_id.tolist(),
        "completed_baseline_buildings": 0,
        "completed_neural_building_seed_pairs": 0,
        "errors": [],
    }
    save_json(RUN_CONFIG, run_config)
    metric_rows: list[dict] = []
    failure_rows: list[dict] = []
    first_chunk = True
    try:
        append_log(f"Loading exactly {len(selected)} Gate1-PASS final power columns")
        raw = load_raw(selected)
        append_log(f"Raw loaded rows={len(raw)} columns={len(raw.columns)}")

        # Baselines are deterministic and have no neural seed.
        for index, building in selected.iterrows():
            test_x, test_y, test_meta = daily_test_windows(raw, temporal, building.building_id)
            baseline_predictions = {
                "previous_day": test_x.copy(),
                "weekly_same_hour": weekly_prediction(raw, building.building_id, test_meta),
            }
            for model_name, pred in baseline_predictions.items():
                frame = prediction_frame(
                    test_meta, test_y, pred, building.site_id, model_name, None, None
                )
                write_prediction_chunk(frame, first_chunk)
                first_chunk = False
                metric_rows.append(
                    execution_metric(
                        building, model_name, None, None, 0, test_y, pred, "ok"
                    )
                )
            run_config["completed_baseline_buildings"] = index + 1
            if (index + 1) % 25 == 0 or index + 1 == len(selected):
                save_json(RUN_CONFIG, run_config)
                append_log(f"Baselines completed {index + 1}/{len(selected)} buildings")

        # Source-only and all five output-head adaptation budgets.
        for seed in SEEDS:
            checkpoint = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=True)
            if checkpoint["seed"] != seed:
                raise AssertionError(f"Checkpoint seed mismatch: {seed}")
            if checkpoint["source_mean"] != SOURCE_MEAN or checkpoint["source_std"] != SOURCE_STD:
                raise AssertionError(f"Checkpoint normalization mismatch: {seed}")
            if checkpoint["config"] != source_cfg["config"]:
                raise AssertionError(f"Checkpoint config mismatch: {seed}")
            base_state = {
                key: value.detach().cpu().clone()
                for key, value in checkpoint["model_state_dict"].items()
            }
            source_model = C0LSTM(config).to(device)
            source_model.load_state_dict(base_state)
            head_bundles: dict[int, dict] = {
                days: {
                    "seed": seed,
                    "adaptation_days": days,
                    "base_source_checkpoint": str(checkpoint_path(seed)),
                    "base_source_checkpoint_sha256": CHECKPOINT_HASHES[seed],
                    "source_mean": SOURCE_MEAN,
                    "source_std": SOURCE_STD,
                    "adaptation_learning_rate": config.adaptation_learning_rate,
                    "adaptation_epochs": config.adaptation_epochs,
                    "adaptation_scope": "model.head parameters only; LSTM frozen",
                    "building_head_state_dict": {},
                }
                for days in ADAPTATION_DAYS
                if days > 0
            }
            for final_index, building in selected.iterrows():
                test_x, test_y, test_meta = daily_test_windows(raw, temporal, building.building_id)
                source_pred = predict(source_model, test_x, test_meta, SOURCE_MEAN, SOURCE_STD, device)
                for days in ADAPTATION_DAYS:
                    model_name = (
                        "c0_source_only_lstm" if days == 0 else "c0_head_only_adaptation_lstm"
                    )
                    adaptation_seed = None
                    sample_count = EXPECTED_SAMPLES[days]
                    final_loss = math.nan
                    used_batch = 0
                    oom_errors: list[str] = []
                    status = "ok"
                    error = ""
                    try:
                        if days == 0:
                            pred = source_pred
                        else:
                            adapt_x, adapt_y, adapt_meta = adaptation_windows(
                                raw, temporal, building.building_id, days
                            )
                            adaptation_seed = seed + (final_index + 1) * 10000 + days * 100
                            adapted_model, final_loss, used_batch, oom_errors = adapt_head(
                                config,
                                base_state,
                                adapt_x,
                                adapt_y,
                                adapt_meta,
                                SOURCE_MEAN,
                                SOURCE_STD,
                                device,
                                adaptation_seed,
                            )
                            pred = predict(
                                adapted_model,
                                test_x,
                                test_meta,
                                SOURCE_MEAN,
                                SOURCE_STD,
                                device,
                            )
                            head_bundles[days]["building_head_state_dict"][building.building_id] = {
                                "adaptation_seed": adaptation_seed,
                                "adaptation_samples": sample_count,
                                "head.weight": adapted_model.head.weight.detach().cpu().clone(),
                                "head.bias": adapted_model.head.bias.detach().cpu().clone(),
                            }
                            del adapted_model
                        frame = prediction_frame(
                            test_meta,
                            test_y,
                            pred,
                            building.site_id,
                            model_name,
                            seed,
                            days,
                            status,
                            error,
                        )
                    except Exception as exc:
                        status = "execution_error"
                        error = repr(exc)
                        pred = np.full((28, 24), np.nan, dtype=float)
                        failure_rows.append(
                            {
                                "seed": seed,
                                "building_id": building.building_id,
                                "site_id": building.site_id,
                                "adaptation_days": days,
                                "stage": "neural_evaluation_or_adaptation",
                                "error": error,
                            }
                        )
                        frame = prediction_frame(
                            test_meta,
                            test_y,
                            pred,
                            building.site_id,
                            model_name,
                            seed,
                            days,
                            status,
                            error,
                        )
                    write_prediction_chunk(frame, first_chunk)
                    first_chunk = False
                    metric_rows.append(
                        execution_metric(
                            building,
                            model_name,
                            seed,
                            days,
                            sample_count,
                            test_y,
                            pred,
                            status,
                            error,
                            adaptation_seed,
                            final_loss,
                            used_batch,
                            len(oom_errors),
                        )
                    )
                run_config["completed_neural_building_seed_pairs"] += 1
                if (final_index + 1) % 10 == 0 or final_index + 1 == len(selected):
                    pd.DataFrame(metric_rows).to_csv(
                        EXECUTION_METRICS, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g"
                    )
                    pd.DataFrame(
                        failure_rows,
                        columns=["seed", "building_id", "site_id", "adaptation_days", "stage", "error"],
                    ).to_csv(FAILURES, index=False, encoding="utf-8", lineterminator="\n")
                    save_json(RUN_CONFIG, run_config)
                    append_log(
                        f"Seed {seed}: neural completed {final_index + 1}/{len(selected)} buildings; "
                        f"failures={len(failure_rows)}"
                    )
            del source_model
            for days, bundle in head_bundles.items():
                path = MODELS_OUT / f"final_v1_head_only_seed_{seed}_{days}d.pt"
                if path.exists():
                    raise FileExistsError(f"Refusing checkpoint overwrite: {path}")
                torch.save(bundle, path)
                append_log(
                    f"Saved head bundle seed={seed} days={days} buildings={len(bundle['building_head_state_dict'])} "
                    f"sha256={sha256(path)}"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        expected_prediction_rows = len(selected) * (2 + len(SEEDS) * len(ADAPTATION_DAYS)) * 672
        actual_prediction_rows = sum(1 for _ in PARTIAL_PREDICTIONS.open("r", encoding="utf-8")) - 1
        if actual_prediction_rows != expected_prediction_rows:
            raise AssertionError(
                f"Raw prediction rows {actual_prediction_rows} != {expected_prediction_rows}"
            )
        pd.DataFrame(metric_rows).to_csv(
            EXECUTION_METRICS, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g"
        )
        pd.DataFrame(
            failure_rows,
            columns=["seed", "building_id", "site_id", "adaptation_days", "stage", "error"],
        ).to_csv(FAILURES, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(PARTIAL_PREDICTIONS, FINAL_PREDICTIONS)
        run_config.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.time() - started,
                "prediction_rows": expected_prediction_rows,
                "execution_metric_rows": len(metric_rows),
                "failure_count": len(failure_rows),
                "outputs": {
                    str(FINAL_PREDICTIONS): sha256(FINAL_PREDICTIONS),
                    str(EXECUTION_METRICS): sha256(EXECUTION_METRICS),
                    str(FAILURES): sha256(FAILURES),
                },
            }
        )
        save_json(RUN_CONFIG, run_config)
        save_json(COMPLETE_MARKER, run_config)
        append_log(
            f"Gate2 COMPLETE rows={expected_prediction_rows} metrics={len(metric_rows)} "
            f"failures={len(failure_rows)} elapsed={time.time() - started:.1f}s"
        )
    except Exception as exc:
        run_config["status"] = "failed"
        run_config["failed_utc"] = utc_now()
        run_config["elapsed_seconds"] = time.time() - started
        run_config["errors"].append(repr(exc))
        save_json(RUN_CONFIG, run_config)
        if metric_rows:
            pd.DataFrame(metric_rows).to_csv(
                EXECUTION_METRICS, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g"
            )
        pd.DataFrame(
            failure_rows,
            columns=["seed", "building_id", "site_id", "adaptation_days", "stage", "error"],
        ).to_csv(FAILURES, index=False, encoding="utf-8", lineterminator="\n")
        append_log(f"Gate2 FAILED: {repr(exc)}")
        raise


if __name__ == "__main__":
    main()
