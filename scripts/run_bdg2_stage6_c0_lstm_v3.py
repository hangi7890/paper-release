"""Train and evaluate the fixed C0 LSTM condition for BDG2 stage 6 v3."""

from __future__ import annotations

import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bdg2_multibuilding_common_v2 import (
    ADAPTATION_DAYS,
    COHORT,
    DEVELOPMENT,
    RAW,
    REPORTS,
    SEEDS,
    SOURCE,
    TEMPORAL,
    adaptation_windows,
    daily_test_windows,
    ensure_dirs,
    load_selected_power,
    metric_row,
    prediction_rows,
    save_json,
    selected_buildings,
    sha256,
    source_windows,
)
from bdg2_stage6_c0_common_v3 import C0_FEATURE_NAMES, make_c0_features


MODEL_DIR = ROOT_MODEL_DIR = Path(os.environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parent.parent))) / "models" / "bdg2_stage6_v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROTOCOL = REPORTS / "bdg2_stage6_c0_protocol_manifest_v3.csv"
SOURCE_METRICS = REPORTS / "bdg2_stage6_c0_source_only_metrics_v3.csv"
SOURCE_PREDICTIONS = REPORTS / "bdg2_stage6_c0_source_only_predictions_v3.csv"
ADAPT_METRICS = REPORTS / "bdg2_stage6_c0_adaptation_metrics_v3.csv"
ADAPT_PREDICTIONS = REPORTS / "bdg2_stage6_c0_adaptation_predictions_v3.csv"
HISTORY = REPORTS / "bdg2_stage6_c0_training_history_v3.csv"
FAILURES = REPORTS / "bdg2_stage6_c0_failures_v3.csv"
REVIEW = REPORTS / "bdg2_stage6_c0_lstm_results_v3.md"
CONFIG_PATH = MODEL_DIR / "bdg2_stage6_c0_lstm_config_v3.json"


@dataclass(frozen=True)
class Config:
    sequence_length: int = 24
    input_size: int = 6
    target_hours: int = 24
    hidden_size: int = 32
    num_layers: int = 1
    source_learning_rate: float = 1e-3
    source_batch_size: int = 256
    source_max_epochs: int = 100
    source_patience: int = 10
    source_min_delta: float = 1e-5
    adaptation_learning_rate: float = 5e-4
    adaptation_epochs: int = 50
    adaptation_batch_size: int = 64


class C0LSTM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, config.target_hours)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_source(raw: pd.DataFrame, source_ids: list[str]):
    parts = {"train": {"x": [], "y": [], "meta": []}, "validation": {"x": [], "y": [], "meta": []}}
    ranges = []
    for building_id in source_ids:
        building_parts = source_windows(raw, building_id)
        for phase in ("train", "validation"):
            x, y, meta = building_parts[phase]
            if len(meta) == 0:
                raise AssertionError(f"{building_id}: no {phase} samples")
            parts[phase]["x"].append(x)
            parts[phase]["y"].append(y)
            parts[phase]["meta"].append(meta)
            ranges.append(
                {
                    "building_id": building_id,
                    "phase": phase,
                    "samples": len(meta),
                    "target_start": meta.target_start.min(),
                    "target_end": meta.target_end.max(),
                }
            )
    return {
        phase: (
            np.concatenate(parts[phase]["x"]),
            np.concatenate(parts[phase]["y"]),
            pd.concat(parts[phase]["meta"], ignore_index=True),
        )
        for phase in ("train", "validation")
    }, pd.DataFrame(ranges)


def make_dataset(x, y, meta, mean, std):
    features = make_c0_features(x, meta, mean, std)
    targets = ((y - mean) / std).astype(np.float32)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))


def validation_loss(model, dataset, device):
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            total += float(torch.sum((model(features) - targets) ** 2).item())
            count += targets.numel()
    return total / count


def fit_source_once(config, train_dataset, validation_dataset, device, seed, batch_size):
    seed_everything(seed)
    model = C0LSTM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.source_learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    best_state, best_loss, best_epoch, stale = None, math.inf, 0, 0
    history = []
    for epoch in range(1, config.source_max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(features)
            count += len(features)
        train_loss = total / count
        val_loss = validation_loss(model, validation_dataset, device)
        improved = val_loss < best_loss - config.source_min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "batch_size": batch_size,
                "train_mse_normalized": train_loss,
                "source_validation_mse_normalized": val_loss,
                "is_best": improved,
            }
        )
        print(
            f"seed={seed} epoch={epoch} train={train_loss:.7g} val={val_loss:.7g} best={best_loss:.7g}",
            flush=True,
        )
        if stale >= config.source_patience:
            break
    if best_state is None:
        raise RuntimeError("No C0 source checkpoint produced")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def fit_source(config, train_dataset, validation_dataset, device, seed):
    oom_errors = []
    for batch_size in (config.source_batch_size, 128, 64):
        try:
            result = fit_source_once(
                config, train_dataset, validation_dataset, device, seed, batch_size
            )
            return (*result, batch_size, oom_errors)
        except torch.cuda.OutOfMemoryError as exc:
            oom_errors.append(f"batch={batch_size}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("Repeated C0 source CUDA OOM: " + " | ".join(oom_errors))


def predict(model, x, meta, mean, std, device):
    features = torch.from_numpy(make_c0_features(x, meta, mean, std)).to(device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            outputs.append(model(features[start : start + 2048]).cpu().numpy())
    return np.concatenate(outputs).astype(float) * std + mean


def adapt_head_once(config, base_state, x, y, meta, mean, std, device, seed, batch_size):
    seed_everything(seed)
    model = C0LSTM(config).to(device)
    model.load_state_dict(base_state)
    for parameter in model.lstm.parameters():
        parameter.requires_grad = False
    dataset = make_dataset(x, y, meta, mean, std)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.adaptation_learning_rate)
    criterion = nn.MSELoss()
    final_loss = math.nan
    for _ in range(config.adaptation_epochs):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
    return model, final_loss


def adapt_head(config, base_state, x, y, meta, mean, std, device, seed):
    oom_errors = []
    for batch_size in (config.adaptation_batch_size, 32, 16):
        try:
            model, loss = adapt_head_once(
                config, base_state, x, y, meta, mean, std, device, seed, batch_size
            )
            return model, loss, batch_size, oom_errors
        except torch.cuda.OutOfMemoryError as exc:
            oom_errors.append(f"batch={batch_size}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("Repeated C0 adaptation CUDA OOM: " + " | ".join(oom_errors))


def save_progress(source_rows, source_predictions, adapt_rows, adapt_predictions, histories, failures):
    pd.DataFrame(source_rows).to_csv(SOURCE_METRICS, index=False, encoding="utf-8-sig")
    if source_predictions:
        pd.concat(source_predictions, ignore_index=True).to_csv(
            SOURCE_PREDICTIONS, index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(adapt_rows).to_csv(ADAPT_METRICS, index=False, encoding="utf-8-sig")
    if adapt_predictions:
        pd.concat(adapt_predictions, ignore_index=True).to_csv(
            ADAPT_PREDICTIONS, index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(histories).to_csv(HISTORY, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        failures, columns=["seed", "building_id", "adaptation_days", "stage", "error"]
    ).to_csv(FAILURES, index=False, encoding="utf-8-sig")


def main() -> None:
    started = time.time()
    ensure_dirs()
    config = Config()
    selected = selected_buildings()
    source_ids = selected.query("alternative_group == @SOURCE").building_id.tolist()
    development = selected.query("alternative_group == @DEVELOPMENT").reset_index(drop=True)
    raw = load_selected_power(selected)
    source_parts, source_ranges = collect_source(raw, source_ids)
    train_x, train_y, train_meta = source_parts["train"]
    val_x, val_y, val_meta = source_parts["validation"]
    normalization_values = np.concatenate([train_x.reshape(-1), train_y.reshape(-1)])
    mean, std = float(normalization_values.mean()), float(normalization_values.std())
    if not np.isfinite(std) or std <= 0:
        raise AssertionError("Invalid C0 source normalization")
    train_dataset = make_dataset(train_x, train_y, train_meta, mean, std)
    validation_dataset = make_dataset(val_x, val_y, val_meta, mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    run_config = {
        "status": "running",
        "input_condition": "C0",
        "feature_names": list(C0_FEATURE_NAMES),
        "calendar_alignment": "row h pairs past power lag h with deterministic target calendar horizon h",
        "future_observed_values_used": False,
        "fixed_hyperparameters_from_v2_except_input_size": True,
        "config": asdict(config),
        "seeds": list(SEEDS),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device),
        "device_name": device_name,
        "source_buildings": source_ids,
        "development_buildings": development.building_id.tolist(),
        "source_training_samples": len(train_dataset),
        "source_validation_samples": len(validation_dataset),
        "source_ranges": source_ranges.to_dict(orient="records"),
        "source_normalization": {
            "mean": mean,
            "std_population": std,
            "computed_from": "source training past-power X and target y only",
            "calendar_features_scaled": False,
        },
        "hashes": {
            "raw": sha256(RAW),
            "cohort_manifest_v2": sha256(COHORT),
            "temporal_manifest_v2": sha256(TEMPORAL),
            "c0_protocol_v3": sha256(PROTOCOL),
        },
        "final_test_power_read": False,
        "completed_seeds": 0,
        "checkpoints": [],
        "errors": [],
    }
    save_json(CONFIG_PATH, run_config)
    source_rows, source_predictions = [], []
    adapt_rows, adapt_predictions = [], []
    histories, failures = [], []
    try:
        for seed in SEEDS:
            model, history, best_epoch, best_loss, batch_size, source_oom = fit_source(
                config, train_dataset, validation_dataset, device, seed
            )
            histories.extend(history)
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint = MODEL_DIR / f"bdg2_stage6_c0_lstm_seed_{seed}_v3.pt"
            torch.save(
                {
                    "model_state_dict": state,
                    "config": asdict(config),
                    "features": list(C0_FEATURE_NAMES),
                    "seed": seed,
                    "source_mean": mean,
                    "source_std": std,
                    "best_epoch": best_epoch,
                    "best_source_validation_mse_normalized": best_loss,
                    "source_buildings": source_ids,
                    "development_buildings_excluded_from_source_training": development.building_id.tolist(),
                },
                checkpoint,
            )
            run_config["checkpoints"].append(
                {
                    "seed": seed,
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "best_epoch": best_epoch,
                    "best_source_validation_mse_normalized": best_loss,
                    "batch_size": batch_size,
                    "oom_errors": source_oom,
                }
            )
            for dev_index, building in enumerate(development.itertuples(index=False)):
                try:
                    test_x, test_y, test_meta = daily_test_windows(raw, building.building_id)
                    if test_x.shape != (28, 24):
                        raise AssertionError(f"Invalid C0 test shape {test_x.shape}")
                    source_pred = predict(model, test_x, test_meta, mean, std, device)
                    source_row = {
                        "building_id": building.building_id,
                        "site_id": building.site_id,
                        "primaryspaceusage": building.primaryspaceusage,
                        "seed": seed,
                        "model": "c0_source_only_lstm",
                        "input_condition": "C0",
                        "adaptation_days": 0,
                        "adaptation_samples": 0,
                        "test_windows": len(test_meta),
                        "best_source_epoch": best_epoch,
                        "best_source_validation_mse_normalized": best_loss,
                    }
                    source_row.update(metric_row(test_y, source_pred))
                    source_rows.append(source_row)
                    source_predictions.append(
                        prediction_rows(
                            test_meta, test_y, source_pred, "c0_source_only_lstm", seed, 0
                        )
                    )
                    base_mae, base_rmse = source_row["mae"], source_row["rmse"]
                    for days in ADAPTATION_DAYS:
                        if days == 0:
                            pred = source_pred
                            sample_count, final_loss, used_batch = 0, math.nan, 0
                            adaptation_oom = []
                            model_name = "c0_source_only_lstm"
                        else:
                            adapt_x, adapt_y, adapt_meta = adaptation_windows(
                                raw, building.building_id, days
                            )
                            sample_count = len(adapt_meta)
                            expected = days * 24 - 23
                            if sample_count != expected:
                                raise AssertionError(
                                    f"{days}d samples {sample_count}, expected {expected}"
                                )
                            adapted_model, final_loss, used_batch, adaptation_oom = adapt_head(
                                config,
                                state,
                                adapt_x,
                                adapt_y,
                                adapt_meta,
                                mean,
                                std,
                                device,
                                seed + (dev_index + 1) * 10000 + days * 100,
                            )
                            pred = predict(adapted_model, test_x, test_meta, mean, std, device)
                            model_name = "c0_head_only_adaptation_lstm"
                        row = {
                            "building_id": building.building_id,
                            "site_id": building.site_id,
                            "primaryspaceusage": building.primaryspaceusage,
                            "seed": seed,
                            "model": model_name,
                            "input_condition": "C0",
                            "adaptation_days": days,
                            "adaptation_samples": sample_count,
                            "adaptation_epochs": 0 if days == 0 else config.adaptation_epochs,
                            "adaptation_batch_size": used_batch,
                            "adaptation_final_train_mse_normalized": final_loss,
                            "test_windows": len(test_meta),
                            "oom_fallback_count": len(adaptation_oom),
                        }
                        row.update(metric_row(test_y, pred))
                        row["mae_delta_vs_same_seed_c0_source_only"] = row["mae"] - base_mae
                        row["rmse_delta_vs_same_seed_c0_source_only"] = row["rmse"] - base_rmse
                        adapt_rows.append(row)
                        adapt_predictions.append(
                            prediction_rows(test_meta, test_y, pred, model_name, seed, days)
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "seed": seed,
                            "building_id": building.building_id,
                            "adaptation_days": "",
                            "stage": "c0_development_evaluation_or_adaptation",
                            "error": repr(exc),
                        }
                    )
                    save_progress(
                        source_rows,
                        source_predictions,
                        adapt_rows,
                        adapt_predictions,
                        histories,
                        failures,
                    )
                    raise
            run_config["completed_seeds"] += 1
            run_config["elapsed_seconds"] = time.time() - started
            save_progress(
                source_rows, source_predictions, adapt_rows, adapt_predictions, histories, failures
            )
            save_json(CONFIG_PATH, run_config)
            print(f"completed C0 seed {seed}; elapsed={time.time() - started:.1f}s", flush=True)

        source_frame = pd.DataFrame(source_rows)
        adapt_frame = pd.DataFrame(adapt_rows)
        source_summary = source_frame.groupby(["building_id", "site_id"], as_index=False).agg(
            completed_seeds=("seed", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        adapt_summary = adapt_frame.groupby(
            ["building_id", "site_id", "adaptation_days"], as_index=False
        ).agg(
            completed_seeds=("seed", "nunique"),
            adaptation_samples=("adaptation_samples", "first"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_delta_mean=("mae_delta_vs_same_seed_c0_source_only", "mean"),
            rmse_delta_mean=("rmse_delta_vs_same_seed_c0_source_only", "mean"),
        )
        report = f"""# BDG2 Stage 6 C0 LSTM 개발 결과 v3

## 완료 상태

- Source 20개 건물 C0 학습·검증: 완료.
- 개발 12개 건물 Source-only 및 0·1·3·7·14·28일 출력층 적응: 완료.
- 사전 고정 시드: {len(SEEDS)}/{len(SEEDS)} 완료.
- 실패: {len(failures)}개. final-test 전력 사용: 아니오.
- 실행 장치: `{device_name}`, 실행 시간 {time.time() - started:.1f}초.

## 입력과 데이터 분리

- C0 feature: {', '.join(C0_FEATURE_NAMES)}.
- 목표 달력 feature는 예측 생성 시 확정되며 미래 관측값이 아니다.
- 정규화 mean={mean:.12g}, std={std:.12g}; Source 학습 X와 y만 사용했다.
- Source 검증만 checkpoint 선택에 사용했다.
- v2 power-only LSTM과 비교해 input size만 1→6으로 바뀌었다.

## Source-only 결과

```
{source_summary.to_string(index=False)}
```

## 출력층 적응 결과

양의 delta는 같은 시드 C0 Source-only보다 악화된 결과이다.

```
{adapt_summary.to_string(index=False)}
```

이 결과는 개발용 C0 예비 실험이며 C1 또는 최종 시험 결과가 아니다.
"""
        REVIEW.write_text(report, encoding="utf-8")
        run_config.update(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "failures": failures,
                "outputs": {
                    str(path): sha256(path)
                    for path in (
                        SOURCE_METRICS,
                        SOURCE_PREDICTIONS,
                        ADAPT_METRICS,
                        ADAPT_PREDICTIONS,
                        HISTORY,
                        FAILURES,
                        REVIEW,
                    )
                },
            }
        )
        save_json(CONFIG_PATH, run_config)
        print(source_summary.to_string(index=False))
        print(adapt_summary.to_string(index=False))
    except Exception as exc:
        run_config["status"] = "failed"
        run_config["errors"].append(repr(exc))
        run_config["elapsed_seconds"] = time.time() - started
        run_config["failures"] = failures
        save_json(CONFIG_PATH, run_config)
        save_progress(
            source_rows, source_predictions, adapt_rows, adapt_predictions, histories, failures
        )
        raise


if __name__ == "__main__":
    main()
