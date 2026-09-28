from __future__ import annotations

import csv
import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[1]))).resolve()
OUT_DIR = ROOT / "figure8_extraction"
SOURCE = ROOT / "reports" / "final_v1_raw_predictions.csv"
FINAL_REPORT = ROOT / "reports" / "final_v1_final_test_report.md"
ARTIFACT_MANIFEST = ROOT / "reports" / "final_v1_artifact_manifest.csv"
GATE2_COMPLETE = ROOT / "reports" / "final_v1" / "GATE2_COMPLETE.json"
GATE2_CODE = ROOT / "scripts" / "final_v1" / "run_final_v1_gate2.py"
GATE3_CODE = ROOT / "scripts" / "final_v1" / "analyze_final_v1_gate3.py"

OUTPUT_PREDICTIONS = OUT_DIR / "case_predictions.csv"
OUTPUT_METRICS = OUT_DIR / "case_metrics.csv"
SEARCH_REPORT = OUT_DIR / "SEARCH_REPORT.md"
VALIDATION_REPORT = OUT_DIR / "VALIDATION_REPORT.md"

EXPECTED_TOTAL_ROWS = 6_088_320
EXPECTED_EXTRACTED_ROWS = 6_720
EXPECTED_SOURCE_SHA256 = "72732c5dc0b0ad9cb7440f326476cd8c2ae57dd9267f7e07398452ae5b636e43"

CASES = [
    ("Hog_office_Roger", 20260926, 0),
    ("Hog_office_Roger", 20260926, 1),
    ("Eagle_education_Brooke", 20260924, 0),
    ("Eagle_education_Brooke", 20260924, 14),
    ("Eagle_education_Brooke", 20260925, 0),
    ("Eagle_education_Brooke", 20260925, 14),
    ("Eagle_education_Brooke", 20260926, 0),
    ("Eagle_education_Brooke", 20260926, 14),
    ("Hog_office_Gustavo", 20260926, 0),
    ("Hog_office_Gustavo", 20260926, 1),
]

SOURCE_COLUMNS = [
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

READ_COLUMNS = [
    "building_id",
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

OUTPUT_COLUMNS = [
    "building_id",
    "seed",
    "adaptation_days",
    "forecast_origin",
    "target_timestamp",
    "horizon",
    "y_true",
    "y_pred",
]

METRIC_COLUMNS = [
    "building_id",
    "seed",
    "adaptation_days",
    "mae",
    "rmse",
    "source_mae",
    "source_rmse",
    "delta_mae",
    "delta_rmse",
    "row_count",
    "validation_status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_preserving_offset(value: str) -> str:
    """Convert the separator to ISO 8601 without inventing timezone information."""
    parsed = datetime.fromisoformat(value.strip())
    return parsed.isoformat(timespec="seconds")


def file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def manifest_record() -> dict[str, str]:
    with ARTIFACT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["relative_path"] == "reports/final_v1_raw_predictions.csv":
                return row
    raise AssertionError("final_v1 raw prediction entry missing from artifact manifest")


def inspect_header() -> list[str]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if header != SOURCE_COLUMNS:
        raise AssertionError(f"Unexpected source header: {header}")
    return header


def scan_and_extract() -> tuple[int, pd.DataFrame]:
    collected: list[pd.DataFrame] = []
    total_rows = 0
    target_buildings = {building for building, _, _ in CASES}
    case_keys = {(building, str(seed), str(days)) for building, seed, days in CASES}

    for chunk in pd.read_csv(
        SOURCE,
        usecols=READ_COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=250_000,
    ):
        total_rows += len(chunk)
        subset = chunk.loc[chunk["building_id"].isin(target_buildings)].copy()
        if subset.empty:
            continue
        tuples = list(zip(subset["building_id"], subset["seed"], subset["adaptation_days"]))
        keep = np.fromiter((key in case_keys for key in tuples), dtype=bool, count=len(tuples))
        subset = subset.loc[keep].copy()
        if subset.empty:
            continue
        expected_model = np.where(
            subset["adaptation_days"].eq("0"),
            "c0_source_only_lstm",
            "c0_head_only_adaptation_lstm",
        )
        subset = subset.loc[subset["model"].to_numpy() == expected_model].copy()
        collected.append(subset)

    if not collected:
        raise AssertionError("No requested cases found")
    extracted = pd.concat(collected, ignore_index=True)
    return total_rows, extracted


def validate_and_standardize(extracted: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    work = extracted.copy()
    work["seed"] = work["seed"].astype(int)
    work["adaptation_days"] = work["adaptation_days"].astype(int)
    work["horizon"] = work["horizon_hour"].astype(int)
    work["forecast_origin"] = work["forecast_origin"].map(iso_preserving_offset)
    work["target_timestamp"] = work["target_timestamp"].map(iso_preserving_offset)
    work["_case_order"] = [CASES.index((b, s, d)) for b, s, d in zip(
        work["building_id"], work["seed"], work["adaptation_days"]
    )]
    work = work.sort_values(
        ["_case_order", "forecast_origin", "target_timestamp", "horizon"],
        kind="stable",
    ).reset_index(drop=True)

    standardized = work[OUTPUT_COLUMNS].copy()
    checks: dict[str, object] = {}
    checks["total_rows_6720"] = len(standardized) == EXPECTED_EXTRACTED_ROWS
    checks["ten_cases_present"] = standardized.groupby(
        ["building_id", "seed", "adaptation_days"], sort=False
    ).ngroups == len(CASES)

    per_case: dict[tuple[str, int, int], dict[str, bool]] = {}
    for case in CASES:
        building, seed, days = case
        group = standardized.loc[
            standardized["building_id"].eq(building)
            & standardized["seed"].eq(seed)
            & standardized["adaptation_days"].eq(days)
        ].copy()
        origins = group["forecast_origin"].nunique()
        horizon_ok = True
        for _, origin_group in group.groupby("forecast_origin", sort=False):
            horizon_ok &= sorted(origin_group["horizon"].tolist()) == list(range(1, 25))
        per_case[case] = {
            "row_count": len(group) == 672,
            "origin_count": origins == 28,
            "horizons": bool(horizon_ok),
        }
    checks["each_case_672_rows"] = all(v["row_count"] for v in per_case.values())
    checks["each_case_28_origins"] = all(v["origin_count"] for v in per_case.values())
    checks["each_origin_horizons_1_24"] = all(v["horizons"] for v in per_case.values())

    pair_results: dict[tuple[str, int, int], bool] = {}
    axes = ["forecast_origin", "target_timestamp", "horizon"]
    for building, seed, days in CASES:
        if days == 0:
            continue
        source = standardized.loc[
            standardized["building_id"].eq(building)
            & standardized["seed"].eq(seed)
            & standardized["adaptation_days"].eq(0)
        ].sort_values(axes, kind="stable")
        adapted = standardized.loc[
            standardized["building_id"].eq(building)
            & standardized["seed"].eq(seed)
            & standardized["adaptation_days"].eq(days)
        ].sort_values(axes, kind="stable")
        same = (
            len(source) == len(adapted) == 672
            and source[axes].reset_index(drop=True).equals(adapted[axes].reset_index(drop=True))
            and source["y_true"].reset_index(drop=True).equals(adapted["y_true"].reset_index(drop=True))
        )
        pair_results[(building, seed, days)] = bool(same)
    checks["source_adapted_axes_and_y_true_exact"] = all(pair_results.values())

    true_values = pd.to_numeric(standardized["y_true"], errors="coerce").to_numpy(float)
    pred_values = pd.to_numeric(standardized["y_pred"], errors="coerce").to_numpy(float)
    checks["finite_y_true_y_pred"] = bool(np.isfinite(true_values).all() and np.isfinite(pred_values).all())
    duplicate_key = ["building_id", "seed", "adaptation_days", "forecast_origin", "target_timestamp", "horizon"]
    checks["no_duplicate_key"] = not standardized.duplicated(duplicate_key).any()
    checks["raw_status_ok"] = bool(extracted["status"].eq("ok").all())
    checks["prediction_available_true"] = bool(extracted["prediction_available"].str.lower().eq("true").all())

    base_metrics: dict[tuple[str, int], tuple[float, float]] = {}
    calculated: dict[tuple[str, int, int], tuple[float, float]] = {}
    for case in CASES:
        building, seed, days = case
        group = standardized.loc[
            standardized["building_id"].eq(building)
            & standardized["seed"].eq(seed)
            & standardized["adaptation_days"].eq(days)
        ]
        actual = pd.to_numeric(group["y_true"], errors="raise").to_numpy(float)
        predicted = pd.to_numeric(group["y_pred"], errors="raise").to_numpy(float)
        mae = float(np.mean(np.abs(predicted - actual)))
        rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
        calculated[case] = (mae, rmse)
        if days == 0:
            base_metrics[(building, seed)] = (mae, rmse)

    metric_rows: list[dict] = []
    for case in CASES:
        building, seed, days = case
        mae, rmse = calculated[case]
        source_mae, source_rmse = base_metrics[(building, seed)]
        status_parts = per_case[case].values()
        pair_ok = True if days == 0 else pair_results[case]
        status = "PASS" if all(status_parts) and pair_ok and checks["finite_y_true_y_pred"] else "FAIL"
        metric_rows.append(
            {
                "building_id": building,
                "seed": seed,
                "adaptation_days": days,
                "mae": mae,
                "rmse": rmse,
                "source_mae": source_mae,
                "source_rmse": source_rmse,
                "delta_mae": mae - source_mae,
                "delta_rmse": rmse - source_rmse,
                "row_count": 672,
                "validation_status": status,
            }
        )

    metrics_by_case = {
        (r["building_id"], int(r["seed"]), int(r["adaptation_days"])): r for r in metric_rows
    }
    roger = metrics_by_case[("Hog_office_Roger", 20260926, 1)]
    brooke_deltas = [
        metrics_by_case[("Eagle_education_Brooke", seed, 14)]["delta_mae"]
        for seed in (20260924, 20260925, 20260926)
    ]
    gustavo = metrics_by_case[("Hog_office_Gustavo", 20260926, 1)]
    checks["roger_delta_mae_positive"] = roger["delta_mae"] > 0
    checks["brooke_mixed_delta_mae_directions"] = min(brooke_deltas) < 0 < max(brooke_deltas)
    checks["brooke_delta_mae_range_matches"] = math.isclose(
        max(brooke_deltas) - min(brooke_deltas),
        226.53299509003176,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    checks["gustavo_delta_mae_negative"] = gustavo["delta_mae"] < 0
    checks["gustavo_delta_rmse_matches"] = math.isclose(
        gustavo["delta_rmse"], 6.484847373417267, rel_tol=0.0, abs_tol=1e-9
    )
    checks["all_case_metric_status_pass"] = all(r["validation_status"] == "PASS" for r in metric_rows)
    checks["all_validation_checks_pass"] = all(bool(value) for value in checks.values())
    return standardized, metric_rows, checks


def write_csv_preserving_numeric_text(frame: pd.DataFrame) -> None:
    with OUTPUT_PREDICTIONS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in frame.to_dict(orient="records"):
            writer.writerow(row)


def write_metrics(rows: list[dict]) -> None:
    with OUTPUT_METRICS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows() -> list[tuple[str, int, str]]:
    candidates = [
        (ROOT / "reports" / "bdg2_stage6_c0_adaptation_predictions_v3.csv", "Stage-6 development adaptation output; not final_v1"),
        (ROOT / "reports" / "bdg2_stage6_c0_source_only_predictions_v3.csv", "Stage-6 development source-only output; not final_v1"),
        (ROOT / "reports" / "bdg2_multibuilding_adaptation_predictions_v2.csv", "v2 development output; not final_v1"),
        (ROOT / "reports" / "bdg2_multibuilding_source_only_predictions_v2.csv", "v2 development output; not final_v1"),
        (ROOT / "reports" / "bdg2_adaptation_predictions_20260924.csv", "four-building preliminary adaptation output"),
        (ROOT / "reports" / "bdg2_source_only_predictions_20260924.csv", "four-building preliminary source-only output"),
        (ROOT / "reports" / "final_v1" / "final_v1_representative_failure_predictions.csv", "derived representative-case subset; not the full raw table"),
    ]
    result = []
    for path, reason in candidates:
        if path.exists():
            result.append((str(path.resolve()), path.stat().st_size, reason))
    return result


def write_search_report(total_rows: int, source_hash: str, header: list[str], manifest: dict[str, str]) -> None:
    meta = file_metadata(SOURCE)
    candidate_lines = "\n".join(
        f"| `{path}` | {size:,} | {reason} |" for path, size, reason in candidate_rows()
    )
    dtype_rows = [
        ("building_id", "string", "producer schema and stored text"),
        ("site_id", "string", "producer schema and stored text"),
        ("model", "string", "producer schema and stored text"),
        ("seed", "nullable integer", "neural rows contain integer seed; baseline rows are blank"),
        ("adaptation_days", "nullable integer", "neural rows contain integer budget; baseline rows are blank"),
        ("forecast_origin", "offset-free datetime text", "producer writes timestamp text; no UTC offset field is stored"),
        ("input_start", "offset-free datetime text", "producer writes timestamp text"),
        ("input_end", "offset-free datetime text", "producer writes timestamp text"),
        ("target_timestamp", "offset-free datetime text", "producer formats `%Y-%m-%d %H:%M:%S`"),
        ("horizon_hour", "integer", "producer tiles 1 through 24"),
        ("y_true", "float64 decimal text", "producer flattens NumPy float arrays"),
        ("y_pred", "float64 decimal text", "producer flattens NumPy float arrays"),
        ("prediction_available", "boolean", "producer uses `np.isfinite(y_pred)`"),
        ("evaluation_mode", "string", "`daily_nonoverlap`"),
        ("status", "string", "execution status"),
        ("error", "string", "blank on success"),
    ]
    dtype_table = "\n".join(f"| {name} | {dtype} | {basis} |" for name, dtype, basis in dtype_rows)
    mapping_table = "\n".join(
        [
            "| 건물 식별자 | `building_id` | `building_id` | producer schema |",
            "| random seed | `seed` | `seed` | final neural execution loops over fixed seeds |",
            "| adaptation budget | `adaptation_days` | `adaptation_days` | Gate 2 fixed budget field |",
            "| forecast origin | `forecast_origin` | `forecast_origin` | `prediction_frame` metadata |",
            "| target timestamp | `target_timestamp` | `target_timestamp` | `prediction_frame` hourly target timestamps |",
            "| horizon | `horizon_hour` | `horizon` | producer explicitly tiles 1–24 |",
            "| observed value | `y_true` | `y_true` | metric code parses this field as float |",
            "| prediction | `y_pred` | `y_pred` | metric code parses this field as float |",
        ]
    )
    report = f"""# Figure 8 raw-prediction search report

## 권위 있는 원시 예측 파일

- 절대경로: `{meta['absolute_path']}`
- 형식: CSV, UTF-8 text, one header row
- 파일 크기: {meta['size_bytes']:,} bytes
- 수정 시각: {meta['modified_local']} (local), {meta['modified_utc']} (UTC)
- 전체 데이터 행 수: {total_rows:,}
- 현재 파일 SHA-256: `{source_hash}`
- 기존 artifact manifest SHA-256: `{manifest['sha256']}`
- manifest 기록 크기: {int(manifest['size_bytes']):,} bytes
- manifest 기록 수정 시각: {manifest['modified_utc']}
- 전체 열: `{', '.join(header)}`

CSV 자체에는 물리적 dtype metadata가 없다. 아래 자료형은 최종 producer 코드와 저장 텍스트를 함께 확인한 논리 자료형이다.

| 원본 열 | 논리 자료형 | 근거 |
|---|---|---|
{dtype_table}

## 최종 실행 결과로 판정한 근거

1. `reports/final_v1_final_test_report.md`가 Gate 2 raw prediction을 6,088,320행으로 기록하며, 독립 metric 재계산 최대 차이는 `1.71e-13`이다.
2. `reports/final_v1/GATE2_COMPLETE.json`이 완료 상태, 453개 건물, 오류 0개, 6,088,320행과 이 파일의 SHA-256을 기록한다.
3. `scripts/final_v1/run_final_v1_gate2.py`의 `FINAL_PREDICTIONS`가 이 경로를 지정하고, `prediction_frame`이 위 schema를 생성한다.
4. `scripts/final_v1/analyze_final_v1_gate3.py`가 이 파일의 `y_true`와 `y_pred`를 직접 읽어 독립 metric을 재계산한다.
5. 실제 스트리밍 행 수, 파일 크기, 현재 SHA-256이 완료 기록과 artifact manifest에 모두 일치한다.

## 열 대응

| 의미 | 원본 열 | 표준 출력 열 | 근거 |
|---|---|---|---|
{mapping_table}

`y_true`와 `y_pred`는 `prediction_frame`에서 실제 평가 배열을 직접 펼친 값이고, Gate 3 metric 코드도 같은 두 열을 float64로 읽는다. 추출 CSV에는 원본 숫자 문자열을 반올림하거나 재서식하지 않고 그대로 기록했다.

원본 timestamp에는 UTC offset 또는 timezone 이름이 저장되어 있지 않다. 따라서 추출 파일은 offset을 발명하지 않고 구분자만 `T`로 바꾼 offset-free ISO 8601 형식으로 보존했다.

## 제외한 후보

| 절대경로 | 크기(bytes) | 제외 근거 |
|---|---:|---|
{candidate_lines}

프로젝트 재귀 검색에서 `Figure_Table_Package_v1.zip`과 `code/make_figure8_from_predictions.py`는 발견되지 않았다. 프로젝트의 ZIP 파일은 Edge Deployment 전송 번들이며 Figure 8 생성 코드가 아니다.
"""
    SEARCH_REPORT.write_text(report, encoding="utf-8")


def fmt(value: float) -> str:
    return format(value, ".15g")


def write_validation_report(metric_rows: list[dict], checks: dict, source_hash: str) -> None:
    check_lines = "\n".join(
        f"| {name} | {'PASS' if bool(value) else 'FAIL'} |" for name, value in checks.items()
    )
    metric_lines = "\n".join(
        "| {building_id} | {seed} | {adaptation_days} | {mae} | {rmse} | {delta_mae} | {delta_rmse} | {row_count} | {validation_status} |".format(
            **{**row, **{k: fmt(row[k]) for k in ("mae", "rmse", "delta_mae", "delta_rmse")}}
        )
        for row in metric_rows
    )
    by_key = {(r["building_id"], r["seed"], r["adaptation_days"]): r for r in metric_rows}
    roger = by_key[("Hog_office_Roger", 20260926, 1)]
    brooke = [by_key[("Eagle_education_Brooke", seed, 14)] for seed in (20260924, 20260925, 20260926)]
    gustavo = by_key[("Hog_office_Gustavo", 20260926, 1)]
    brooke_range = max(r["delta_mae"] for r in brooke) - min(r["delta_mae"] for r in brooke)
    report = f"""# Figure 8 extraction validation report

## 입력과 방법

- 원시 입력: `{SOURCE.resolve()}`
- 원시 입력 SHA-256: `{source_hash}`
- 추출 방식: pandas chunked CSV scan (`chunksize=250000`, selected columns only)
- 모델 학습·적응·추론·prediction 재생성: 수행하지 않음
- metric: 저장된 672개 `y_true`, `y_pred`에서 `MAE = mean(abs(y_pred-y_true))`, `RMSE = sqrt(mean((y_pred-y_true)^2))`로 독립 재계산

## 검증 결과

| 검사 | 결과 |
|---|---|
{check_lines}

## 사례별 metric

| building_id | seed | days | MAE | RMSE | ΔMAE | ΔRMSE | rows | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{metric_lines}

ΔMAE와 ΔRMSE는 `adapted - source-only`이다. 0일 행의 delta는 0이다.

## 고정 결과 재현

- `Hog_office_Roger`, seed 20260926, 1일: ΔMAE = `{roger['delta_mae']!r}` (> 0), ΔRMSE = `{roger['delta_rmse']!r}`.
- `Eagle_education_Brooke`, 14일 ΔMAE: seed 20260924 = `{brooke[0]['delta_mae']!r}`, 20260925 = `{brooke[1]['delta_mae']!r}`, 20260926 = `{brooke[2]['delta_mae']!r}`. 개선·악화 방향이 모두 존재한다.
- Brooke 최대–최소 ΔMAE = `{brooke_range!r}`; 기준값 `226.53299509003176`과 절대오차 1e-9 이내로 일치한다.
- `Hog_office_Gustavo`, seed 20260926, 1일: ΔMAE = `{gustavo['delta_mae']!r}` (< 0), ΔRMSE = `{gustavo['delta_rmse']!r}`; 기준값 `6.484847373417267`과 절대오차 1e-9 이내로 일치한다.

## Figure 생성 여부

프로젝트에서 지정된 `Figure_Table_Package_v1.zip` 또는 `code/make_figure8_from_predictions.py`를 찾지 못했다. 요청 조건에 따라 검증된 CSV와 metric 보고서까지만 생성했으며 Figure 시계열이나 seed 값을 새로 만들지 않았다.
"""
    VALIDATION_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    for output in (OUTPUT_PREDICTIONS, OUTPUT_METRICS, SEARCH_REPORT, VALIDATION_REPORT):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for required in (SOURCE, FINAL_REPORT, ARTIFACT_MANIFEST, GATE2_COMPLETE, GATE2_CODE, GATE3_CODE):
        if not required.exists():
            raise FileNotFoundError(required)

    header = inspect_header()
    manifest = manifest_record()
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256 or source_hash != manifest["sha256"]:
        raise AssertionError("Source SHA-256 does not match frozen final_v1 records")
    if SOURCE.stat().st_size != int(manifest["size_bytes"]):
        raise AssertionError("Source size does not match artifact manifest")

    total_rows, extracted = scan_and_extract()
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise AssertionError(f"Unexpected source row count: {total_rows}")

    standardized, metric_rows, checks = validate_and_standardize(extracted)
    write_csv_preserving_numeric_text(standardized)
    write_metrics(metric_rows)
    write_search_report(total_rows, source_hash, header, manifest)
    write_validation_report(metric_rows, checks, source_hash)

    if not checks["all_validation_checks_pass"]:
        raise AssertionError(f"Validation failed: {checks}")
    print(f"SOURCE_ROWS={total_rows}")
    print(f"EXTRACTED_ROWS={len(standardized)}")
    print("VALIDATION=PASS")


if __name__ == "__main__":
    main()
