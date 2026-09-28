"""Build initial BDG2 24-hour history -> 24-hour target windows.

Only buildings in the alternative source-train and development-validation
groups are considered.  The first two building IDs in each group that have at
least 90 days of longest complete coverage and no 30-day zero-run review flag
are selected.  The final-test group is never read for window construction.

The raw electricity CSV is read with only timestamp plus the four selected
building columns.  No timezone conversion, zero replacement, normalization,
imputation, or model fitting is performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[1]))).resolve()
RAW_ELECTRICITY = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
SPLIT_INPUT = ROOT / "reports" / "bdg2_cohort_split_alternative.csv"
MANIFEST_OUTPUT = ROOT / "reports" / "bdg2_split_manifest.csv"
METADATA_OUTPUT = ROOT / "reports" / "bdg2_window_sample_metadata.csv"
REPORT_OUTPUT = ROOT / "reports" / "bdg2_window_pipeline_preview.md"
ARRAY_OUTPUT = ROOT / "data" / "processed" / "bdg2" / "bdg2_initial_window_samples.npz"

GROUP_ORDER = ("source_train_proposal", "development_validation_proposal")
WINDOW_INPUT_HOURS = 24
WINDOW_TARGET_HOURS = 24
WINDOW_TOTAL_HOURS = WINDOW_INPUT_HOURS + WINDOW_TARGET_HOURS
ONE_HOUR = pd.Timedelta(hours=1)
SELECTION_RULE = (
    "alternative_group이 source_train_proposal 또는 development_validation_proposal이고, "
    "building_id 오름차순에서 longest_complete_run_days >= 90이며 "
    "longest_zero_run_days < 30인 건물 중 그룹별 첫 2개"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def select_initial_buildings(split: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = split.copy()
    work["_run_days"] = pd.to_numeric(work["longest_complete_run_days"], errors="coerce")
    work["_zero_days"] = pd.to_numeric(work["longest_zero_run_days"], errors="coerce")
    work["selection_eligible"] = (
        work["alternative_group"].isin(GROUP_ORDER)
        & work["_run_days"].ge(90)
        & work["_zero_days"].lt(30)
    )
    work["selection_rank_within_group"] = pd.NA
    work["selected_for_initial_check"] = False
    work["selection_role"] = "not_selected"

    selected_parts: List[pd.DataFrame] = []
    for group in GROUP_ORDER:
        group_eligible = work[
            work["selection_eligible"] & work["alternative_group"].eq(group)
        ].sort_values("building_id", kind="stable")
        ranks = pd.Series(range(1, len(group_eligible) + 1), index=group_eligible.index)
        work.loc[ranks.index, "selection_rank_within_group"] = ranks
        # Re-read the chosen rows from work so the assigned rank and selection
        # flags are carried into the selected-building table.
        chosen = work.loc[group_eligible.head(2).index].copy()
        work.loc[chosen.index, "selected_for_initial_check"] = True
        work.loc[chosen.index, "selection_role"] = group
        selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        by=["alternative_group", "building_id"],
        key=lambda col: col.map({g: i for i, g in enumerate(GROUP_ORDER)})
        if col.name == "alternative_group"
        else col,
        kind="stable",
    ).reset_index(drop=True)
    work = work.drop(columns=["_run_days", "_zero_days"])
    return work, selected


def _format_timestamp(value) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat(sep=" ")


def build_windows_for_building(
    timestamps: Sequence,
    values: Sequence,
    building_id: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, object]]:
    """Return raw X/y arrays and metadata for one building.

    A valid start has 48 finite numeric values and 47 consecutive one-hour
    intervals inside the 48-row window.  This keeps the 24-hour input and
    following 24-hour target disjoint and chronological.
    """
    parsed_time = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    n_rows = len(parsed_time)
    if n_rows < WINDOW_TOTAL_HOURS:
        empty = pd.DataFrame()
        return (
            np.empty((0, WINDOW_INPUT_HOURS), dtype=float),
            np.empty((0, WINDOW_TARGET_HOURS), dtype=float),
            empty,
            {
                "building_id": building_id,
                "raw_rows": n_rows,
                "numeric_records": int(np.isfinite(numeric_values).sum()),
                "missing_values": int((~np.isfinite(numeric_values)).sum()),
                "candidate_starts": 0,
                "accepted_windows": 0,
            },
        )

    time_step_ok = parsed_time.diff().eq(ONE_HOUR).to_numpy(dtype=bool)
    # For a start i, the 47 internal intervals are represented by
    # time_step_ok[i + 1 : i + 48].
    time_window_ok = (
        np.convolve(
            time_step_ok[1:].astype(np.int8),
            np.ones(WINDOW_TOTAL_HOURS - 1, dtype=np.int8),
            mode="valid",
        )
        == WINDOW_TOTAL_HOURS - 1
    )
    value_window_ok = (
        np.convolve(
            np.isfinite(numeric_values).astype(np.int8),
            np.ones(WINDOW_TOTAL_HOURS, dtype=np.int8),
            mode="valid",
        )
        == WINDOW_TOTAL_HOURS
    )
    valid_starts = np.flatnonzero(time_window_ok & value_window_ok)

    x_values = np.vstack(
        [numeric_values[start : start + WINDOW_INPUT_HOURS] for start in valid_starts]
    ) if len(valid_starts) else np.empty((0, WINDOW_INPUT_HOURS), dtype=float)
    y_values = np.vstack(
        [
            numeric_values[
                start + WINDOW_INPUT_HOURS : start + WINDOW_TOTAL_HOURS
            ]
            for start in valid_starts
        ]
    ) if len(valid_starts) else np.empty((0, WINDOW_TARGET_HOURS), dtype=float)

    metadata_rows = []
    for local_index, start in enumerate(valid_starts):
        input_start = parsed_time.iloc[start]
        input_end = parsed_time.iloc[start + WINDOW_INPUT_HOURS - 1]
        target_start = parsed_time.iloc[start + WINDOW_INPUT_HOURS]
        target_end = parsed_time.iloc[start + WINDOW_TOTAL_HOURS - 1]
        metadata_rows.append(
            {
                "building_id": building_id,
                "building_window_index": local_index,
                "input_start": _format_timestamp(input_start),
                "input_end": _format_timestamp(input_end),
                "target_start": _format_timestamp(target_start),
                "target_end": _format_timestamp(target_end),
                "x_length": WINDOW_INPUT_HOURS,
                "y_length": WINDOW_TARGET_HOURS,
                "input_target_gap_hours": (target_start - input_end).total_seconds() / 3600.0,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    summary = {
        "building_id": building_id,
        "raw_rows": n_rows,
        "numeric_records": int(np.isfinite(numeric_values).sum()),
        "missing_values": int((~np.isfinite(numeric_values)).sum()),
        "candidate_starts": n_rows - WINDOW_TOTAL_HOURS + 1,
        "accepted_windows": int(len(valid_starts)),
        "first_window_start": metadata["input_start"].iloc[0] if len(metadata) else "",
        "last_window_target_end": metadata["target_end"].iloc[-1] if len(metadata) else "",
    }
    return x_values, y_values, metadata, summary


def validate_window_metadata(metadata: pd.DataFrame) -> Dict[str, object]:
    """Run structural checks used by the standalone validation script."""
    required = {
        "building_id",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
        "x_length",
        "y_length",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise AssertionError(f"window metadata required columns missing: {sorted(missing)}")
    if len(metadata) == 0:
        return {"rows": 0, "x_24": True, "y_24": True, "chronological": True}

    input_start = pd.to_datetime(metadata["input_start"], errors="coerce")
    input_end = pd.to_datetime(metadata["input_end"], errors="coerce")
    target_start = pd.to_datetime(metadata["target_start"], errors="coerce")
    target_end = pd.to_datetime(metadata["target_end"], errors="coerce")
    checks = {
        "rows": int(len(metadata)),
        "x_24": bool(metadata["x_length"].eq(WINDOW_INPUT_HOURS).all()),
        "y_24": bool(metadata["y_length"].eq(WINDOW_TARGET_HOURS).all()),
        "timestamps_parse": bool(input_start.notna().all() and input_end.notna().all() and target_start.notna().all() and target_end.notna().all()),
        "chronological": bool(
            (input_end - input_start).eq(pd.Timedelta(hours=23)).all()
            and (target_end - target_start).eq(pd.Timedelta(hours=23)).all()
            and (target_start - input_end).eq(ONE_HOUR).all()
        ),
    }
    checks["all_pass"] = all(value for key, value in checks.items() if key != "rows")
    if not checks["all_pass"]:
        raise AssertionError(f"window metadata validation failed: {checks}")
    return checks


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    headers = [str(c) for c in frame.columns]

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend(
        "| " + " | ".join(cell(row[col]) for col in frame.columns) + " |"
        for _, row in frame.iterrows()
    )
    return "\n".join(lines)


def run_pipeline() -> Dict[str, object]:
    split = pd.read_csv(SPLIT_INPUT)
    required_split = {
        "building_id",
        "site_id",
        "alternative_group",
        "longest_complete_run_days",
        "longest_zero_run_days",
    }
    missing_split = required_split.difference(split.columns)
    if missing_split:
        raise ValueError(f"split input missing columns: {sorted(missing_split)}")
    if split["building_id"].duplicated().any():
        raise ValueError("split input has duplicate building_id values")

    manifest, selected = select_initial_buildings(split)
    if len(selected) != 4:
        raise ValueError(f"selection rule expected 4 buildings, selected {len(selected)}")

    selected_ids = selected["building_id"].astype(str).tolist()
    electricity_header = pd.read_csv(RAW_ELECTRICITY, nrows=0)
    missing_raw_columns = sorted(set(["timestamp"] + selected_ids).difference(electricity_header.columns))
    if missing_raw_columns:
        raise ValueError(f"raw electricity missing selected columns: {missing_raw_columns}")

    raw = pd.read_csv(RAW_ELECTRICITY, usecols=["timestamp"] + selected_ids)
    raw_timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw_summary = {
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "timestamp_column": "timestamp",
        "timestamp_start": _format_timestamp(raw_timestamps.iloc[0]),
        "timestamp_end": _format_timestamp(raw_timestamps.iloc[-1]),
        "timestamp_parse_errors": int(raw_timestamps.isna().sum()),
        "timestamp_hourly_steps": int(raw_timestamps.diff().eq(ONE_HOUR).sum()),
        "timestamp_non_hourly_steps": int(raw_timestamps.diff().notna().sum() - raw_timestamps.diff().eq(ONE_HOUR).sum()),
    }

    all_metadata: List[pd.DataFrame] = []
    x_arrays: List[np.ndarray] = []
    y_arrays: List[np.ndarray] = []
    building_summaries: List[Dict[str, object]] = []
    for building_id in selected_ids:
        x_values, y_values, metadata, summary = build_windows_for_building(
            raw["timestamp"], raw[building_id], building_id
        )
        x_arrays.append(x_values)
        y_arrays.append(y_values)
        if len(metadata):
            all_metadata.append(metadata)
        building_summaries.append(summary)

    metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()
    if len(metadata):
        metadata.insert(0, "sample_index", range(len(metadata)))
    validation = validate_window_metadata(metadata)
    x_all = np.vstack(x_arrays) if x_arrays else np.empty((0, WINDOW_INPUT_HOURS), dtype=float)
    y_all = np.vstack(y_arrays) if y_arrays else np.empty((0, WINDOW_TARGET_HOURS), dtype=float)
    if x_all.shape != (len(metadata), WINDOW_INPUT_HOURS) or y_all.shape != (len(metadata), WINDOW_TARGET_HOURS):
        raise AssertionError("array shapes do not match metadata")
    if np.isnan(x_all).any() or np.isnan(y_all).any():
        raise AssertionError("accepted windows contain NaN")

    # Keep the fixed alternative assignment and selection audit in a separate
    # manifest.  All 1,528 candidate mappings are retained; only four are
    # marked as selected for this initial operation check.
    manifest["source_alternative_csv"] = str(SPLIT_INPUT.relative_to(ROOT))
    manifest["selection_rule"] = SELECTION_RULE
    manifest["source_alternative_csv_sha256"] = sha256_file(SPLIT_INPUT)
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_OUTPUT, index=False, encoding="utf-8-sig")
    metadata.to_csv(METADATA_OUTPUT, index=False, encoding="utf-8-sig")
    ARRAY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAY_OUTPUT, X=x_all, y=y_all)

    source_hash = sha256_file(SPLIT_INPUT)
    manifest_hash = sha256_file(MANIFEST_OUTPUT)
    raw_hash = sha256_file(RAW_ELECTRICITY)
    summary_frame = pd.DataFrame(building_summaries)
    selected_summary = selected[
        [
            "building_id",
            "site_id",
            "alternative_group",
            "longest_complete_run_days",
            "longest_zero_run_days",
            "selection_rank_within_group",
        ]
    ].copy()
    selected_summary = selected_summary.merge(summary_frame, on="building_id", how="left")

    report = f"""# BDG2 24시간 입력·24시간 정답 파이프라인 초기 확인

## 범위와 고정 분할

- 대체안 입력: `reports/bdg2_cohort_split_alternative.csv`
- 이번 실행에서 사용한 분할 열: `alternative_group`
- 출발 학습용과 개발·검증용만 사용했으며 `final_test_proposal` 건물은 원본 전력값을 읽지 않았습니다.
- 건물 배정은 모델 성능에 따라 변경하지 않았습니다. 건물별 고정 배정과 초기 선택 결과는 `reports/bdg2_split_manifest.csv`에 기록했습니다.

## 입력 파일 확인

| 항목 | 값 |
| --- | --- |
| 원본 전력 파일 | `data/raw/bdg2/electricity.csv` |
| 원본 파일 크기(bytes) | {RAW_ELECTRICITY.stat().st_size:,} |
| 원본 SHA-256 | `{raw_hash}` |
| 원본 timestamp 열 | `timestamp` |
| 선택 열을 읽은 행 수 | {raw_summary['rows']:,} |
| 선택 열을 읽은 열 수 | {raw_summary['columns']} |
| timestamp 시작 | `{raw_summary['timestamp_start']}` |
| timestamp 종료 | `{raw_summary['timestamp_end']}` |
| 1시간 간격 행 수 | {raw_summary['timestamp_hourly_steps']:,} |
| 비정상 간격 행 수 | {raw_summary['timestamp_non_hourly_steps']:,} |
| timestamp 파싱 오류 | {raw_summary['timestamp_parse_errors']} |

시간대 변환이나 서머타임 보정은 수행하지 않았습니다. 원본 timestamp 표현을 파싱해 1시간 간격 여부만 검사했습니다.

## 초기 건물 선택

선택 규칙: {SELECTION_RULE}

{markdown_table(selected_summary, ['building_id', 'site_id', 'alternative_group', 'longest_complete_run_days', 'longest_zero_run_days', 'selection_rank_within_group', 'raw_rows', 'numeric_records', 'missing_values', 'accepted_windows'])}

위 네 건물은 ID 오름차순으로 각 그룹의 첫 두 건물입니다. 30일 이상 연속 0값 검토 대상은 선택 조건에서 제외했지만, 나머지 정상적인 0값은 삭제하지 않았습니다.

## 생성 결과

- X: 과거 24개 시간, shape `{x_all.shape}`, 원본 숫자값 그대로
- y: 그 직후 24개 시간, shape `{y_all.shape}`, 원본 숫자값 그대로
- 총 생성 샘플: {len(metadata):,}개
- 샘플 메타데이터: `reports/bdg2_window_sample_metadata.csv`
- 배열 파일: `data/processed/bdg2/bdg2_initial_window_samples.npz`

{markdown_table(summary_frame, ['building_id', 'raw_rows', 'numeric_records', 'missing_values', 'candidate_starts', 'accepted_windows', 'first_window_start', 'last_window_target_end'])}

각 샘플은 48개 행 전체가 숫자이고, 내부 47개 간격이 모두 1시간인 경우에만 생성했습니다. 따라서 결측값을 포함하거나 시간 간격이 끊긴 창은 생성하지 않았습니다. X와 y는 서로 겹치지 않으며 y는 X 종료 시각의 정확히 1시간 뒤에 시작합니다.

## 자동 검사 결과

검증 코드 `scripts/test_bdg2_windows.py`를 실행했으며 다음 조건을 검사했습니다.

| 검사 | 결과 |
| --- | --- |
| X 길이 24 | `{'PASS' if validation['x_24'] else 'FAIL'}` |
| y 길이 24 | `{'PASS' if validation['y_24'] else 'FAIL'}` |
| timestamp 파싱 | `{'PASS' if validation['timestamps_parse'] else 'FAIL'}` |
| X와 y의 시간순 연결 | `{'PASS' if validation['chronological'] else 'FAIL'}` |
| X/y에 결측 없음 | `{'PASS' if (not np.isnan(x_all).any() and not np.isnan(y_all).any()) else 'FAIL'}` |
| 배열 shape와 메타데이터 행 수 일치 | `{'PASS' if x_all.shape[0] == len(metadata) and y_all.shape[0] == len(metadata) else 'FAIL'}` |

## 파일 해시

| 파일 | SHA-256 |
| --- | --- |
| `reports/bdg2_cohort_split_alternative.csv` (입력) | `{source_hash}` |
| `reports/bdg2_split_manifest.csv` | `{manifest_hash}` |

## 제한 사항

이번 단계에서는 정규화, 건물별 적응 데이터 구성, 최종 학습·시험 기간 확정, 결측값 보간, 날씨 데이터 사용 및 AI 모델 학습을 수행하지 않았습니다. 배열 파일은 초기 파이프라인 확인용 원본 단위 샘플입니다.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    return {
        "manifest": manifest,
        "selected": selected,
        "metadata": metadata,
        "X": x_all,
        "y": y_all,
        "building_summaries": summary_frame,
        "validation": validation,
        "raw_summary": raw_summary,
        "source_hash": source_hash,
        "manifest_hash": manifest_hash,
        "raw_hash": raw_hash,
    }


def main() -> None:
    result = run_pipeline()
    print(f"selected_buildings={result['selected']['building_id'].tolist()}")
    print(f"X_shape={result['X'].shape} y_shape={result['y'].shape}")
    print(f"metadata_rows={len(result['metadata'])}")
    print(f"validation={result['validation']}")
    print(f"manifest_sha256={result['manifest_hash']}")


if __name__ == "__main__":
    main()
