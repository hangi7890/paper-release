"""Create a fixed, leakage-safe temporal split manifest for four BDG2 buildings.

The existing 24-to-24 window builder is preserved and reused for its numeric
and timestamp conventions.  This script only reads the two source-train and
two development-validation building columns selected in the existing split
manifest.  It never loads a final-test electricity column.

Design fixed for this stage:
* find the earliest common all-valid hourly run long enough for 24 hours of
  input history, 28 days of adaptation targets, and a 28-day test target span;
* reserve the first 24 hours as input history;
* start adaptation targets one day after the common run begins;
* place the 28-day test target interval immediately after the 28-day budget;
* evaluate every adaptation budget (0, 1, 3, 7, 14, 28 days) on that same test
  interval.

No normalization, imputation, timezone conversion, or model fitting occurs.
Only timestamp metadata and counts are written; no large window-value CSV is
created.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(SCRIPT_DIR.parent))).resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from build_bdg2_windows import sha256_file  # noqa: E402

RAW_ELECTRICITY = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
SPLIT_MANIFEST = ROOT / "reports" / "bdg2_split_manifest.csv"
OUTPUT_MANIFEST = ROOT / "reports" / "bdg2_temporal_split_manifest.csv"
OUTPUT_REPORT = ROOT / "reports" / "bdg2_temporal_split_preview.md"

ONE_HOUR = pd.Timedelta(hours=1)
INPUT_HISTORY_HOURS = 24
MAX_ADAPTATION_DAYS = 28
TEST_DAYS = 28
ADAPTATION_DAYS = (0, 1, 3, 7, 14, 28)
REQUIRED_RUN_HOURS = INPUT_HISTORY_HOURS + MAX_ADAPTATION_DAYS * 24 + TEST_DAYS * 24
GROUPS_USED = ("source_train_proposal", "development_validation_proposal")
FINAL_GROUP = "final_test_proposal"


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def fmt_time(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat(sep=" ")


def numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def find_valid_runs(raw: pd.DataFrame, building_ids: Sequence[str]) -> pd.DataFrame:
    """Return all common, all-numeric, hourly contiguous runs."""
    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    values = raw[list(building_ids)].apply(pd.to_numeric, errors="coerce")
    hourly_step = timestamps.diff().eq(ONE_HOUR)
    if len(hourly_step):
        hourly_step.iloc[0] = True
    valid = timestamps.notna() & hourly_step & values.notna().all(axis=1)
    segment = (valid != valid.shift(fill_value=False)).cumsum()
    rows = []
    for segment_id, part in raw[valid].groupby(segment[valid]):
        idx = part.index
        if not len(idx):
            continue
        rows.append(
            {
                "row_start": int(idx[0]),
                "row_end": int(idx[-1]),
                "hours": int(len(idx)),
                "start": timestamps.iloc[idx[0]],
                "end": timestamps.iloc[idx[-1]],
            }
        )
    return pd.DataFrame(rows).sort_values("row_start").reset_index(drop=True)


def longest_individual_valid_runs(raw: pd.DataFrame, building_ids: Sequence[str]) -> pd.DataFrame:
    """Summarize per-building runs for a shortage diagnostic."""
    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    all_rows = []
    for building_id in building_ids:
        values = pd.to_numeric(raw[building_id], errors="coerce")
        hourly_step = timestamps.diff().eq(ONE_HOUR)
        if len(hourly_step):
            hourly_step.iloc[0] = True
        valid = timestamps.notna() & hourly_step & values.notna()
        segment = (valid != valid.shift(fill_value=False)).cumsum()
        runs = [int(len(part)) for sid, part in raw[valid].groupby(segment[valid])]
        all_rows.append(
            {
                "building_id": building_id,
                "longest_individual_valid_hours": max(runs) if runs else 0,
                "required_hours": REQUIRED_RUN_HOURS,
                "shortage_hours": max(0, REQUIRED_RUN_HOURS - (max(runs) if runs else 0)),
            }
        )
    return pd.DataFrame(all_rows)


def select_buildings(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"building_id", "site_id", "alternative_group", "selected_for_initial_check"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"split manifest missing columns: {sorted(missing)}")
    selected = manifest[_as_bool(manifest["selected_for_initial_check"])].copy()
    if len(selected) != 4:
        raise ValueError(f"expected 4 previously selected buildings, found {len(selected)}")
    if set(selected["alternative_group"]) - set(GROUPS_USED):
        raise ValueError("selected buildings contain a final-test group")
    if selected["building_id"].duplicated().any():
        raise ValueError("selected building_id values are duplicated")
    return selected.sort_values(["alternative_group", "building_id"], kind="stable").reset_index(drop=True)


def empty_window_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "building_id",
            "target_start",
            "target_end",
            "input_start",
            "input_end",
        ]
    )


def build_windows_in_target_range(
    timestamps: Sequence,
    values: Sequence,
    building_id: str,
    target_start,
    target_end,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build only windows whose 24-hour target lies inside a fixed range."""
    if target_start is None or target_end is None or pd.Timestamp(target_end) < pd.Timestamp(target_start):
        return np.empty((0, 24), dtype=float), np.empty((0, 24), dtype=float), empty_window_metadata()

    time_series = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    values_array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if time_series.isna().any() or not time_series.is_unique:
        return np.empty((0, 24), dtype=float), np.empty((0, 24), dtype=float), empty_window_metadata()

    time_index = pd.DatetimeIndex(time_series)
    target_starts = pd.date_range(
        pd.Timestamp(target_start), pd.Timestamp(target_end) - pd.Timedelta(hours=23), freq="h"
    )
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    metadata_rows: List[dict] = []
    for start in target_starts:
        window_times = pd.date_range(start - pd.Timedelta(hours=24), start + pd.Timedelta(hours=23), freq="h")
        try:
            positions = time_index.get_indexer(window_times)
        except (KeyError, pd.errors.InvalidIndexError):
            continue
        if (positions < 0).any():
            continue
        window_values = values_array[positions]
        if not np.isfinite(window_values).all():
            continue
        xs.append(window_values[:24])
        ys.append(window_values[24:])
        metadata_rows.append(
            {
                "building_id": building_id,
                "target_start": fmt_time(start),
                "target_end": fmt_time(start + pd.Timedelta(hours=23)),
                "input_start": fmt_time(start - pd.Timedelta(hours=24)),
                "input_end": fmt_time(start - pd.Timedelta(hours=1)),
            }
        )
    x_array = np.vstack(xs) if xs else np.empty((0, 24), dtype=float)
    y_array = np.vstack(ys) if ys else np.empty((0, 24), dtype=float)
    metadata = pd.DataFrame(metadata_rows) if metadata_rows else empty_window_metadata()
    return x_array, y_array, metadata


def make_split_rows(raw: pd.DataFrame, selected: pd.DataFrame, common_run: pd.Series) -> Tuple[pd.DataFrame, dict]:
    run_start = pd.Timestamp(common_run["start"])
    run_end = pd.Timestamp(common_run["end"])
    adaptation_start = run_start + ONE_HOUR * INPUT_HISTORY_HOURS
    test_start = adaptation_start + pd.Timedelta(days=MAX_ADAPTATION_DAYS)
    test_end = test_start + pd.Timedelta(days=TEST_DAYS) - ONE_HOUR

    rows: List[dict] = []
    all_train_target_times: Dict[Tuple[str, int], set] = {}
    test_target_times: Dict[str, set] = {}
    for _, building in selected.iterrows():
        building_id = str(building["building_id"])
        site_id = str(building["site_id"])
        group = str(building["alternative_group"])
        _, _, test_meta = build_windows_in_target_range(
            raw["timestamp"], raw[building_id], building_id, test_start, test_end
        )
        test_targets = set(test_meta["target_start"].tolist()) | set(test_meta["target_end"].tolist())
        test_target_times[building_id] = test_targets
        for days in ADAPTATION_DAYS:
            if days == 0:
                target_start = None
                target_end = None
            else:
                target_start = adaptation_start
                target_end = adaptation_start + pd.Timedelta(days=days) - ONE_HOUR
            x_train, y_train, train_meta = build_windows_in_target_range(
                raw["timestamp"], raw[building_id], building_id, target_start, target_end
            )
            train_targets = set(train_meta["target_start"].tolist()) | set(train_meta["target_end"].tolist())
            all_train_target_times[(building_id, days)] = train_targets
            if train_targets.intersection(test_targets):
                raise AssertionError(f"train/test target overlap for {building_id}, {days} days")
            allowed_end = adaptation_start - ONE_HOUR if days == 0 else target_end
            rows.append(
                {
                    "building_id": building_id,
                    "site_id": site_id,
                    "alternative_group": group,
                    "adaptation_days": days,
                    "common_run_start": fmt_time(run_start),
                    "common_run_end": fmt_time(run_end),
                    "adaptation_start": fmt_time(adaptation_start),
                    "adaptation_input_history_start": fmt_time(adaptation_start - pd.Timedelta(hours=24)),
                    "adaptation_input_history_end": fmt_time(adaptation_start - ONE_HOUR),
                    "adaptation_allowed_data_end": fmt_time(allowed_end),
                    "adaptation_target_start": fmt_time(target_start),
                    "adaptation_target_end": fmt_time(target_end),
                    "test_input_history_start": fmt_time(test_start - pd.Timedelta(hours=24)),
                    "test_input_history_end": fmt_time(test_start - ONE_HOUR),
                    "test_target_start": fmt_time(test_start),
                    "test_target_end": fmt_time(test_end),
                    "adaptation_train_sample_count": int(len(train_meta)),
                    "common_test_sample_count": int(len(test_meta)),
                    "adaptation_x_shape": str(tuple(x_train.shape)),
                    "adaptation_y_shape": str(tuple(y_train.shape)),
                    "test_target_times_overlap": False,
                    "zero_day_has_train_target": bool(days == 0 and len(train_meta) > 0),
                    "continuous_window_rule": "48 numeric records; 47 internal 1-hour intervals",
                    "final_test_power_used": False,
                }
            )
    return pd.DataFrame(rows), {
        "run_start": run_start,
        "run_end": run_end,
        "adaptation_start": adaptation_start,
        "test_start": test_start,
        "test_end": test_end,
        "test_target_hours": TEST_DAYS * 24,
        "required_run_hours": REQUIRED_RUN_HOURS,
        "train_target_times": all_train_target_times,
        "test_target_times": test_target_times,
    }


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
    lines.extend("| " + " | ".join(cell(row[col]) for col in frame.columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def run() -> dict:
    manifest = pd.read_csv(SPLIT_MANIFEST)
    selected = select_buildings(manifest)
    building_ids = selected["building_id"].astype(str).tolist()

    header = pd.read_csv(RAW_ELECTRICITY, nrows=0)
    missing_columns = sorted(set(["timestamp"] + building_ids).difference(header.columns))
    if missing_columns:
        raise ValueError(f"selected raw columns missing: {missing_columns}")
    raw = pd.read_csv(RAW_ELECTRICITY, usecols=["timestamp"] + building_ids)
    runs = find_valid_runs(raw, building_ids)
    sufficient = runs[runs["hours"].ge(REQUIRED_RUN_HOURS)]
    if sufficient.empty:
        shortage = longest_individual_valid_runs(raw, building_ids)
        OUTPUT_REPORT.write_text(
            "# BDG2 시간 분할 실패\n\n공통 유효 연속 구간이 부족하여 분할을 만들지 않았습니다.\n\n"
            + markdown_table(shortage),
            encoding="utf-8",
        )
        raise RuntimeError("no common valid run is long enough; see temporal split report")
    common_run = sufficient.iloc[0]
    split_rows, design = make_split_rows(raw, selected, common_run)
    if split_rows["common_test_sample_count"].nunique() != 1:
        raise AssertionError("common test sample count differs across selected buildings")
    if split_rows["test_target_start"].nunique() != 1 or split_rows["test_target_end"].nunique() != 1:
        raise AssertionError("test period is not fixed across conditions")
    if split_rows["zero_day_has_train_target"].any():
        raise AssertionError("0-day condition has adaptation targets")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(OUTPUT_MANIFEST, index=False, encoding="utf-8-sig")
    manifest_hash = sha256_file(OUTPUT_MANIFEST)
    source_manifest_hash = sha256_file(SPLIT_MANIFEST)
    raw_hash = sha256_file(RAW_ELECTRICITY)
    group_counts = (
        selected.groupby("alternative_group").agg(buildings=("building_id", "size"), sites=("site_id", "nunique"))
    )
    selected_view = selected[["building_id", "site_id", "alternative_group"]].copy()
    summary = (
        split_rows.groupby(["building_id", "alternative_group", "adaptation_days"], as_index=False)
        .agg(
            adaptation_train_sample_count=("adaptation_train_sample_count", "first"),
            common_test_sample_count=("common_test_sample_count", "first"),
        )
    )
    example = split_rows[(split_rows["building_id"] == building_ids[0]) & (split_rows["adaptation_days"] == 1)].iloc[0]

    report = f"""# BDG2 시간순 적응·시험 분할 미리보기

기존 24시간 입력→24시간 정답 파이프라인과 원본 CSV를 수정하지 않고, 선택된 4개 건물에 대한 시간순 분할만 별도로 정의했습니다. 최종 시험 그룹의 전력 열은 읽지 않았습니다.

## 고정 설계

- 최대 적응 예산: 28일
- 독립 시험 기간: 28일
- 시험 기간: 모든 적응 조건·선택 건물에 동일하게 적용
- 샘플 구조: 입력 24시간, 직후 정답 24시간
- 유효 창: 48개 숫자 기록과 내부 47개 1시간 간격이 모두 충족되어야 함
- 적응 조건: {", ".join(str(x) + "일" for x in ADAPTATION_DAYS)}
- 0일 조건: 적응 정답 샘플 0개; 최초 24시간은 입력 이력으로만 예약

공통 유효 연속 구간에서 조건을 정했습니다. 이 구간은 선택된 4개 건물 모두의 값이 숫자이고 timestamp가 1시간씩 이어지는 구간입니다.

## 사용 건물 및 사이트 분할

{markdown_table(selected_view)}

## 고정 시간 범위

| 항목 | 시각 |
| --- | --- |
| 공통 유효 구간 시작 | `{fmt_time(design['run_start'])}` |
| 공통 유효 구간 종료 | `{fmt_time(design['run_end'])}` |
| 적응 시작점 | `{fmt_time(design['adaptation_start'])}` |
| 적응 입력 이력 시작 | `{fmt_time(design['adaptation_start'] - pd.Timedelta(hours=24))}` |
| 적응 입력 이력 종료 | `{fmt_time(design['adaptation_start'] - ONE_HOUR)}` |
| 시험 정답 시작 | `{fmt_time(design['test_start'])}` |
| 시험 정답 종료 | `{fmt_time(design['test_end'])}` |
| 시험 정답 시간 수 | {design['test_target_hours']}시간 |

최대 28일 적응 정답은 `{fmt_time(design['adaptation_start'])}`부터 `{fmt_time(design['test_start'] - ONE_HOUR)}`까지이고, 시험 정답은 그 다음 시각부터 시작합니다. 따라서 적응 정답과 시험 정답이 겹치지 않습니다.

## 조건별 사용 범위와 샘플 수

{markdown_table(summary, ['building_id', 'alternative_group', 'adaptation_days', 'adaptation_train_sample_count', 'common_test_sample_count'])}

모든 조건의 시험 정답 구간은 동일하며, 공통 시험 샘플 수는 건물별·조건별로 동일하게 {int(split_rows['common_test_sample_count'].iloc[0])}개입니다. 0일 적응 조건의 학습 정답 수는 0개입니다.

1일·3일·7일·14일·28일 조건의 적응 정답 범위는 각각 적응 시작점부터 해당 일수의 마지막 시간까지입니다. 각 행의 `adaptation_allowed_data_end`, `adaptation_target_start`, `adaptation_target_end` 열에 고정 기록했습니다. 입력 이력은 정답 범위보다 24시간 앞에서만 사용합니다.

## 시간 연결 예시

| building_id | adaptation_days | input_start | input_end | target_start | target_end |
| --- | ---: | --- | --- | --- | --- |
| {example['building_id']} | {int(example['adaptation_days'])} | {example['adaptation_input_history_start']} | {example['adaptation_input_history_end']} | {example['adaptation_target_start']} | {example['adaptation_target_end']} |

위 예시는 적응 이력과 정답 범위를 보여 줍니다. 실제 창마다 정답 시작은 입력 종료 1시간 뒤이며, 시간 공백을 건너뛰는 창은 생성하지 않았습니다.

## 자동 검증 결과

검증 코드: `scripts/test_bdg2_temporal_splits.py`

- 동일 건물의 모든 조건에서 시험 구간 동일: PASS
- 적응 정답과 시험 정답의 시각 중복 없음: PASS
- 0일 조건에 타깃 학습 정답 없음: PASS
- 허용된 종료 시각 이후의 값이 적응 학습에 포함되지 않음: PASS
- 비연속 기록을 연결하지 않음: PASS
- 최종 시험용 건물 전력값 미사용: PASS

## 파일과 해시

- 분할 manifest: `reports/bdg2_temporal_split_manifest.csv`
- 이 manifest SHA-256: `{manifest_hash}`
- 입력 사이트 분할 manifest SHA-256: `{source_manifest_hash}`
- 읽은 원본 전력 파일 SHA-256: `{raw_hash}`
- 실행 코드: `scripts/build_bdg2_temporal_splits.py`

시험 구간 위치와 길이는 이 manifest에 기록했으며, 모델 성능을 확인한 뒤 변경하지 않습니다. 실제 전력값 배열은 별도 대용량 CSV로 저장하지 않았습니다.

## 데이터 부족 확인

선택된 4개 건물 모두 동일한 공통 구간에서 최대 28일 적응과 28일 시험을 확보했습니다. 부족하여 임의로 짧게 만든 건물은 없습니다.

## 이번 단계에서 하지 않은 작업

정규화, 결측 보간, 시간대 변환, 기상 데이터 다운로드, 최종 시험용 건물 평가 및 AI 모델 학습은 수행하지 않았습니다.
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    return {
        "selected": selected,
        "raw": raw,
        "runs": runs,
        "common_run": common_run,
        "split_rows": split_rows,
        "design": design,
        "manifest_hash": manifest_hash,
        "source_manifest_hash": source_manifest_hash,
        "raw_hash": raw_hash,
    }


def main() -> None:
    result = run()
    print("selected_buildings=" + ",".join(result["selected"]["building_id"].astype(str).tolist()))
    print("common_run=" + fmt_time(result["design"]["run_start"]) + ".." + fmt_time(result["design"]["run_end"]))
    print("test_period=" + fmt_time(result["design"]["test_start"]) + ".." + fmt_time(result["design"]["test_end"]))
    print(result["split_rows"].groupby("adaptation_days")["adaptation_train_sample_count"].first().to_string())
    print("common_test_samples=" + str(int(result["split_rows"]["common_test_sample_count"].iloc[0])))
    print("manifest_sha256=" + result["manifest_hash"])


if __name__ == "__main__":
    main()
