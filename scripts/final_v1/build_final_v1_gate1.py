"""Gate 1: first final-power access and frozen temporal structure validation."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"
SUPPORT_DIR = REPORTS / "final_v1"
RAW = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
COHORT = REPORTS / "final_v1_cohort_manifest.csv"
PROTOCOL = REPORTS / "final_v1_protocol_manifest.md"
V5_PROTOCOL = REPORTS / "bdg2_v5_final_analysis_protocol_draft.md"
TEMPORAL_OUT = REPORTS / "final_v1_temporal_manifest.csv"
REPORT_OUT = REPORTS / "final_v1_structure_validation.md"
LOG_OUT = SUPPORT_DIR / "gate1_execution.log"
FAILURES_OUT = SUPPORT_DIR / "gate1_data_integrity_failures.csv"

ADAPTATION_DAYS = (0, 1, 3, 7, 14, 28)
EXPECTED_RAW_SHA256 = "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_signature(values: pd.DatetimeIndex) -> str:
    payload = "\n".join(ts.isoformat(sep=" ") for ts in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def longest_complete_run(
    timestamps: pd.Series | pd.DatetimeIndex, values: pd.Series | np.ndarray
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Exact v2 rule: longest finite one-hour run, earliest run wins ties."""
    ts = pd.DatetimeIndex(timestamps)
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric)
    best_start = best_end = -1
    best_len = 0
    current_start: int | None = None
    for i in range(len(ts)):
        continues = (
            i > 0
            and valid[i]
            and valid[i - 1]
            and ts[i] - ts[i - 1] == pd.Timedelta(hours=1)
        )
        if valid[i] and (current_start is None or not continues):
            current_start = i
        if current_start is not None:
            segment_ends = i == len(ts) - 1 or not (
                valid[i + 1]
                and valid[i]
                and ts[i + 1] - ts[i] == pd.Timedelta(hours=1)
            )
            if segment_ends:
                length = i - current_start + 1
                if length > best_len:
                    best_start, best_end, best_len = current_start, i, length
                current_start = None
    if best_len == 0:
        raise AssertionError("No valid hourly run")
    return pd.Timestamp(ts[best_start]), pd.Timestamp(ts[best_end]), best_len


def range_is_complete(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    expected = pd.date_range(start, end, freq="h")
    values = series.reindex(expected).to_numpy(float)
    return len(values) == len(expected) and np.isfinite(values).all()


def frozen_boundaries(run_end: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """The exact development split rule from prepare_bdg2_multibuilding_v2.py."""
    common_start = run_end - pd.Timedelta(days=57) + pd.Timedelta(hours=1)
    adaptation_start = common_start + pd.Timedelta(hours=24)
    test_start = adaptation_start + pd.Timedelta(days=28)
    return {
        "common_start": common_start,
        "initial_history_start": common_start,
        "initial_history_end": adaptation_start - pd.Timedelta(hours=1),
        "adaptation_start": adaptation_start,
        "test_start": test_start,
        "test_end": run_end,
    }


def validate_building_structure(
    building_id: str,
    series: pd.Series,
) -> tuple[list[dict], dict, list[str]]:
    run_start, run_end, run_hours = longest_complete_run(series.index, series.to_numpy())
    bounds = frozen_boundaries(run_end)
    reasons: list[str] = []
    if run_hours < 90 * 24:
        reasons.append("actual_longest_complete_run_lt_90d")
    if bounds["common_start"] < run_start:
        reasons.append("frozen_57d_range_precedes_longest_run")
    if not range_is_complete(series, bounds["common_start"], bounds["test_end"]):
        reasons.append("frozen_57d_range_not_complete_hourly_numeric")

    test_starts = pd.date_range(
        bounds["test_start"], bounds["test_end"] - pd.Timedelta(hours=23), freq="24h"
    )
    test_targets = pd.date_range(bounds["test_start"], bounds["test_end"], freq="h")
    if len(test_starts) != 28:
        reasons.append(f"daily_test_window_count_{len(test_starts)}_not_28")
    if len(test_targets) != 672:
        reasons.append(f"test_target_hours_{len(test_targets)}_not_672")

    x_y_checks = []
    for origin in test_starts:
        x_times = pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="h")
        y_times = pd.date_range(origin, periods=24, freq="h")
        x_values = series.reindex(x_times).to_numpy(float)
        y_values = series.reindex(y_times).to_numpy(float)
        x_y_checks.append(
            len(x_times) == 24
            and len(y_times) == 24
            and np.isfinite(x_values).all()
            and np.isfinite(y_values).all()
            and x_times[-1] + pd.Timedelta(hours=1) == y_times[0]
            and set(x_times).isdisjoint(set(y_times))
        )
    if not all(x_y_checks):
        reasons.append("one_or_more_test_windows_failed_X24_y24_contiguity")

    previous_day_times = test_targets - pd.Timedelta(hours=24)
    weekly_times = test_targets - pd.Timedelta(hours=168)
    previous_day_available = int(np.isfinite(series.reindex(previous_day_times).to_numpy(float)).sum())
    weekly_available = int(np.isfinite(series.reindex(weekly_times).to_numpy(float)).sum())
    if previous_day_available != 672:
        reasons.append(f"previous_day_available_{previous_day_available}_not_672")
    if weekly_available != 672:
        reasons.append(f"weekly_available_{weekly_available}_not_672")

    temporal_rows = []
    adaptation_target_sets: dict[int, set[pd.Timestamp]] = {}
    for days in ADAPTATION_DAYS:
        if days == 0:
            adaptation_end = pd.NaT
            adaptation_count = 0
            adaptation_targets = pd.DatetimeIndex([])
        else:
            adaptation_end = (
                bounds["adaptation_start"] + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
            )
            starts = pd.date_range(
                bounds["adaptation_start"], adaptation_end - pd.Timedelta(hours=23), freq="h"
            )
            expected_count = days * 24 - 23
            adaptation_count = 0
            targets: set[pd.Timestamp] = set()
            for origin in starts:
                times = pd.date_range(origin - pd.Timedelta(hours=24), periods=48, freq="h")
                values = series.reindex(times).to_numpy(float)
                if np.isfinite(values).all():
                    adaptation_count += 1
                    targets.update(pd.date_range(origin, periods=24, freq="h"))
            adaptation_targets = pd.DatetimeIndex(sorted(targets))
            if adaptation_count != expected_count:
                reasons.append(f"adaptation_{days}d_samples_{adaptation_count}_not_{expected_count}")
        adaptation_target_sets[days] = set(adaptation_targets)
        overlap = len(set(adaptation_targets).intersection(set(test_targets)))
        if overlap:
            reasons.append(f"adaptation_{days}d_test_target_overlap_{overlap}")
        temporal_rows.append(
            {
                "building_id": building_id,
                "adaptation_days": days,
                "actual_longest_run_start": run_start,
                "actual_longest_run_end": run_end,
                "actual_longest_run_hours": run_hours,
                "initial_history_start": bounds["initial_history_start"],
                "initial_history_end": bounds["initial_history_end"],
                "adaptation_target_start": bounds["adaptation_start"] if days else pd.NaT,
                "adaptation_target_end": adaptation_end,
                "adaptation_train_sample_count": adaptation_count,
                "test_target_start": bounds["test_start"],
                "test_target_end": bounds["test_end"],
                "daily_test_sample_count": len(test_starts),
                "test_target_hour_count": len(test_targets),
                "test_target_timestamp_sha256": timestamp_signature(test_targets),
                "forecast_origin_sha256": timestamp_signature(test_starts),
                "adaptation_test_target_overlap_count": overlap,
                "previous_day_available_target_count": previous_day_available,
                "weekly_same_hour_available_target_count": weekly_available,
                "all_test_windows_X24_y24_contiguous": all(x_y_checks),
            }
        )

    summary = {
        "actual_longest_run_start": run_start,
        "actual_longest_run_end": run_end,
        "actual_longest_run_hours": run_hours,
        "test_target_start": bounds["test_start"],
        "test_target_end": bounds["test_end"],
        "daily_test_sample_count": len(test_starts),
        "test_target_timestamp_sha256": timestamp_signature(test_targets),
        "structure_status": "PASS" if not reasons else "data_integrity_failure",
    }
    return temporal_rows, summary, sorted(set(reasons))


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    for path in (TEMPORAL_OUT, REPORT_OUT, LOG_OUT, FAILURES_OUT):
        if path.exists():
            raise FileExistsError(f"Gate 1 refuses to overwrite existing artifact: {path}")
    cohort = pd.read_csv(COHORT)
    selected = cohort.loc[cohort.selection_status.eq("selected_for_final_v1")].copy()
    if len(selected) != 453 or selected.site_id.nunique() != 7:
        raise AssertionError("Frozen Gate 0 cohort changed")

    raw_hash = sha256(RAW)
    if raw_hash != EXPECTED_RAW_SHA256:
        raise AssertionError(f"Raw electricity hash changed: {raw_hash}")
    header = pd.read_csv(RAW, nrows=0).columns.tolist()
    requested = ["timestamp", *selected.building_id.sort_values().tolist()]
    missing = sorted(set(requested).difference(header))
    if missing:
        raise AssertionError(f"Final selected power columns missing: {missing}")
    raw = pd.read_csv(RAW, usecols=requested)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    if raw.timestamp.duplicated().any():
        raise AssertionError("Raw timestamps duplicated")
    if not raw.timestamp.is_monotonic_increasing:
        raise AssertionError("Raw timestamps are not sorted")
    index = pd.DatetimeIndex(raw.timestamp)

    temporal_rows: list[dict] = []
    building_summaries: list[dict] = []
    failures: list[dict] = []
    metadata = selected.set_index("building_id")
    for building_id in sorted(selected.building_id):
        series = pd.Series(
            pd.to_numeric(raw[building_id], errors="coerce").to_numpy(float),
            index=index,
            name=building_id,
        )
        rows, summary, reasons = validate_building_structure(building_id, series)
        info = metadata.loc[building_id]
        for row in rows:
            row.update({
                "site_id": info.site_id,
                "primaryspaceusage": info.primaryspaceusage,
                "gate0_selection_status": info.selection_status,
                "structure_status": summary["structure_status"],
                "failure_reason": ";".join(reasons),
                "temporal_rule_version": "v2_longest_run_end_57d_context24_adapt28_test28",
                "performance_used_to_choose_period": False,
            })
            temporal_rows.append(row)
        building_summaries.append({
            "building_id": building_id,
            "site_id": info.site_id,
            "primaryspaceusage": info.primaryspaceusage,
            **summary,
            "failure_reason": ";".join(reasons),
        })
        if reasons:
            failures.append({
                "building_id": building_id,
                "site_id": info.site_id,
                "failure_reason": ";".join(reasons),
            })

    temporal = pd.DataFrame(temporal_rows).sort_values(["site_id", "building_id", "adaptation_days"])
    temporal.to_csv(TEMPORAL_OUT, index=False, encoding="utf-8", lineterminator="\n")
    failure_frame = pd.DataFrame(failures, columns=["building_id", "site_id", "failure_reason"])
    failure_frame.to_csv(FAILURES_OUT, index=False, encoding="utf-8", lineterminator="\n")
    summary_frame = pd.DataFrame(building_summaries)
    passed = summary_frame.loc[summary_frame.structure_status.eq("PASS")]

    site_rows = []
    for site_id, frame in summary_frame.groupby("site_id", sort=True):
        site_rows.append([
            site_id,
            len(frame),
            int(frame.structure_status.eq("PASS").sum()),
            int(frame.structure_status.ne("PASS").sum()),
            frame.loc[frame.structure_status.eq("PASS"), "test_target_start"].min() if frame.structure_status.eq("PASS").any() else "NA",
            frame.loc[frame.structure_status.eq("PASS"), "test_target_end"].max() if frame.structure_status.eq("PASS").any() else "NA",
        ])
    structure_status = "PASS" if len(passed) else "FAIL"
    inference_status = (
        "eligible_for_site_cluster_bootstrap"
        if passed.site_id.nunique() >= 5
        else "inconclusive_fewer_than_5_evaluable_sites"
    )
    failure_rows = (
        failure_frame[["building_id", "site_id", "failure_reason"]].values.tolist()
        if len(failure_frame)
        else [["없음", "없음", "없음"]]
    )
    report = f"""# final_v1 Gate 1 구조 검증

## 판정

- Gate 1 전체 상태: **{structure_status}**
- Gate 0 적격: {len(selected)}개 건물, {selected.site_id.nunique()}개 사이트
- 구조 검증 통과: **{len(passed)}개 건물, {passed.site_id.nunique()}개 사이트**
- `data_integrity_failure`: **{len(failure_frame)}개 건물**
- H1 cluster bootstrap 가능성: `{inference_status}`
- 성능 지표 계산·모델 실행: **아직 수행하지 않음**

## 최초 final 전력 접근

- 원본: `data/raw/bdg2/electricity.csv`
- SHA-256: `{raw_hash}` (기존 고정값과 일치)
- 요청 전력 열: Gate 0에서 동결한 453개만 사용
- 제외된 57개와 다른 final 건물의 전력 열: 읽지 않음
- timestamp 행: {len(raw)}; 중복 0; 파싱 오류 0; 오름차순 확인

## 동일한 v2 시간 구간 생성 규칙

각 건물의 가장 긴 유효 1시간 연속 구간을 찾고, 그 끝에서 역산하여 `57일 = 초기 입력 24시간 + 최대 적응 28일 + 시험 28일`을 고정했다. 동일 길이 최장 구간이 여러 개면 기존 코드와 같이 가장 이른 구간을 사용한다. 성능을 보거나 다른 기간을 탐색하지 않았다.

{md_table(["site_id", "Gate0 적격", "PASS", "실패", "가장 이른 시험 시작", "가장 늦은 시험 종료"], site_rows)}

## 자동 구조 검사

- 실제 최장 완전 연속 기록 ≥90일
- 고정 57일 구간의 모든 값이 유효 숫자이며 timestamp 1시간 연속
- 모든 예산의 적응 샘플 수: `0, 1, 49, 145, 313, 649`
- 28개 일별 비중첩 시험창, 각 X=24와 y=24
- X 마지막 시각과 y 첫 시각 1시간 차이, X/y timestamp 교집합 0
- 적응 정답과 시험 정답 timestamp 교집합 0
- 전일·전주 기준선 각각 672개 예측값 가용
- 예산·시드·기준선 공통 시험 정답 timestamp signature 고정

## 실패 건물

{md_table(["building_id", "site_id", "정확한 사유"], failure_rows)}

실패 건물은 대체하거나 기간을 이동하지 않는다. Gate 2는 구조 검증 PASS 건물에만 단 한 번 실행한다.

## 해시

- Gate 0 cohort: `{sha256(COHORT)}`
- Gate 0 protocol manifest: `{sha256(PROTOCOL)}`
- v5 final protocol: `{sha256(V5_PROTOCOL)}`
- final temporal manifest: `{sha256(TEMPORAL_OUT)}`

## 데이터 누수 상태

적응 정답과 시험 정답의 교집합은 통과 건물에서 모두 0이다. 시험 입력은 각 forecast origin 직전 24시간으로 끝나며 미래 실제 전력을 포함하지 않는다. 이 단계에서는 모델 성능을 계산하지 않았다.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    LOG_OUT.write_text(
        "\n".join([
            f"completed_utc={datetime.now(timezone.utc).isoformat()}",
            "gate=1",
            f"status={structure_status}",
            "first_final_power_access=True",
            "performance_metric_computed=False",
            f"raw_sha256={raw_hash}",
            f"requested_power_columns={len(selected)}",
            f"raw_rows={len(raw)}",
            f"passed_buildings={len(passed)}",
            f"passed_sites={passed.site_id.nunique()}",
            f"failed_buildings={len(failure_frame)}",
            f"temporal_sha256={sha256(TEMPORAL_OUT)}",
            f"elapsed_seconds={time.time() - started:.3f}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Gate1 {structure_status}: pass={len(passed)} sites={passed.site_id.nunique()} failures={len(failure_frame)}")
    print(f"temporal_sha256={sha256(TEMPORAL_OUT)}")


if __name__ == "__main__":
    main()
