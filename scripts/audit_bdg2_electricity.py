#!/usr/bin/env python3
"""Streamed quality audit for the BDG2 electricity CSV.

The input CSV is opened read-only.  The audit makes two streaming passes:
one for timestamp structure and one for per-building value statistics and
complete hourly runs.  No values are changed or written back to the source.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
from collections import Counter
from statistics import median
from typing import Iterable, Optional


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
WINDOW_RECORDS = 48  # 24 complete hourly records before + 24 after


def configure_csv() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def parse_timestamp(raw: str) -> Optional[dt.datetime]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        # Deliberately timezone-naive: no timezone or DST conversion is done.
        return dt.datetime.strptime(text, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def iter_rows(path: str) -> tuple[object, csv.reader, list[str]]:
    handle = open(path, "r", encoding="utf-8-sig", newline="")
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration:
        handle.close()
        raise ValueError("CSV 파일이 비어 있습니다.")
    if not header or header[0].strip().lower() != "timestamp":
        handle.close()
        raise ValueError("첫 번째 열이 timestamp가 아닙니다.")
    return handle, reader, header


def timestamp_audit(path: str) -> dict:
    handle, reader, header = iter_rows(path)
    timestamps: list[Optional[dt.datetime]] = []
    row_lengths_mismatch = 0
    format_errors = 0
    empty_timestamps = 0
    valid_rows = 0
    first_valid: Optional[dt.datetime] = None
    last_valid: Optional[dt.datetime] = None
    minimum: Optional[dt.datetime] = None
    maximum: Optional[dt.datetime] = None
    counts: Counter[dt.datetime] = Counter()
    deltas: list[int] = []
    previous_valid: Optional[dt.datetime] = None
    data_rows = 0

    try:
        for row in reader:
            data_rows += 1
            if len(row) != len(header):
                row_lengths_mismatch += 1
            raw = row[0] if row else ""
            ts = parse_timestamp(raw)
            timestamps.append(ts)
            if ts is None:
                format_errors += 1
                if not raw.strip():
                    empty_timestamps += 1
                continue
            valid_rows += 1
            counts[ts] += 1
            if first_valid is None:
                first_valid = ts
            last_valid = ts
            minimum = ts if minimum is None or ts < minimum else minimum
            maximum = ts if maximum is None or ts > maximum else maximum
            if previous_valid is not None:
                deltas.append(int((ts - previous_valid).total_seconds()))
            previous_valid = ts
    finally:
        handle.close()

    positive = Counter(delta for delta in deltas if delta > 0)
    if positive:
        expected_seconds = max(positive.items(), key=lambda item: (item[1], -item[0]))[0]
    elif deltas:
        expected_seconds = max(Counter(deltas).items(), key=lambda item: (item[1], -item[0]))[0]
    else:
        expected_seconds = None

    duplicate_stamps = sum(1 for count in counts.values() if count > 1)
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    gap_segments = 0
    missing_expected_slots = 0
    irregular_gap_segments = 0
    out_of_order = 0
    zero_deltas = 0
    non_expected = 0
    if expected_seconds is not None and expected_seconds > 0:
        for delta in deltas:
            if delta == 0:
                zero_deltas += 1
            if delta < 0:
                out_of_order += 1
            if delta != expected_seconds:
                non_expected += 1
            if delta > expected_seconds:
                gap_segments += 1
                if delta % expected_seconds == 0:
                    missing_expected_slots += delta // expected_seconds - 1
                else:
                    irregular_gap_segments += 1

    return {
        "header": header,
        "building_ids": header[1:],
        "timestamps": timestamps,
        "data_rows": data_rows,
        "row_lengths_mismatch": row_lengths_mismatch,
        "format_errors": format_errors,
        "empty_timestamps": empty_timestamps,
        "valid_rows": valid_rows,
        "first_valid": first_valid,
        "last_valid": last_valid,
        "minimum": minimum,
        "maximum": maximum,
        "duplicate_stamps": duplicate_stamps,
        "duplicate_rows": duplicate_rows,
        "deltas": deltas,
        "expected_seconds": expected_seconds,
        "gap_segments": gap_segments,
        "missing_expected_slots": missing_expected_slots,
        "irregular_gap_segments": irregular_gap_segments,
        "out_of_order": out_of_order,
        "zero_deltas": zero_deltas,
        "non_expected": non_expected,
        "delta_distribution": Counter(deltas),
    }


def classify_value(raw: str) -> tuple[str, Optional[float]]:
    """Return ('missing'|'nonnumeric'|'numeric', finite numeric value)."""
    text = (raw or "").strip()
    if text.lower() in MISSING_TOKENS:
        return "missing", None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return "nonnumeric", None
    if math.isnan(value):
        return "missing", None
    if not math.isfinite(value):
        return "nonnumeric", None
    return "numeric", value


def complete_run_update(
    index: int,
    row_index: int,
    timestamp: Optional[dt.datetime],
    current_lengths: list[int],
    current_starts: list[Optional[int]],
    best_lengths: list[int],
    best_starts: list[Optional[int]],
    best_ends: list[Optional[int]],
    window_counts: list[int],
) -> None:
    """Finalize one building's current run at a non-complete row."""
    length = current_lengths[index]
    if length <= 0:
        return
    start = current_starts[index]
    if length >= WINDOW_RECORDS:
        window_counts[index] += length - WINDOW_RECORDS + 1
    if length > best_lengths[index]:
        best_lengths[index] = length
        best_starts[index] = start
        best_ends[index] = row_index - 1
    current_lengths[index] = 0
    current_starts[index] = None


def value_audit(path: str, timestamp_info: dict) -> list[dict]:
    building_ids = timestamp_info["building_ids"]
    n_buildings = len(building_ids)
    total = [0] * n_buildings
    missing = [0] * n_buildings
    nonnumeric = [0] * n_buildings
    zeros = [0] * n_buildings
    negatives = [0] * n_buildings
    numeric = [0] * n_buildings
    current_lengths = [0] * n_buildings
    current_starts: list[Optional[int]] = [None] * n_buildings
    best_lengths = [0] * n_buildings
    best_starts: list[Optional[int]] = [None] * n_buildings
    best_ends: list[Optional[int]] = [None] * n_buildings
    window_counts = [0] * n_buildings
    expected_seconds = timestamp_info["expected_seconds"]
    timestamps: list[Optional[dt.datetime]] = timestamp_info["timestamps"]
    if expected_seconds is None or expected_seconds <= 0:
        expected_seconds = 0

    handle, reader, header = iter_rows(path)
    data_rows = 0
    row_lengths_mismatch = 0
    try:
        for row_index, row in enumerate(reader):
            data_rows += 1
            if len(row) != len(header):
                row_lengths_mismatch += 1
            current_ts = timestamps[row_index] if row_index < len(timestamps) else None
            previous_ts = timestamps[row_index - 1] if row_index > 0 else None
            timestamp_continuous = (
                expected_seconds > 0
                and current_ts is not None
                and previous_ts is not None
                and int((current_ts - previous_ts).total_seconds()) == expected_seconds
            )
            for index in range(n_buildings):
                total[index] += 1
                raw = row[index + 1] if index + 1 < len(row) else ""
                kind, value = classify_value(raw)
                if kind == "missing":
                    missing[index] += 1
                elif kind == "nonnumeric":
                    nonnumeric[index] += 1
                else:
                    numeric[index] += 1
                    if value == 0:
                        zeros[index] += 1
                    elif value < 0:
                        negatives[index] += 1

                candidate = kind == "numeric" and current_ts is not None
                if candidate and current_lengths[index] > 0 and timestamp_continuous:
                    current_lengths[index] += 1
                elif candidate:
                    # A valid numeric row starts a new run after a gap, an
                    # invalid timestamp, or an unavailable previous value.
                    complete_run_update(
                        index,
                        row_index,
                        current_ts,
                        current_lengths,
                        current_starts,
                        best_lengths,
                        best_starts,
                        best_ends,
                        window_counts,
                    )
                    current_lengths[index] = 1
                    current_starts[index] = row_index
                else:
                    complete_run_update(
                        index,
                        row_index,
                        current_ts,
                        current_lengths,
                        current_starts,
                        best_lengths,
                        best_starts,
                        best_ends,
                        window_counts,
                    )
    finally:
        handle.close()

    # Finalize runs that reach end of file.
    for index in range(n_buildings):
        complete_run_update(
            index,
            data_rows,
            timestamps[-1] if timestamps else None,
            current_lengths,
            current_starts,
            best_lengths,
            best_starts,
            best_ends,
            window_counts,
        )

    rows: list[dict] = []
    expected_hours = expected_seconds / 3600 if expected_seconds else None
    for index, building_id in enumerate(building_ids):
        start_index = best_starts[index]
        end_index = best_ends[index]
        start_ts = timestamps[start_index] if start_index is not None and start_index < len(timestamps) else None
        end_ts = timestamps[end_index] if end_index is not None and end_index < len(timestamps) else None
        length = best_lengths[index]
        rows.append(
            {
                "building_id": building_id,
                "total_records": total[index],
                "numeric_records": numeric[index],
                "nonnumeric_count": nonnumeric[index],
                "missing_count": missing[index],
                "zero_count": zeros[index],
                "negative_count": negatives[index],
                "longest_complete_run_records": length,
                "longest_complete_run_coverage_hours": (length * expected_hours if expected_hours else 0),
                "longest_complete_run_elapsed_hours": ((length - 1) * expected_hours if length and expected_hours else 0),
                "longest_complete_run_start": start_ts.strftime(TIMESTAMP_FORMAT) if start_ts else "",
                "longest_complete_run_end": end_ts.strftime(TIMESTAMP_FORMAT) if end_ts else "",
                "forecast_windows_24h_past_24h_future": window_counts[index],
            }
        )
    if row_lengths_mismatch != timestamp_info["row_lengths_mismatch"]:
        raise ValueError("두 번째 CSV pass에서 행 필드 수 불일치 개수가 달라졌습니다.")
    return rows


def format_dt(value: Optional[dt.datetime]) -> str:
    return value.strftime(TIMESTAMP_FORMAT) if value else ""


def write_results(result_rows: list[dict], output_csv: str, output_md: str, timestamp_info: dict, input_path: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    fieldnames = list(result_rows[0].keys()) if result_rows else ["building_id"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)

    expected_seconds = timestamp_info["expected_seconds"]
    expected_text = f"{expected_seconds / 3600:g}시간" if expected_seconds else "확인 불가"
    distribution = timestamp_info["delta_distribution"].most_common(8)
    total_missing = sum(row["missing_count"] for row in result_rows)
    total_nonnumeric = sum(row["nonnumeric_count"] for row in result_rows)
    total_zero = sum(row["zero_count"] for row in result_rows)
    total_negative = sum(row["negative_count"] for row in result_rows)
    columns_with_missing = sum(row["missing_count"] > 0 for row in result_rows)
    columns_with_nonnumeric = sum(row["nonnumeric_count"] > 0 for row in result_rows)
    columns_with_zero = sum(row["zero_count"] > 0 for row in result_rows)
    columns_with_negative = sum(row["negative_count"] > 0 for row in result_rows)
    run_lengths = [row["longest_complete_run_records"] for row in result_rows]
    windows = [row["forecast_windows_24h_past_24h_future"] for row in result_rows]
    summary = (
        f"검사 대상 데이터 행: {timestamp_info['data_rows']:,}\n"
        f"전력 열: {len(result_rows):,}\n"
        f"timestamp 시작(첫 유효): `{format_dt(timestamp_info['first_valid'])}`\n"
        f"timestamp 종료(마지막 유효): `{format_dt(timestamp_info['last_valid'])}`\n"
        f"timestamp 유효 범위(min/max): `{format_dt(timestamp_info['minimum'])}` ~ `{format_dt(timestamp_info['maximum'])}`\n"
        f"관측된 최빈 양의 간격: **{expected_text}** ({expected_seconds or 'NA'}초)\n"
        f"날짜·시간 형식 오류: {timestamp_info['format_errors']:,} (그중 빈 timestamp: {timestamp_info['empty_timestamps']:,})\n"
        f"중복 timestamp 값 수: {timestamp_info['duplicate_stamps']:,}; 중복 행 수: {timestamp_info['duplicate_rows']:,}\n"
        f"최빈 간격 기준 비정상 간격 수: {timestamp_info['non_expected']:,}; 순서 역전: {timestamp_info['out_of_order']:,}; 인접 중복 간격: {timestamp_info['zero_deltas']:,}\n"
        f"간격이 비어 있는 구간 수: {timestamp_info['gap_segments']:,}; 누락된 예상 시간 슬롯 수: {timestamp_info['missing_expected_slots']:,}; 비정수 간격 구간: {timestamp_info['irregular_gap_segments']:,}\n"
        f"행 필드 수 불일치: {timestamp_info['row_lengths_mismatch']:,}\n"
    )
    value_summary = (
        f"전체 전력값 검사 건수: {timestamp_info['data_rows'] * len(result_rows):,}\n"
        f"숫자값: {sum(row['numeric_records'] for row in result_rows):,}\n"
        f"숫자로 해석할 수 없는 값: {total_nonnumeric:,} ({columns_with_nonnumeric:,}개 열에서 1개 이상)\n"
        f"결측값(빈 문자열/NA/NaN/null 등): {total_missing:,} ({columns_with_missing:,}개 열에서 1개 이상)\n"
        f"0값: {total_zero:,} ({columns_with_zero:,}개 열에서 1개 이상)\n"
        f"음수값: {total_negative:,} ({columns_with_negative:,}개 열에서 1개 이상)\n"
    )
    run_summary = (
        f"최장 완전 연속 구간(시간당 유효 숫자값이며 timestamp도 최빈 간격으로 연속인 경우):\n"
        f"- 열별 레코드 수 기준 최소/중앙값/최대: {min(run_lengths):,} / {median(run_lengths):g} / {max(run_lengths):,}\n"
        f"- 24시간 과거 + 24시간 미래 예측 구간 수 합계: {sum(windows):,}\n"
        f"- 예측 구간이 0개인 열: {sum(value == 0 for value in windows):,}\n"
    )
    lines = [
        "# BDG2 electricity.csv 품질 검사",
        "",
        "- 입력 파일: `data/raw/bdg2/electricity.csv`",
        f"- 검사 스크립트: `scripts/audit_bdg2_electricity.py`",
        "- 원본 CSV는 수정하지 않았으며, 기상 데이터 다운로드·결측 보간·전처리·건물 선정·학습은 수행하지 않았다.",
        "- 결과 CSV: `reports/bdg2_electricity_quality.csv`",
        "",
        "## 1. 시간 정보",
        "",
        "```text",
        summary.rstrip(),
        "```",
        "",
        "시간 간격 분포 상위(연속적으로 유효한 timestamp 사이, 초 단위):",
        "",
        "| 간격(초) | 횟수 |",
        "|---:|---:|",
    ]
    lines.extend(f"| {delta} | {count:,} |" for delta, count in distribution)
    lines.extend(
        [
            "",
            "## 2. 전력값 통계",
            "",
            "```text",
            value_summary.rstrip(),
            "```",
            "",
            "0값과 음수값은 삭제하거나 결측으로 변환하지 않고 발견 개수만 기록했다.",
            "",
            "## 3. 연속 구간과 예측 구간 정의",
            "",
            "```text",
            run_summary.rstrip(),
            "```",
            "",
            "- 완전 연속 구간은 유효한 timestamp가 최빈 간격으로 이어지고, 해당 건물 값이 유한한 숫자인 행의 연속으로 정의했다. 빈 값·NA·NaN·null 등 결측과 숫자로 변환되지 않는 값은 구간을 끊는다.",
            "- 예측 구간 1개는 연속된 시간당 레코드 48개(과거 24개 + 미래 24개)로 정의했으며, 각 건물의 완전 연속 구간에서 가능한 시작 위치를 합산했다.",
            "- 시간대 변환이나 서머타임 보정은 하지 않았다. timestamp는 파일에 기록된 문자열을 그대로 파싱했다.",
            "",
            "## 4. 확인이 필요한 문제",
            "",
        ]
    )
    issues = []
    if timestamp_info["format_errors"]:
        issues.append(f"timestamp 형식 오류 {timestamp_info['format_errors']:,}건")
    if timestamp_info["duplicate_rows"]:
        issues.append(f"중복 timestamp 행 {timestamp_info['duplicate_rows']:,}건")
    if timestamp_info["gap_segments"]:
        issues.append(f"최빈 간격 기준 시간 공백 {timestamp_info['gap_segments']:,}구간({timestamp_info['missing_expected_slots']:,} 슬롯)")
    if timestamp_info["row_lengths_mismatch"]:
        issues.append(f"행 필드 수 불일치 {timestamp_info['row_lengths_mismatch']:,}건")
    if total_missing:
        issues.append(f"결측값 {total_missing:,}건")
    if total_nonnumeric:
        issues.append(f"비수치 값 {total_nonnumeric:,}건")
    if total_zero:
        issues.append(f"0값 {total_zero:,}건")
    if total_negative:
        issues.append(f"음수값 {total_negative:,}건")
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- 이번 규칙으로 집계된 확인 필요 항목은 없다.")
    lines.extend(
        [
            "",
            "## 5. 해석상 제한",
            "",
            "- 이 검사는 값의 통계적 존재 여부와 시간 연속성만 확인한다. 0이나 음수가 물리적으로 타당한지는 판정하지 않는다.",
            "- 계량 열이 건물 전체 계량인지 하위계량기인지의 물리적 범위는 이 검사로 판단하지 않는다.",
            "- `building_id`와 결과 통계의 연결은 원본 electricity.csv 헤더 순서를 그대로 사용했다.",
        ]
    )
    with open(output_md, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "data", "raw", "bdg2", "electricity.csv"))
    parser.add_argument("--output-csv", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_electricity_quality.csv"))
    parser.add_argument("--output-md", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_electricity_quality.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_csv()
    try:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(args.input)
        timestamp_info = timestamp_audit(args.input)
        result_rows = value_audit(args.input, timestamp_info)
        write_results(result_rows, args.output_csv, args.output_md, timestamp_info, args.input)
        print(f"완료: {len(result_rows):,}개 전력 열, {timestamp_info['data_rows']:,}개 데이터 행")
        print(f"결과 CSV: {args.output_csv}")
        print(f"결과 보고서: {args.output_md}")
        return 0
    except MemoryError:
        print("메모리 부족으로 검사를 중단했습니다. 결과 파일을 완성하지 못했습니다.", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - command-line diagnostic
        print(f"검사 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
