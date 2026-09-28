#!/usr/bin/env python3
"""Audit zero and missing-value distributions in BDG2 electricity data.

The raw electricity CSV is read-only.  Existing per-building counts from
bdg2_electricity_quality.csv are joined to a streamed pass over the raw file
to find the longest consecutive zero run for every building.
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
from typing import Optional


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


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
        return dt.datetime.strptime(text, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def classify_value(raw: str) -> tuple[str, Optional[float]]:
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


def open_csv(path: str) -> tuple[object, csv.reader, list[str]]:
    handle = open(path, "r", encoding="utf-8-sig", newline="")
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration:
        handle.close()
        raise ValueError(f"빈 CSV 파일: {path}")
    if not header or header[0].strip().lower() != "timestamp":
        handle.close()
        raise ValueError("electricity.csv의 첫 열이 timestamp가 아닙니다.")
    return handle, reader, header


def load_quality(path: str) -> tuple[dict[str, dict], list[str]]:
    quality: dict[str, dict] = {}
    order: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "building_id",
            "total_records",
            "numeric_records",
            "nonnumeric_count",
            "missing_count",
            "zero_count",
            "negative_count",
            "longest_complete_run_records",
            "forecast_windows_24h_past_24h_future",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"품질 결과에 필요한 열이 없습니다: {sorted(missing)}")
        for row in reader:
            building_id = row["building_id"]
            if building_id in quality:
                raise ValueError(f"품질 결과에 중복 building_id: {building_id}")
            quality[building_id] = row
            order.append(building_id)
    return quality, order


def read_timestamps(path: str) -> dict:
    handle, reader, header = open_csv(path)
    timestamps: list[Optional[dt.datetime]] = []
    raw_rows = 0
    format_errors = 0
    deltas: list[int] = []
    previous: Optional[dt.datetime] = None
    try:
        for row in reader:
            raw_rows += 1
            raw = row[0] if row else ""
            current = parse_timestamp(raw)
            timestamps.append(current)
            if current is None:
                format_errors += 1
                continue
            if previous is not None:
                deltas.append(int((current - previous).total_seconds()))
            previous = current
    finally:
        handle.close()
    positive = Counter(delta for delta in deltas if delta > 0)
    expected = max(positive.items(), key=lambda item: (item[1], -item[0]))[0] if positive else None
    gap_segments = 0
    missing_slots = 0
    if expected and expected > 0:
        for delta in deltas:
            if delta > expected:
                gap_segments += 1
                if delta % expected == 0:
                    missing_slots += delta // expected - 1
    return {
        "header": header,
        "timestamps": timestamps,
        "data_rows": raw_rows,
        "format_errors": format_errors,
        "deltas": deltas,
        "expected_seconds": expected,
        "gap_segments": gap_segments,
        "missing_slots": missing_slots,
    }


def finish_zero_run(
    index: int,
    end_index_exclusive: int,
    current_lengths: list[int],
    current_starts: list[Optional[int]],
    best_lengths: list[int],
    best_starts: list[Optional[int]],
    best_ends: list[Optional[int]],
) -> None:
    length = current_lengths[index]
    if length <= 0:
        return
    if length > best_lengths[index]:
        best_lengths[index] = length
        best_starts[index] = current_starts[index]
        best_ends[index] = end_index_exclusive - 1
    current_lengths[index] = 0
    current_starts[index] = None


def find_zero_runs(path: str, timestamp_info: dict, quality: dict[str, dict], building_ids: list[str]) -> dict:
    timestamps = timestamp_info["timestamps"]
    expected = timestamp_info["expected_seconds"] or 0
    n = len(building_ids)
    current_lengths = [0] * n
    current_starts: list[Optional[int]] = [None] * n
    best_lengths = [0] * n
    best_starts: list[Optional[int]] = [None] * n
    best_ends: list[Optional[int]] = [None] * n
    raw_zero_counts = [0] * n
    raw_numeric_counts = [0] * n
    raw_missing_counts = [0] * n
    row_length_mismatch = 0
    header_mismatch = []

    handle, reader, header = open_csv(path)
    if header[1:] != building_ids:
        header_mismatch = [
            f"품질 CSV 순서와 raw 헤더가 일치하지 않음 (raw={len(header)-1}, quality={len(building_ids)})"
        ]
    try:
        for row_index, row in enumerate(reader):
            if len(row) != len(header):
                row_length_mismatch += 1
            current_ts = timestamps[row_index] if row_index < len(timestamps) else None
            previous_ts = timestamps[row_index - 1] if row_index > 0 else None
            continuous_timestamp = (
                expected > 0
                and current_ts is not None
                and previous_ts is not None
                and int((current_ts - previous_ts).total_seconds()) == expected
            )
            for index in range(n):
                raw = row[index + 1] if index + 1 < len(row) else ""
                kind, value = classify_value(raw)
                if kind == "missing":
                    raw_missing_counts[index] += 1
                elif kind == "numeric":
                    raw_numeric_counts[index] += 1
                    if value == 0:
                        raw_zero_counts[index] += 1

                zero_candidate = kind == "numeric" and value == 0 and current_ts is not None
                if zero_candidate and current_lengths[index] > 0 and continuous_timestamp:
                    current_lengths[index] += 1
                elif zero_candidate:
                    finish_zero_run(
                        index,
                        row_index,
                        current_lengths,
                        current_starts,
                        best_lengths,
                        best_starts,
                        best_ends,
                    )
                    current_lengths[index] = 1
                    current_starts[index] = row_index
                else:
                    finish_zero_run(
                        index,
                        row_index,
                        current_lengths,
                        current_starts,
                        best_lengths,
                        best_starts,
                        best_ends,
                    )
    finally:
        handle.close()

    for index in range(n):
        finish_zero_run(
            index,
            len(timestamps),
            current_lengths,
            current_starts,
            best_lengths,
            best_starts,
            best_ends,
        )

    mismatches = []
    for index, building_id in enumerate(building_ids):
        row = quality.get(building_id)
        if row is None:
            mismatches.append(f"quality CSV에 없는 raw building_id: {building_id}")
            continue
        expected_counts = {
            "total_records": int(row["total_records"]),
            "numeric_records": int(row["numeric_records"]),
            "missing_count": int(row["missing_count"]),
            "zero_count": int(row["zero_count"]),
        }
        actual_counts = {
            "total_records": len(timestamps),
            "numeric_records": raw_numeric_counts[index],
            "missing_count": raw_missing_counts[index],
            "zero_count": raw_zero_counts[index],
        }
        for key in expected_counts:
            if expected_counts[key] != actual_counts[key]:
                mismatches.append(f"{building_id}: {key} quality={expected_counts[key]} raw={actual_counts[key]}")

    return {
        "best_lengths": best_lengths,
        "best_starts": best_starts,
        "best_ends": best_ends,
        "raw_zero_counts": raw_zero_counts,
        "raw_numeric_counts": raw_numeric_counts,
        "raw_missing_counts": raw_missing_counts,
        "row_length_mismatch": row_length_mismatch,
        "header_mismatch": header_mismatch,
        "mismatches": mismatches,
    }


def rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return ""
    return f"{numerator / denominator:.8f}"


def build_rows(quality: dict[str, dict], building_ids: list[str], run_info: dict, timestamp_info: dict) -> list[dict]:
    expected_hours = (timestamp_info["expected_seconds"] or 0) / 3600
    timestamps = timestamp_info["timestamps"]
    output: list[dict] = []
    for index, building_id in enumerate(building_ids):
        source = quality[building_id]
        total = int(source["total_records"])
        numeric = int(source["numeric_records"])
        missing = int(source["missing_count"])
        zero = int(source["zero_count"])
        longest = run_info["best_lengths"][index]
        start_index = run_info["best_starts"][index]
        end_index = run_info["best_ends"][index]
        start = timestamps[start_index] if start_index is not None else None
        end = timestamps[end_index] if end_index is not None else None
        if numeric == 0:
            status = "확인 필요: 숫자 기록 없음"
        elif zero == numeric:
            status = "확인 필요: 숫자 기록이 모두 0"
        elif zero > 0:
            status = "확인 필요: 0값의 정상/계량 오류 판별 근거 없음"
        else:
            status = "0값 없음"
        output.append(
            {
                "building_id": building_id,
                "total_records": total,
                "numeric_records": numeric,
                "nonnumeric_count": int(source["nonnumeric_count"]),
                "missing_count": missing,
                "missing_rate_all_records": rate(missing, total),
                "zero_count": zero,
                "zero_rate_all_records": rate(zero, total),
                "zero_rate_numeric_records": rate(zero, numeric),
                "negative_count": int(source["negative_count"]),
                "longest_complete_run_records": int(source["longest_complete_run_records"]),
                "forecast_windows_24h_past_24h_future": int(source["forecast_windows_24h_past_24h_future"]),
                "longest_zero_run_records": longest,
                "longest_zero_run_coverage_hours": f"{longest * expected_hours:g}" if expected_hours else "",
                "longest_zero_run_start": start.strftime(TIMESTAMP_FORMAT) if start else "",
                "longest_zero_run_end": end.strftime(TIMESTAMP_FORMAT) if end else "",
                "review_status": status,
            }
        )
    return output


def write_outputs(rows: list[dict], output_csv: str, output_md: str, timestamp_info: dict, run_info: dict, quality: dict) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["building_id"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    all_zero = [row for row in rows if int(row["numeric_records"]) > 0 and int(row["zero_count"]) == int(row["numeric_records"])]
    no_numeric = [row for row in rows if int(row["numeric_records"]) == 0]
    any_zero = [row for row in rows if int(row["zero_count"]) > 0]
    missing_rates = [float(row["missing_rate_all_records"]) for row in rows]
    zero_all_rates = [float(row["zero_rate_all_records"]) for row in rows]
    zero_numeric_rates = [float(row["zero_rate_numeric_records"]) for row in rows if row["zero_rate_numeric_records"]]
    zero_lengths = [int(row["longest_zero_run_records"]) for row in rows]
    long_zero = sorted(rows, key=lambda row: int(row["longest_zero_run_records"]), reverse=True)
    expected_hours = (timestamp_info["expected_seconds"] or 0) / 3600
    lines = [
        "# BDG2 0값·결측 분포 검사",
        "",
        "- 입력 원본: `data/raw/bdg2/electricity.csv`",
        "- 결합한 기존 결과: `reports/bdg2_electricity_quality.csv`",
        "- 검사 코드: `scripts/audit_bdg2_zero_values.py`",
        "- 원본 데이터는 수정하지 않았으며, 0값을 결측으로 변환하거나 건물을 자동 제외하지 않았다.",
        "",
        "## 1. 계산 기준",
        "",
        "- 결측률 = `missing_count / total_records`.",
        "- 전체 기록 기준 0값 비율 = `zero_count / total_records`.",
        "- 숫자 기록 기준 0값 비율 = `zero_count / numeric_records`; 숫자 기록이 0개이면 `n.a.`(빈 칸)로 표시했다.",
        "- 최장 연속 0구간은 유한한 숫자값이 정확히 0이고 timestamp가 관측 최빈 간격으로 이어지는 행의 연속으로 계산했다. 결측, 비수치값, 0이 아닌 숫자, timestamp 오류·공백은 구간을 끊는다.",
        "- 시간대 변환과 서머타임 보정은 수행하지 않았다.",
        "",
        "## 2. 전체 요약",
        "",
        f"- 검사 건물/전력 열: **{total:,}개**",
        f"- 0값이 한 번 이상 있는 건물: **{len(any_zero):,}개**",
        f"- 숫자 기록이 모두 0인 건물: **{len(all_zero):,}개**",
        f"- 숫자 기록이 전혀 없는 건물: **{len(no_numeric):,}개**",
        f"- 전체 0값 개수: **{sum(int(row['zero_count']) for row in rows):,}개**",
        f"- 전체 결측값 개수: **{sum(int(row['missing_count']) for row in rows):,}개**",
        f"- 최장 연속 0구간 길이(행 수) 최소/중앙값/최대: **{min(zero_lengths):,} / {median(zero_lengths):g} / {max(zero_lengths):,}**",
        f"- 최장 연속 0구간 최대 길이의 시간당 기록 범위: **{max(zero_lengths) * expected_hours:g}시간**",
        "",
        "건물별 비율의 최소/중앙값/최대:",
        "",
        "| 지표 | 최소 | 중앙값 | 최대 |",
        "|---|---:|---:|---:|",
        f"| 결측률(전체 기록 기준) | {min(missing_rates):.6f} | {median(missing_rates):.6f} | {max(missing_rates):.6f} |",
        f"| 0값 비율(전체 기록 기준) | {min(zero_all_rates):.6f} | {median(zero_all_rates):.6f} | {max(zero_all_rates):.6f} |",
        f"| 0값 비율(숫자 기록 기준, numeric>0) | {min(zero_numeric_rates):.6f} | {median(zero_numeric_rates):.6f} | {max(zero_numeric_rates):.6f} |",
        "",
        "## 3. 숫자 기록이 모두 0인 건물 4개",
        "",
        "| building_id | 숫자 기록 수 | 0값 수 | 최장 0구간 | 시작 | 종료 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in all_zero:
        lines.append(
            f"| `{row['building_id']}` | {int(row['numeric_records']):,} | {int(row['zero_count']):,} | {int(row['longest_zero_run_records']):,} | {row['longest_zero_run_start'] or 'n.a.'} | {row['longest_zero_run_end'] or 'n.a.'} |"
        )
    lines.extend(
        [
            "",
            "## 4. 숫자 기록이 전혀 없는 건물 2개",
            "",
            "| building_id | 전체 기록 | 결측 | 0값 | 상태 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in no_numeric:
        lines.append(
            f"| `{row['building_id']}` | {int(row['total_records']):,} | {int(row['missing_count']):,} | {int(row['zero_count']):,} | {row['review_status']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 최장 연속 0구간 상위 사례",
            "",
            "| building_id | 최장 0구간(행) | 시작 | 종료 | 숫자 기록 기준 0값 비율 | 판정 |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for row in long_zero[:10]:
        lines.append(
            f"| `{row['building_id']}` | {int(row['longest_zero_run_records']):,} | {row['longest_zero_run_start'] or 'n.a.'} | {row['longest_zero_run_end'] or 'n.a.'} | {row['zero_rate_numeric_records'] or 'n.a.'} | 확인 필요 |"
        )
    lines.extend(
        [
            "",
            "## 6. 확인 필요 사항",
            "",
            "- 숫자 기록이 모두 0인 4개 건물과 숫자 기록이 전혀 없는 2개 건물은 별도 검토 대상이다. 이 결과만으로 정상적인 무부하 상태인지, 계량 오류인지, 수집 실패인지 결정하지 않았다.",
            "- 0값이 있는 나머지 건물도 정상 0과 계량 오류를 구분할 공식 근거가 없으므로 결과 CSV의 `review_status`를 `확인 필요`로 표시했다.",
            "- 결과 CSV에는 기존 품질 검사 수치와 결측률, 두 가지 0값 비율, 최장 연속 기록 길이, 최장 연속 0구간을 함께 기록했다.",
        ]
    )
    if run_info["mismatches"]:
        lines.extend(["", "원본 재검산과 기존 품질 CSV의 수치가 다른 항목:", ""])
        lines.extend(f"- {item}" for item in run_info["mismatches"][:50])
    else:
        lines.extend(["", "원본 재검산과 기존 품질 CSV의 전체 기록·숫자·결측·0값 수가 일치했다."])
    if run_info["header_mismatch"]:
        lines.extend(["", "헤더 확인:", ""])
        lines.extend(f"- {item}" for item in run_info["header_mismatch"])
    with open(output_md, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "data", "raw", "bdg2", "electricity.csv"))
    parser.add_argument("--quality", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_electricity_quality.csv"))
    parser.add_argument("--output-csv", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_zero_value_audit.csv"))
    parser.add_argument("--output-md", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_zero_value_audit.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_csv()
    try:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(args.input)
        if not os.path.isfile(args.quality):
            raise FileNotFoundError(args.quality)
        quality, quality_order = load_quality(args.quality)
        timestamp_info = read_timestamps(args.input)
        raw_building_ids = timestamp_info["header"][1:]
        if raw_building_ids != quality_order:
            raise ValueError("raw electricity 헤더와 기존 품질 결과의 building_id 순서가 일치하지 않습니다.")
        run_info = find_zero_runs(args.input, timestamp_info, quality, raw_building_ids)
        rows = build_rows(quality, raw_building_ids, run_info, timestamp_info)
        write_outputs(rows, args.output_csv, args.output_md, timestamp_info, run_info, quality)
        print(f"완료: {len(rows):,}개 건물, {timestamp_info['data_rows']:,}개 데이터 행")
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
