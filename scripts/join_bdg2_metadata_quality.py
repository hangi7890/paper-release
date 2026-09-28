#!/usr/bin/env python3
"""Join BDG2 metadata and existing electricity-quality audit outputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from statistics import median


def read_keyed_csv(path: str, key: str) -> tuple[dict[str, dict], list[str]]:
    rows: dict[str, dict] = {}
    order: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key not in reader.fieldnames:
            raise ValueError(f"{path}에 키 열 `{key}`가 없습니다.")
        for row in reader:
            value = row.get(key, "")
            if value in rows:
                raise ValueError(f"{path}에 중복 {key}: {value}")
            rows[value] = row
            order.append(value)
    return rows, order


def label(value: str) -> str:
    return value.strip() if value and value.strip() else "[missing]"


def int_value(row: dict, key: str) -> int:
    return int(row.get(key, "0") or 0)


def float_value(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def make_join_rows(metadata: dict[str, dict], quality: dict[str, dict], zero: dict[str, dict], order: list[str]) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    output: list[dict] = []
    for building_id in order:
        if building_id not in metadata:
            errors.append(f"metadata.csv에 없는 building_id: {building_id}")
            continue
        if building_id not in zero:
            errors.append(f"zero_value_audit.csv에 없는 building_id: {building_id}")
            continue
        m = metadata[building_id]
        q = quality[building_id]
        z = zero[building_id]
        for key in ("total_records", "numeric_records", "missing_count", "zero_count"):
            if q.get(key, "") != z.get(key, ""):
                errors.append(f"{building_id}: quality와 zero audit의 {key} 불일치")
        numeric = int_value(q, "numeric_records")
        zeros = int_value(q, "zero_count")
        output.append(
            {
                "building_id": building_id,
                "site_id": m.get("site_id", ""),
                "primaryspaceusage": m.get("primaryspaceusage", ""),
                "sqm": m.get("sqm", ""),
                "timezone": m.get("timezone", ""),
                "metadata_electricity_flag": m.get("electricity", ""),
                "total_records": q.get("total_records", ""),
                "numeric_records": q.get("numeric_records", ""),
                "nonnumeric_count": q.get("nonnumeric_count", ""),
                "missing_count": q.get("missing_count", ""),
                "missing_rate_all_records": z.get("missing_rate_all_records", ""),
                "zero_count": q.get("zero_count", ""),
                "zero_rate_all_records": z.get("zero_rate_all_records", ""),
                "zero_rate_numeric_records": z.get("zero_rate_numeric_records", ""),
                "negative_count": q.get("negative_count", ""),
                "longest_complete_run_records": q.get("longest_complete_run_records", ""),
                "forecast_windows_24h_past_24h_future": q.get("forecast_windows_24h_past_24h_future", ""),
                "longest_zero_run_records": z.get("longest_zero_run_records", ""),
                "longest_zero_run_start": z.get("longest_zero_run_start", ""),
                "longest_zero_run_end": z.get("longest_zero_run_end", ""),
                "all_numeric_values_zero": "True" if numeric > 0 and zeros == numeric else "False",
                "no_numeric_values": "True" if numeric == 0 else "False",
                "review_status": z.get("review_status", "확인 필요"),
                "meter_scope_status": "확인 필요",
                "join_status": "matched",
            }
        )
    return output, errors


def distribution_table(rows: list[dict], key: str) -> list[tuple[str, int, int, int]]:
    totals: Counter[str] = Counter()
    all_zero: Counter[str] = Counter()
    no_numeric: Counter[str] = Counter()
    for row in rows:
        group = label(row.get(key, ""))
        totals[group] += 1
        if row["all_numeric_values_zero"] == "True":
            all_zero[group] += 1
        if row["no_numeric_values"] == "True":
            no_numeric[group] += 1
    return [(group, totals[group], all_zero[group], no_numeric[group]) for group in sorted(totals, key=lambda item: (-totals[item], item))]


def write_outputs(rows: list[dict], errors: list[str], output_csv: str, output_md: str, metadata_count: int, quality_count: int, zero_count: int) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["building_id"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    all_zero = [row for row in rows if row["all_numeric_values_zero"] == "True"]
    no_numeric = [row for row in rows if row["no_numeric_values"] == "True"]
    missing_rates = [float_value(row, "missing_rate_all_records") for row in rows]
    zero_rates = [float_value(row, "zero_rate_all_records") for row in rows]
    sites = distribution_table(rows, "site_id")
    usages = distribution_table(rows, "primaryspaceusage")

    lines = [
        "# BDG2 메타데이터·전력 품질 결합 및 계량 범위 검토",
        "",
        "- 결합 키: `building_id`",
        "- 입력: `data/raw/bdg2/metadata.csv`, `reports/bdg2_electricity_quality.csv`, `reports/bdg2_zero_value_audit.csv`",
        "- 출력 표: `reports/bdg2_metadata_quality_join.csv`",
        "- 원본과 기존 결과 파일은 수정하지 않았으며, 건물 자동 제외·결측 처리·기상 데이터 다운로드·학습은 수행하지 않았다.",
        "",
        "## 1. 결합 확인",
        "",
        f"- metadata 행: {metadata_count:,}개",
        f"- 기존 전력 품질 행: {quality_count:,}개",
        f"- 기존 0값 감사 행: {zero_count:,}개",
        f"- 결합 성공 전력 건물: **{len(rows):,}개**",
        f"- 결합 오류: {len(errors):,}건",
        "- 전력 품질 결과의 1,578개 `building_id`가 모두 metadata와 1:1로 결합되도록 검사했다.",
        "",
        "## 2. 전력 건물의 사이트별 분포",
        "",
        "| site_id | 전력 건물 수 | 숫자 기록 모두 0 | 숫자 기록 없음 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {group} | {total:,} | {all_zero:,} | {no_numeric:,} |" for group, total, all_zero, no_numeric in sites)
    lines.extend(
        [
            "",
            "## 3. 전력 건물의 용도별 분포",
            "",
            "| primaryspaceusage | 전력 건물 수 | 숫자 기록 모두 0 | 숫자 기록 없음 |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(f"| {group} | {total:,} | {all_zero:,} | {no_numeric:,} |" for group, total, all_zero, no_numeric in usages)
    lines.extend(
        [
            "",
            "## 4. 결합 표의 요약 지표",
            "",
            f"- 결측률(전체 기록 기준) 건물별 최소/중앙값/최대: {min(missing_rates):.6f} / {median(missing_rates):.6f} / {max(missing_rates):.6f}",
            f"- 0값 비율(전체 기록 기준) 건물별 최소/중앙값/최대: {min(zero_rates):.6f} / {median(zero_rates):.6f} / {max(zero_rates):.6f}",
            f"- 숫자 기록 모두 0: {len(all_zero):,}개",
            f"- 숫자 기록 없음: {len(no_numeric):,}개",
            "",
            "결합 CSV에는 `site_id`, `primaryspaceusage`, `sqm`, `timezone`, 기존 품질 수치, 결측률·0값 비율, 최장 연속 유효 기록 길이, 최장 연속 0구간 및 검토 상태를 함께 기록했다.",
            "",
            "## 5. BDG2 공식 자료에서 확인되는 계량 범위 설명",
            "",
            "### 공식 자료에서 확인된 사실",
            "",
            "1. BDG2 README는 데이터셋을 1,636개 건물의 3,053개 에너지 계량기로 설명하고, 2016~2017년의 시간별 electricity, heating/cooling water, steam, irrigation 계량값을 제공한다고 명시한다.",
            "2. README의 인용 논문 초록은 건물별 하나 이상의 계량기가 whole building electrical, heating/cooling water, steam, solar energy, water, irrigation 등을 측정한다고 데이터셋 수준에서 설명한다.",
            "3. Meters data description Wiki는 raw meter 자료가 2016~2017년 시간당 한 건의 kWh 계량값이며, 계량기 종류별 데이터셋이 있다고 설명한다.",
            "4. Metadata 문서는 `electricity`를 전력 계량기 보유 여부를 나타내는 메타데이터 특징으로 설명하지만, 개별 열의 물리적 계량 경계나 submeter 여부를 나타내는 필드는 제시하지 않는다.",
            "",
            "### 개별 건물의 총소비전력 여부에 관한 미확인 사항",
            "",
            "- 데이터셋 전체 설명의 `whole building` 표현만으로 `electricity.csv`의 각 열이 건물 전체의 실제 총소비전력이라고 확정할 수 없다.",
            "- 이번 세 파일에는 계량기 식별자, 계량 지점, 물리적 경계, main meter/submeter 구분, 전력 열별 공식 단위·범위 매핑이 없다.",
            "- 따라서 결합 표의 모든 전력 건물에 `meter_scope_status=확인 필요`를 기록했다. 숫자 기록이 모두 0인 4개와 숫자 기록이 없는 2개도 자동 제외하지 않았다.",
            "- 개별 건물의 총소비전력 여부는 시설별 원자료 설명이나 별도 계량기 매핑 없이는 확인할 수 없다.",
            "",
            "## 6. 공식 출처 URL",
            "",
            "- [BDG2 공식 README](https://github.com/buds-lab/building-data-genome-project-2)",
            "- [Meters data description Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-description)",
            "- [Meters data features Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features)",
            "- [Metadata features Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Metadata-features)",
            "- [BDG2 데이터 논문 DOI](https://doi.org/10.1038/s41597-020-00712-x)",
        ]
    )
    if errors:
        lines.extend(["", "## 7. 결합 오류 상세", ""])
        lines.extend(f"- {error}" for error in errors[:100])
    with open(output_md, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "data", "raw", "bdg2", "metadata.csv"))
    parser.add_argument("--quality", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_electricity_quality.csv"))
    parser.add_argument("--zero-audit", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_zero_value_audit.csv"))
    parser.add_argument("--output-csv", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_metadata_quality_join.csv"))
    parser.add_argument("--output-md", default=os.path.join(os.environ.get("PAPER_WORKSPACE", "."), "reports", "bdg2_meter_scope_review.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata, _ = read_keyed_csv(args.metadata, "building_id")
        quality, quality_order = read_keyed_csv(args.quality, "building_id")
        zero, zero_order = read_keyed_csv(args.zero_audit, "building_id")
        if quality_order != zero_order:
            raise ValueError("quality CSV와 zero audit CSV의 building_id 순서가 일치하지 않습니다.")
        rows, errors = make_join_rows(metadata, quality, zero, quality_order)
        write_outputs(rows, errors, args.output_csv, args.output_md, len(metadata), len(quality), len(zero))
        print(f"완료: {len(rows):,}개 전력 건물 결합")
        print(f"결과 CSV: {args.output_csv}")
        print(f"계량 범위 보고서: {args.output_md}")
        if errors:
            print(f"결합 확인 필요 항목: {len(errors):,}건", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001 - command-line diagnostic
        print(f"결합 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
