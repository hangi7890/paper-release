#!/usr/bin/env python3
"""Summarize BDG2 building conditions for experiment-candidate review.

This is a review-only analysis. It reads the three existing BDG2 reports,
checks their shared building IDs and metric consistency, and writes a merged
building table plus a Markdown summary. It does not alter raw data, impute
missing values, select buildings, download weather data, or train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


QUALITY_SHARED = [
    "total_records",
    "numeric_records",
    "nonnumeric_count",
    "missing_count",
    "longest_complete_run_records",
    "forecast_windows_24h_past_24h_future",
]


def read_inputs(reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(reports_dir / "bdg2_metadata_quality_join.csv")
    quality = pd.read_csv(reports_dir / "bdg2_electricity_quality.csv")
    zero = pd.read_csv(reports_dir / "bdg2_zero_value_audit.csv")
    for name, frame in [("metadata", metadata), ("quality", quality), ("zero", zero)]:
        if "building_id" not in frame.columns:
            raise ValueError(f"{name} report has no building_id column")
        if frame["building_id"].duplicated().any():
            raise ValueError(f"{name} report contains duplicate building_id values")
    return metadata, quality, zero


def compare_shared_fields(metadata: pd.DataFrame, quality: pd.DataFrame, zero: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    indexed = {name: frame.set_index("building_id") for name, frame in [("metadata", metadata), ("quality", quality), ("zero", zero)]}
    for field in QUALITY_SHARED:
        if field not in indexed["metadata"].columns:
            continue
        reference = indexed["metadata"][field]
        for other_name in ["quality", "zero"]:
            if field not in indexed[other_name].columns:
                continue
            other = indexed[other_name][field].reindex(reference.index)
            mismatch = ~(reference.eq(other) | (reference.isna() & other.isna()))
            for building_id in reference.index[mismatch]:
                issues.append(f"{field}: {building_id} ({reference.loc[building_id]} != {other.loc[building_id]} in {other_name})")
    return issues


def build_table(metadata: pd.DataFrame, quality: pd.DataFrame, zero: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    consistency_issues = compare_shared_fields(metadata, quality, zero)

    metadata_fields = [
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sqm",
        "timezone",
        "metadata_electricity_flag",
        "meter_scope_status",
        "join_status",
    ]
    quality_fields = [
        "building_id",
        "total_records",
        "numeric_records",
        "nonnumeric_count",
        "missing_count",
        "longest_complete_run_records",
        "longest_complete_run_coverage_hours",
        "longest_complete_run_elapsed_hours",
        "longest_complete_run_start",
        "longest_complete_run_end",
        "forecast_windows_24h_past_24h_future",
    ]
    zero_fields = [
        "building_id",
        "missing_rate_all_records",
        "zero_count",
        "zero_rate_all_records",
        "zero_rate_numeric_records",
        "negative_count",
        "longest_zero_run_records",
        "longest_zero_run_coverage_hours",
        "longest_zero_run_start",
        "longest_zero_run_end",
        "review_status",
    ]

    merged = metadata[metadata_fields].merge(quality[quality_fields], on="building_id", how="outer", validate="one_to_one")
    merged = merged.merge(zero[zero_fields], on="building_id", how="outer", validate="one_to_one")
    if merged["building_id"].isna().any():
        raise ValueError("Merged table contains missing building_id")

    merged["longest_complete_run_days"] = merged["longest_complete_run_coverage_hours"] / 24.0
    merged["no_numeric_values"] = merged["numeric_records"].eq(0)
    merged["all_numeric_values_zero"] = merged["no_numeric_values"].eq(False) & merged["zero_rate_numeric_records"].eq(1.0)
    merged["special_review_target"] = ""
    merged.loc[merged["no_numeric_values"], "special_review_target"] = "숫자 기록 없음"
    merged.loc[merged["all_numeric_values_zero"], "special_review_target"] = "숫자 기록 모두 0"
    merged["source_consistency_status"] = "일치"
    if consistency_issues:
        # Keep the row-level table reviewable while making the mismatch visible.
        merged["source_consistency_status"] = "공통 필드 불일치 확인 필요"

    merged = merged.sort_values(["site_id", "building_id"], kind="stable").reset_index(drop=True)
    return merged, consistency_issues


def threshold_counts(table: pd.DataFrame) -> pd.DataFrame:
    thresholds = [30, 90, 180, 365]
    rows = []
    for days in thresholds:
        rows.append({"minimum_continuous_days": days, "building_count": int(table["longest_complete_run_days"].ge(days).sum())})
    return pd.DataFrame(rows)


def site_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_id, group in table.groupby("site_id", dropna=False, sort=True):
        rows.append(
            {
                "site_id": site_id,
                "building_count": int(len(group)),
                "at_least_30d": int(group["longest_complete_run_days"].ge(30).sum()),
                "at_least_90d": int(group["longest_complete_run_days"].ge(90).sum()),
                "at_least_180d": int(group["longest_complete_run_days"].ge(180).sum()),
                "at_least_365d": int(group["longest_complete_run_days"].ge(365).sum()),
                "no_numeric_count": int(group["no_numeric_values"].sum()),
                "all_numeric_zero_count": int(group["all_numeric_values_zero"].sum()),
            }
        )
    return pd.DataFrame(rows)


def markdown_report(table: pd.DataFrame, consistency_issues: list[str], output_csv: Path) -> str:
    threshold = threshold_counts(table)
    sites = site_summary(table)
    special_none = table.loc[table["no_numeric_values"], "building_id"].tolist()
    special_zero = table.loc[table["all_numeric_values_zero"], "building_id"].tolist()

    lines = [
        "# BDG2 24시간 전력 예측 실험 후보 데이터 조건 검토",
        "",
        "- **검토일:** 2026-09-23",
        "- **목적:** 기존 BDG2 품질 결과를 이용해 건물별 실험 후보 조건을 정리",
        "- **범위:** 결측률·0값 비율·연속 기록·24시간 예측 구간·사이트·건물 용도 집계",
        "- **제외한 작업:** 건물 최종 선정·제외, 결측 보간, 기상 데이터 다운로드, 원본 수정, 모델 학습",
        "",
        "## 1. 입력과 결합 상태",
        "",
        "세 입력 보고서를 `building_id`로 결합했다. 각 입력에는 중복 ID가 없었고, 결합 결과는 1,578개 건물이다.",
        "",
        "- `bdg2_metadata_quality_join.csv`: 사이트·건물 용도·면적·시간대 및 기존 결합 상태",
        "- `bdg2_electricity_quality.csv`: 기록 수·결측/숫자 상태·최장 연속 유효 기록·24시간 예측 구간",
        "- `bdg2_zero_value_audit.csv`: 결측률·0값 비율·최장 연속 0값 구간",
        "",
        "연속 기간은 기존 품질 보고서의 `longest_complete_run_coverage_hours`를 24로 나누어 일수로 표시했다. 이는 기존 시간별 품질 검사의 연속 시간 기준을 그대로 사용한 것이며, 새 보간이나 시간대 변환을 수행하지 않았다.",
        "",
        f"공통 필드 불일치 건수: **{len(consistency_issues)}**",
    ]
    if consistency_issues:
        lines.extend(["", "불일치 예시:", *[f"- {item}" for item in consistency_issues[:20]]])

    lines.extend(
        [
            "",
            "## 2. 연속 기록 기간별 건물 수",
            "",
            "| 최장 연속 유효 기록 기준 | 건물 수 |",
            "|---:|---:|",
        ]
    )
    lines.extend([f"| {int(row.minimum_continuous_days)}일 이상 | {int(row.building_count)} |" for row in threshold.itertuples()])

    lines.extend(
        [
            "",
            "## 3. 사이트별 장기 기록 현황",
            "",
            "| site_id | 건물 수 | 30일 이상 | 90일 이상 | 180일 이상 | 365일 이상 | 숫자 기록 없음 | 숫자 기록 모두 0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sites.itertuples():
        lines.append(
            f"| {row.site_id} | {row.building_count} | {row.at_least_30d} | {row.at_least_90d} | {row.at_least_180d} | {row.at_least_365d} | {row.no_numeric_count} | {row.all_numeric_zero_count} |"
        )

    lines.extend(
        [
            "",
            "## 4. 별도 검토 대상",
            "",
            f"- **숫자 기록이 전혀 없는 건물 ({len(special_none)}개):** {', '.join(special_none) if special_none else '없음'}",
            f"- **숫자 기록이 모두 0인 건물 ({len(special_zero)}개):** {', '.join(special_zero) if special_zero else '없음'}",
            "",
            "이 표시는 검토 우선순위를 위한 것이며 자동 제외나 최종 학습·시험 선정이 아니다. 0값을 결측으로 바꾸지 않았고, 정상적인 0값과 계량 오류를 판정하지 않았다.",
            "",
            "## 5. 건물별 검토표",
            "",
            f"전체 건물별 결과는 [`{output_csv.name}`](../{output_csv.parent.name}/{output_csv.name})에 저장했다. 주요 열은 `site_id`, `primaryspaceusage`, `missing_rate_all_records`, `zero_rate_all_records`, `zero_rate_numeric_records`, `longest_complete_run_days`, `forecast_windows_24h_past_24h_future`, `special_review_target`이다.",
            "",
            "## 6. 재현 코드",
            "",
            "```powershell",
            'python `',
            '  "research_stage1\\scripts\\review_bdg2_experiment_candidates.py" `',
            '  --reports-dir "research_stage1\\reports" `',
            '  --output-csv "research_stage1\\reports\\bdg2_experiment_candidates.csv" `',
            '  --output-md "research_stage1\\reports\\bdg2_experiment_candidate_review.md"',
            "```",
            "",
            "이 코드는 기존 보고서만 읽으며 원본 전력 CSV를 수정하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("research_stage1/reports"))
    parser.add_argument("--output-csv", type=Path, default=Path("research_stage1/reports/bdg2_experiment_candidates.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("research_stage1/reports/bdg2_experiment_candidate_review.md"))
    args = parser.parse_args()

    metadata, quality, zero = read_inputs(args.reports_dir)
    table, issues = build_table(metadata, quality, zero)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    args.output_md.write_text(markdown_report(table, issues, args.output_csv), encoding="utf-8")
    print(f"rows={len(table)}")
    print(threshold_counts(table).to_string(index=False))
    print(f"no_numeric={int(table['no_numeric_values'].sum())}")
    print(f"all_numeric_zero={int(table['all_numeric_values_zero'].sum())}")
    print(f"consistency_issues={len(issues)}")


if __name__ == "__main__":
    main()
