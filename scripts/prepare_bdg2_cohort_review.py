#!/usr/bin/env python3
"""Prepare a provisional site-level BDG2 cohort proposal.

The script applies the user's preliminary eligibility conditions to the
existing candidate report, marks long consecutive-zero runs for review, and
creates a deterministic site-level proposal. It does not select final sites,
split time windows, alter raw data, impute values, download weather data, or
train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUPS = [
    "source_train_proposal",
    "development_validation_proposal",
    "final_test_proposal",
]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype("string").str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def add_review_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["no_numeric_values"] = as_bool(out["no_numeric_values"])
    out["all_numeric_values_zero"] = as_bool(out["all_numeric_values_zero"])
    if "longest_zero_run_coverage_hours" in out:
        zero_hours = out["longest_zero_run_coverage_hours"].fillna(out["longest_zero_run_records"])
    else:
        zero_hours = out["longest_zero_run_records"]
    out["longest_zero_run_days"] = zero_hours / 24.0
    zero_days = out["longest_zero_run_days"]
    out["zero_run_review_flag"] = "연속 0값 구간 없음"
    out.loc[zero_days.gt(0) & zero_days.lt(1), "zero_run_review_flag"] = "연속 0값 검토(24시간 미만)"
    out.loc[zero_days.ge(1) & zero_days.lt(7), "zero_run_review_flag"] = "연속 0값 검토(24시간 이상)"
    out.loc[zero_days.ge(7) & zero_days.lt(30), "zero_run_review_flag"] = "장기 0값 검토(7일 이상)"
    out.loc[zero_days.ge(30), "zero_run_review_flag"] = "장기 0값 검토(30일 이상)"
    out["preliminary_eligible"] = (
        out["longest_complete_run_days"].ge(90)
        & ~out["no_numeric_values"]
        & ~out["all_numeric_values_zero"]
    )
    return out


def propose_site_groups(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_counts = (
        candidates.groupby("site_id", dropna=False)
        .size()
        .rename("candidate_building_count")
        .reset_index()
        .sort_values(["candidate_building_count", "site_id"], ascending=[False, True], kind="stable")
    )
    running = {group: 0 for group in GROUPS}
    assignments: dict[str, str] = {}
    site_rows = []
    for row in site_counts.itertuples(index=False):
        group = min(GROUPS, key=lambda name: (running[name], GROUPS.index(name)))
        assignments[row.site_id] = group
        running[group] += int(row.candidate_building_count)
        site_rows.append(
            {
                "site_id": row.site_id,
                "proposed_group": group,
                "candidate_building_count": int(row.candidate_building_count),
            }
        )
    out = candidates.copy()
    out["proposed_group"] = out["site_id"].map(assignments)
    out["proposal_status"] = "사이트 단위 임시 제안; 연구자 검토 전"
    out["site_candidate_building_count"] = out["site_id"].map(dict(zip(site_counts.site_id, site_counts.candidate_building_count))).astype(int)
    return out, pd.DataFrame(site_rows)


def continuity_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        frame = candidates.loc[candidates["proposed_group"].eq(group)]
        if frame.empty:
            continue
        rows.append(
            {
                "proposed_group": group,
                "building_count": int(len(frame)),
                "site_count": int(frame["site_id"].nunique()),
                "min_continuous_days": float(frame["longest_complete_run_days"].min()),
                "median_continuous_days": float(frame["longest_complete_run_days"].median()),
                "mean_continuous_days": float(frame["longest_complete_run_days"].mean()),
                "max_continuous_days": float(frame["longest_complete_run_days"].max()),
                "at_least_180d": int(frame["longest_complete_run_days"].ge(180).sum()),
                "at_least_365d": int(frame["longest_complete_run_days"].ge(365).sum()),
                "forecast_windows_total": int(frame["forecast_windows_24h_past_24h_future"].sum()),
                "zero_run_30d_review_count": int(frame["longest_zero_run_days"].ge(30).sum()),
            }
        )
    return pd.DataFrame(rows)


def usage_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    usage = candidates["primaryspaceusage"].fillna("(용도 미상)")
    table = pd.crosstab(usage, candidates["proposed_group"])
    for group in GROUPS:
        if group not in table.columns:
            table[group] = 0
    table = table[GROUPS]
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def write_report(
    original: pd.DataFrame,
    reviewed: pd.DataFrame,
    candidates: pd.DataFrame,
    site_assignments: pd.DataFrame,
    output_csv: Path,
    output_md: Path,
) -> None:
    excluded = reviewed.loc[~reviewed["preliminary_eligible"]]
    continuity = continuity_summary(candidates)
    usage = usage_summary(candidates)
    zero_flags = candidates["zero_run_review_flag"].value_counts().reindex(
        [
            "장기 0값 검토(30일 이상)",
            "장기 0값 검토(7일 이상)",
            "연속 0값 검토(24시간 이상)",
            "연속 0값 검토(24시간 미만)",
            "연속 0값 구간 없음",
        ],
        fill_value=0,
    )
    special_none = original.loc[as_bool(original["no_numeric_values"]), "building_id"].tolist()
    special_zero = original.loc[as_bool(original["all_numeric_values_zero"]), "building_id"].tolist()

    lines = [
        "# BDG2 사이트 단위 실험 코호트 검토",
        "",
        "- **검토일:** 2026-09-23",
        "- **입력:** `reports/bdg2_experiment_candidates.csv`",
        "- **목적:** 예측 실험 후보 목록과 사이트 단위 전이학습 그룹 제안 작성",
        "- **제외한 작업:** 시간 구간의 최종 train/validation/test 분할, 결측 보간, 기상 데이터 다운로드, AI 모델 학습",
        "",
        "## 1. 예비 후보 조건 적용",
        "",
        "다음 조건을 모두 만족하는 건물을 예비 후보로 표시했다.",
        "",
        "1. 최장 연속 유효 기록 `longest_complete_run_days >= 90`",
        "2. `no_numeric_values == False`",
        "3. `all_numeric_values_zero == False`",
        "",
        f"- 입력 건물: **{len(original):,}개**",
        f"- 예비 후보: **{len(candidates):,}개**",
        f"- 예비 조건에서 제외: **{len(excluded):,}개**",
        f"  - 최장 연속 기록 90일 미만: **{int(original['longest_complete_run_days'].lt(90).sum()):,}개**",
        f"  - 90일 이상이지만 숫자 기록 없음 또는 숫자 기록 모두 0: **{int((original['longest_complete_run_days'].ge(90) & (as_bool(original['no_numeric_values']) | as_bool(original['all_numeric_values_zero']))).sum()):,}개**",
        "",
        "이 조건은 실험 준비용 예비 기준이다. 후보 안의 0값은 삭제하거나 결측으로 변경하지 않았다.",
        "",
        "## 2. 예비 후보의 사이트별 분포",
        "",
        "| site_id | 후보 건물 수 |",
        "|---|---:|",
    ]
    site_counts = candidates["site_id"].value_counts().sort_index()
    lines.extend([f"| {site_id} | {int(count)} |" for site_id, count in site_counts.items()])

    lines.extend(
        [
            "",
            "## 3. 예비 후보의 건물 용도별 분포",
            "",
            "| 건물 용도 | 후보 건물 수 |",
            "|---|---:|",
        ]
    )
    usage_total = candidates["primaryspaceusage"].fillna("(용도 미상)").value_counts()
    lines.extend([f"| {usage} | {int(count)} |" for usage, count in usage_total.items()])

    lines.extend(
        [
            "",
            "## 4. 연속 0값 별도 검토 표시",
            "",
            "`longest_zero_run_days`를 숫자로 보존하고, 아래 구간은 검토용 표시만 추가했다. 구간 기준은 현황을 읽기 위한 기술적 구간이며 자동 제외 기준이 아니다.",
            "",
            "| 표시 | 후보 건물 수 |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {label} | {int(count)} |" for label, count in zero_flags.items()])
    lines.extend(
        [
            "",
            f"특히 30일 이상 연속 0값으로 표시된 예비 후보는 **{int(zero_flags['장기 0값 검토(30일 이상)'])}개**이다. 이 건물들은 계량 상태나 실제 무부하 여부를 별도 확인해야 한다.",
            "",
            f"기존 보고서의 숫자 기록 없음 건물({len(special_none)}개): {', '.join(special_none)}",
            f"기존 보고서의 숫자 기록 모두 0인 건물({len(special_zero)}개): {', '.join(special_zero)}",
            "",
            "## 5. 사이트 단위 그룹 제안",
            "",
            "다음 배정은 최종 시험 사이트 확정이 아닌 검토용 제안이다. 후보 건물 수가 많은 사이트부터 내림차순으로 정렬한 뒤, 현재 후보 건물 수가 가장 작은 그룹에 순서대로 배정하는 단일 결정적 규칙을 사용했다. 동률은 `source_train_proposal`, `development_validation_proposal`, `final_test_proposal` 순서로 처리했다.",
            "",
            "사이트는 한 그룹에만 배정되므로 출발 학습 그룹과 최종 시험 제안 그룹 사이에 사이트 중복이 없다. 실제 시간 구간을 나누거나 모델을 학습하지 않았다.",
            "",
            "| 그룹 코드 | 용도 | 사이트 수 | 후보 건물 수 | 사이트 목록 |",
            "|---|---|---:|---:|---|",
        ]
    )
    group_labels = {
        "source_train_proposal": "출발 건물 학습용 제안",
        "development_validation_proposal": "모델 개발·검증용 제안",
        "final_test_proposal": "최종 시험용 제안",
    }
    for group in GROUPS:
        sites = site_assignments.loc[site_assignments["proposed_group"] == group, "site_id"].tolist()
        n = int((candidates["proposed_group"] == group).sum())
        lines.append(f"| `{group}` | {group_labels[group]} | {len(sites)} | {n} | {', '.join(sites)} |")

    lines.extend(
        [
            "",
            "### 그룹별 연속 데이터 확보 현황",
            "",
            "| 그룹 | 건물 수 | 사이트 수 | 최단 연속일 | 중앙값 연속일 | 평균 연속일 | 최장 연속일 | 180일 이상 | 365일 이상 | 24시간 예측 구간 합계 | 30일 이상 연속 0값 검토 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in continuity.itertuples(index=False):
        lines.append(
            f"| `{row.proposed_group}` | {row.building_count} | {row.site_count} | {row.min_continuous_days:.1f} | {row.median_continuous_days:.1f} | {row.mean_continuous_days:.1f} | {row.max_continuous_days:.1f} | {row.at_least_180d} | {row.at_least_365d} | {row.forecast_windows_total} | {row.zero_run_30d_review_count} |"
        )

    lines.extend(
        [
            "",
            "### 그룹별 건물 용도 분포",
            "",
            "| 건물 용도 | 출발 학습 제안 | 개발·검증 제안 | 최종 시험 제안 | 합계 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for usage, row in usage.iterrows():
        lines.append(f"| {usage} | {int(row[GROUPS[0]])} | {int(row[GROUPS[1]])} | {int(row[GROUPS[2]])} | {int(row['total'])} |")

    lines.extend(
        [
            "",
            "## 6. 연구자 검토가 필요한 사항",
            "",
            "- 최종 시험 사이트는 아직 확정하지 않았다. 위 배정은 후보 수 균형을 위한 첫 번째 제안이다.",
            "- 최종 시험 제안 그룹은 30일 이상 연속 0값 검토 건물이 상대적으로 많으므로, 계량 오류 여부를 확인한 뒤 그룹안을 재검토해야 한다.",
            "- 90일 조건은 건물 후보 필터로만 적용했다. 실제 학습·검증·시험 시간 구간의 경계와 누수 방지 방식은 별도 설계가 필요하다.",
            "- 건물 용도가 특정 그룹에만 존재하는 희소 범주는 사이트 단위 분할의 제약으로 그룹 간 직접 비교가 어렵다.",
            "",
            "## 7. 산출물과 재현 코드",
            "",
            f"예비 후보와 제안 그룹은 [`{output_csv.name}`](../reports/{output_csv.name})에 저장했다. 각 행에는 `proposed_group`, `proposal_status`, `longest_zero_run_days`, `zero_run_review_flag`가 포함된다.",
            "",
            "재현 명령:",
            "",
            "```powershell",
            'python `',
            '  "research_stage1\\scripts\\prepare_bdg2_cohort_review.py" `',
            '  --input-csv "research_stage1\\reports\\bdg2_experiment_candidates.csv" `',
            '  --output-csv "research_stage1\\reports\\bdg2_cohort_candidates.csv" `',
            '  --output-md "research_stage1\\reports\\bdg2_cohort_review.md"',
            "```",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("research_stage1/reports/bdg2_experiment_candidates.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("research_stage1/reports/bdg2_cohort_candidates.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("research_stage1/reports/bdg2_cohort_review.md"))
    args = parser.parse_args()

    original = pd.read_csv(args.input_csv)
    reviewed = add_review_flags(original)
    candidates = reviewed.loc[reviewed["preliminary_eligible"]].copy()
    candidates, site_assignments = propose_site_groups(candidates)
    candidates = candidates.sort_values(["proposed_group", "site_id", "building_id"], kind="stable").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_report(original, reviewed, candidates, site_assignments, args.output_csv, args.output_md)

    print(f"input_rows={len(original)}")
    print(f"eligible_rows={len(candidates)}")
    print(f"eligible_sites={candidates['site_id'].nunique()}")
    print(candidates["proposed_group"].value_counts().to_string())
    print("long_zero_30d_review=" + str(int(candidates["longest_zero_run_days"].ge(30).sum())))


if __name__ == "__main__":
    main()
