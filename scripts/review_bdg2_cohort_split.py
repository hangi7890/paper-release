"""Review the existing BDG2 site split and create one data-only alternative.

The script reads the existing candidate table only.  It does not read or modify
the raw electricity file and it does not run a model.  The alternative split is
generated deterministically from site-level building counts, long-zero counts,
and the sum of longest complete-run days.  The predeclared objective minimizes
the squared deviation of those three quantities from one-third of the total.
Single-site moves and two-site swaps are considered in a fixed order until no
strictly improving change remains; ties are resolved by site name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[1]))).resolve()
INPUT = ROOT / "reports" / "bdg2_cohort_candidates.csv"
OUT_CSV = ROOT / "reports" / "bdg2_cohort_split_alternative.csv"
OUT_MD = ROOT / "reports" / "bdg2_cohort_split_review.md"

GROUPS = (
    "source_train_proposal",
    "development_validation_proposal",
    "final_test_proposal",
)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def zero_bin(days: float) -> str:
    if pd.isna(days) or days <= 0:
        return "0일 (연속 0값 없음)"
    if days < 1:
        return "1일 미만"
    if days < 7:
        return "1일 이상 7일 미만"
    if days < 30:
        return "7일 이상 30일 미만"
    return "30일 이상"


def make_site_stats(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_run_days"] = numeric_series(work, "longest_complete_run_days")
    work["_zero_days"] = numeric_series(work, "longest_zero_run_days")
    work["_z30"] = (work["_zero_days"] >= 30).astype(int)
    return (
        work.groupby("site_id", dropna=False)
        .agg(
            buildings=("building_id", "size"),
            z30_count=("_z30", "sum"),
            complete_days_sum=("_run_days", "sum"),
            complete_days_median=("_run_days", "median"),
            complete_days_mean=("_run_days", "mean"),
        )
        .reset_index()
    )


def site_loads(site_stats: pd.DataFrame, assignment: Dict[str, str]):
    lookup = site_stats.set_index("site_id").to_dict("index")
    loads = {g: [0.0, 0.0, 0.0] for g in GROUPS}
    for site, group in assignment.items():
        row = lookup[site]
        loads[group][0] += float(row["buildings"])
        loads[group][1] += float(row["z30_count"])
        loads[group][2] += float(row["complete_days_sum"])
    return loads


def objective(site_stats: pd.DataFrame, assignment: Dict[str, str]) -> float:
    loads = site_loads(site_stats, assignment)
    total_n = float(site_stats["buildings"].sum())
    total_z = float(site_stats["z30_count"].sum())
    total_days = float(site_stats["complete_days_sum"].sum())
    targets = (total_n / 3.0, total_z / 3.0, total_days / 3.0)
    score = 0.0
    for n, z, days in loads.values():
        score += ((n - targets[0]) / targets[0]) ** 2
        score += ((z - targets[1]) / targets[1]) ** 2
        # Continuous coverage is a supporting balance measure, so it has 0.25
        # of the weight of each count measure.
        score += 0.25 * ((days - targets[2]) / targets[2]) ** 2
    return score


def deterministic_alternative(site_stats: pd.DataFrame, current: Dict[str, str]):
    """Apply deterministic best-improving site moves and site swaps."""
    assignment = dict(current)
    sites = sorted(site_stats["site_id"].astype(str))
    while True:
        before = objective(site_stats, assignment)
        best_key = None
        best_assignment = None

        # Fixed order: all single-site moves, then all two-site swaps.
        for site in sites:
            old = assignment[site]
            for group in GROUPS:
                if group == old:
                    continue
                candidate = dict(assignment)
                candidate[site] = group
                if len(set(candidate.values())) < 3:
                    continue
                score = objective(site_stats, candidate)
                key = (score, "move", site, group)
                if score < before - 1e-12 and (best_key is None or key < best_key):
                    best_key = key
                    best_assignment = candidate

        for i, first in enumerate(sites):
            for second in sites[i + 1 :]:
                if assignment[first] == assignment[second]:
                    continue
                candidate = dict(assignment)
                candidate[first], candidate[second] = assignment[second], assignment[first]
                score = objective(site_stats, candidate)
                key = (score, "swap", first, second)
                if score < before - 1e-12 and (best_key is None or key < best_key):
                    best_key = key
                    best_assignment = candidate

        if best_assignment is None:
            break
        assignment = best_assignment
    return assignment


def fmt_number(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "확인 필요"
    return f"{value:,.{digits}f}"


def group_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = frame.copy()
    work["_run_days"] = numeric_series(work, "longest_complete_run_days")
    work["_zero_days"] = numeric_series(work, "longest_zero_run_days")
    work["_z30"] = work["_zero_days"] >= 30
    work["_z7"] = work["_zero_days"] >= 7
    rows = []
    present_groups = [g for g in GROUPS if g in set(work[group_col].astype(str))]
    for group in present_groups:
        part = work[work[group_col].astype(str) == group]
        usage = part["primaryspaceusage"].fillna("미상").astype(str).value_counts()
        usage_text = ", ".join(f"{k} {v}" for k, v in usage.items())
        rows.append(
            {
                "group": group,
                "buildings": len(part),
                "sites": part["site_id"].nunique(),
                "usage": usage_text,
                "run_min_days": part["_run_days"].min(),
                "run_median_days": part["_run_days"].median(),
                "run_mean_days": part["_run_days"].mean(),
                "run_max_days": part["_run_days"].max(),
                "run_ge_180": int((part["_run_days"] >= 180).sum()),
                "run_ge_365": int((part["_run_days"] >= 365).sum()),
                "forecast_windows": numeric_series(part, "forecast_windows_24h_past_24h_future").sum(),
                "zero_ge_7": int(part["_z7"].sum()),
                "zero_7_to_30": int(((part["_zero_days"] >= 7) & (part["_zero_days"] < 30)).sum()),
                "zero_ge_30": int(part["_z30"].sum()),
                "zero_ge_30_rate": float(part["_z30"].mean()),
            }
        )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    # Avoid an optional tabulate dependency so the inspection remains
    # reproducible in the existing environment without installing packages.
    view = frame.copy()
    headers = [str(c) for c in view.columns]

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {"building_id", "site_id", "proposed_group", "primaryspaceusage"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    data["_zero_days"] = numeric_series(data, "longest_zero_run_days")
    data["zero_run_bin_explicit"] = data["_zero_days"].map(zero_bin)
    data["zero_ge_7_flag"] = data["_zero_days"] >= 7
    data["zero_ge_30_flag"] = data["_zero_days"] >= 30

    site_stats = make_site_stats(data)
    current = data.groupby("site_id")["proposed_group"].first().astype(str).to_dict()
    if set(current.values()) != set(GROUPS):
        raise ValueError("현재 proposed_group이 예상한 세 그룹과 다릅니다.")
    alternative = deterministic_alternative(site_stats, current)

    data["current_group"] = data["proposed_group"]
    data["alternative_group"] = data["site_id"].map(alternative)
    data["site_moved"] = data["current_group"] != data["alternative_group"]
    data["alternative_proposal_status"] = "검토용 제안; 최종 확정 아님"
    data = data.drop(columns=["_zero_days"])
    data.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    current_summary = group_summary(data, "current_group")
    alternative_summary = group_summary(data, "alternative_group")

    site_out = site_stats.copy()
    site_out["z30_rate"] = site_out["z30_count"] / site_out["buildings"]
    site_out["current_group"] = site_out["site_id"].map(current)
    site_out["alternative_group"] = site_out["site_id"].map(alternative)
    site_out["site_moved"] = site_out["current_group"] != site_out["alternative_group"]
    site_out = site_out.sort_values(["site_moved", "site_id"], ascending=[False, True])

    zero_overall = data["zero_run_bin_explicit"].value_counts()
    zero_order = [
        "0일 (연속 0값 없음)",
        "1일 미만",
        "1일 이상 7일 미만",
        "7일 이상 30일 미만",
        "30일 이상",
    ]
    zero_rows = []
    for label in zero_order:
        zero_rows.append({"구간": label, "건물 수": int(zero_overall.get(label, 0))})
    zero_rows.append({"구간": "7일 이상 전체", "건물 수": int((data["_zero_days"] >= 7).sum()) if "_zero_days" in data else int((numeric_series(pd.read_csv(INPUT), "longest_zero_run_days") >= 7).sum())})
    zero_table = pd.DataFrame(zero_rows)

    current_z = current_summary.copy()
    current_z.insert(1, "구분", "기존")
    alt_z = alternative_summary.copy()
    alt_z.insert(1, "구분", "대체")
    compare = pd.concat([current_z, alt_z], ignore_index=True)

    moved = site_out[site_out["site_moved"]].copy()
    moved_lines = []
    for _, row in moved.iterrows():
        reason = "건물 수·30일 이상 연속 0값 검토 대상·연속 기록 기간을 함께 균형화"
        if row["z30_count"] >= 30:
            reason += "; 장기 0값이 많은 사이트의 집중 완화"
        elif row["z30_count"] == 0:
            reason += "; 장기 0값이 없는 사이트를 보완 배치"
        moved_lines.append(
            {
                "site_id": row["site_id"],
                "기존 그룹": row["current_group"],
                "대체 그룹": row["alternative_group"],
                "건물 수": int(row["buildings"]),
                "30일 이상 연속 0값": int(row["z30_count"]),
                "30일 이상 비율": row["z30_rate"],
                "최장 연속 기록 일수 합": row["complete_days_sum"],
                "이동 사유": reason,
            }
        )
    moved_table = pd.DataFrame(moved_lines)

    report = f"""# BDG2 코호트 사이트 분할 재검토

## 범위

- 입력: `reports/bdg2_cohort_candidates.csv` (후보 {len(data):,}개 건물)
- 원본 전력 CSV는 읽거나 수정하지 않았습니다. 결측값 보간, 기상 자료 다운로드, 모델 학습 및 성능 비교는 수행하지 않았습니다.
- 현재 그룹은 입력 CSV의 `proposed_group`을 그대로 사용했습니다. 대체안은 사이트 전체를 한 그룹에만 배정하는 검토용 제안이며 최종 시험 사이트를 확정하지 않습니다.

## 연속 0값 구간 명칭과 전체 분포

기존의 모호한 집계명을 다음처럼 구분했습니다. `7일 이상 전체`는 두 장기 구간의 합입니다. 한 건물에서 가장 긴 연속 0값 구간 하나만 사용했으며, 0값을 삭제하거나 결측으로 바꾸지 않았습니다.

{md_table(zero_table)}

## 기존안과 대체안 비교

`30일 이상 연속 0값`의 비율은 각 그룹 건물 수를 분모로 계산했습니다. 연속 기록 기간은 `longest_complete_run_days`의 최솟값·중앙값·평균·최댓값으로 요약했습니다.

{md_table(compare, ["구분", "group", "buildings", "sites", "run_min_days", "run_median_days", "run_mean_days", "run_max_days", "run_ge_180", "run_ge_365", "zero_ge_7", "zero_7_to_30", "zero_ge_30", "zero_ge_30_rate", "forecast_windows"])}

### 건물 용도 분포

{md_table(compare[["구분", "group", "usage"]])}

기존안에서는 최종 시험 제안 그룹에 `30일 이상` 검토 대상이 {int(current_summary.loc[current_summary.group == "final_test_proposal", "zero_ge_30"].iloc[0])}개({current_summary.loc[current_summary.group == "final_test_proposal", "zero_ge_30_rate"].iloc[0]:.1%}) 집중되어 있습니다. 대체안에서는 세 그룹이 각각 {", ".join(f'{int(v)}개' for v in alternative_summary["zero_ge_30"])}이며 비율은 {", ".join(f'{v:.1%}' for v in alternative_summary["zero_ge_30_rate"])}입니다. 이는 품질 문제를 제거한 결과가 아니라 그룹 간 분포를 확인하기 위한 재배치입니다.

## 대체 사이트 분할 규칙

대체안은 다음 고정 규칙으로 재현했습니다.

1. 사이트별 건물 수, `30일 이상` 연속 0값 건물 수, 최장 연속 유효 기록 일수의 합을 계산했습니다.
2. 세 그룹의 목표를 전체의 1/3로 두고, 위 세 지표의 정규화 제곱편차 합을 사전 정의했습니다. 연속 기록 기간 항목에는 건물 수·장기 0값 수와 같은 중요도를 주지 않도록 0.25 가중치를 적용했습니다.
3. 현재안에서 시작해 사이트 하나 이동 또는 두 사이트 교환 중 점수가 엄격히 줄어드는 변경만 고정된 사이트명 순서로 적용했습니다. 모델 성능을 사용하지 않았고, 30일 이상 0값 건물을 자동 제외하지 않았습니다.

## 사이트 이동 내역과 사유

{md_table(moved_table)}

대체안의 전체 사이트 배정은 산출물 CSV의 `alternative_group` 열에서 확인할 수 있습니다. 사이트가 그룹을 가로질러 나뉘지 않도록 모든 동일 `site_id` 건물에 같은 그룹을 부여했습니다.

## 전체 사이트별 비교

| site_id | 기존 그룹 | 대체 그룹 | 건물 수 | 30일 이상 연속 0값 | 30일 이상 비율 | 최장 연속 기록 중앙값(일) | 최장 연속 기록 평균(일) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row['site_id']} | {row['current_group']} | {row['alternative_group']} | {int(row['buildings'])} | {int(row['z30_count'])} | {row['z30_rate']:.1%} | {row['complete_days_median']:.1f} | {row['complete_days_mean']:.1f} |" for _, row in site_out.sort_values('site_id').iterrows())}

## 확인이 필요한 사항

- 30일 이상 연속 0값은 계량 오류, 실제 무부하, 장비 정지 중 무엇인지 이 표만으로 구별할 수 없습니다. 따라서 모든 해당 건물을 검토 대상으로 유지했습니다.
- 대체안은 데이터 품질 및 건물 특성의 균형을 위한 제안일 뿐이며, 최종 학습·개발·시험 사이트 확정이나 시간 구간 분할을 의미하지 않습니다.
- 건물 용도와 연속 기록 분포의 차이가 예측 난이도에 미치는 영향은 모델 실험 전에는 판단할 수 없습니다.

## 산출물

- `reports/bdg2_cohort_split_alternative.csv`: 기존 그룹과 대체 그룹을 건물별로 함께 기록
- `scripts/review_bdg2_cohort_split.py`: 위 집계와 대체 배정의 재현 코드
"""
    OUT_MD.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
