"""Freeze a quality-based multi-building BDG2 cohort and temporal split (v2).

Selection uses only existing quality reports. Raw electricity is then read for the
selected source/development columns only to verify continuous periods. Final-test
columns are never requested.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parent.parent))).resolve()
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw" / "bdg2" / "electricity.csv"
COHORT_INPUT = REPORTS / "bdg2_cohort_split_alternative.csv"
COHORT_MANIFEST = REPORTS / "bdg2_multibuilding_cohort_manifest_v2.csv"
TEMPORAL_MANIFEST = REPORTS / "bdg2_multibuilding_temporal_manifest_v2.csv"
REVIEW = REPORTS / "bdg2_multibuilding_selection_review_v2.md"

SOURCE = "source_train_proposal"
DEVELOPMENT = "development_validation_proposal"
FINAL = "final_test_proposal"
QUOTA = {SOURCE: 5, DEVELOPMENT: 2}  # per site: 4*5=20, 6*2=12
ADAPTATION_DAYS = (0, 1, 3, 7, 14, 28)
MIN_RUN_DAYS = 90
MAX_MISSING_RATE = 0.20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def eligibility(data: pd.DataFrame) -> pd.Series:
    return (
        data.alternative_group.isin([SOURCE, DEVELOPMENT])
        & truthy(data.preliminary_eligible)
        & ~truthy(data.no_numeric_values)
        & ~truthy(data.all_numeric_values_zero)
        & (pd.to_numeric(data.longest_complete_run_days, errors="coerce") >= MIN_RUN_DAYS)
        & ~truthy(data.zero_ge_30_flag)
        & (pd.to_numeric(data.negative_count, errors="coerce") == 0)
        & (pd.to_numeric(data.missing_rate_all_records, errors="coerce") <= MAX_MISSING_RATE)
    )


def diversity_quality_select(site_data: pd.DataFrame, quota: int) -> list[str]:
    """Greedy round-robin by use, then missing/run/zero/building quality order."""
    remaining = site_data.copy()
    selected: list[str] = []
    usage_counts = defaultdict(int)
    while len(selected) < quota and len(remaining):
        ranked = remaining.assign(
            _usage_selected=remaining.primaryspaceusage.map(usage_counts).fillna(0),
            _missing=pd.to_numeric(remaining.missing_rate_all_records, errors="coerce"),
            _run=-pd.to_numeric(remaining.longest_complete_run_days, errors="coerce"),
            _zero=pd.to_numeric(remaining.zero_rate_numeric_records, errors="coerce").fillna(np.inf),
        ).sort_values(
            ["_usage_selected", "_missing", "_run", "_zero", "primaryspaceusage", "building_id"],
            kind="mergesort",
        )
        choice = ranked.iloc[0]
        selected.append(str(choice.building_id))
        usage_counts[str(choice.primaryspaceusage)] += 1
        remaining = remaining.loc[remaining.building_id != choice.building_id]
    if len(selected) != quota:
        raise AssertionError(f"Site quota {quota} cannot be met; selected {len(selected)}")
    return selected


def longest_complete_run(timestamps: pd.Series, values: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    ts = pd.DatetimeIndex(timestamps)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric)
    best_start = best_end = -1
    best_len = 0
    current_start = None
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


def count_complete_windows(series: pd.Series, starts: pd.DatetimeIndex) -> int:
    count = 0
    for target_start in starts:
        times = pd.date_range(target_start - pd.Timedelta(hours=24), periods=48, freq="h")
        values = series.reindex(times).to_numpy(float)
        count += int(np.isfinite(values).all())
    return count


def main() -> None:
    cohort = pd.read_csv(COHORT_INPUT)
    if cohort.building_id.duplicated().any():
        raise AssertionError("Input cohort has duplicate building_id")
    cohort["selection_eligible_v2"] = eligibility(cohort)
    selected_ids: list[str] = []
    selected_rank = {}
    for group in (SOURCE, DEVELOPMENT):
        group_data = cohort.loc[
            (cohort.alternative_group == group) & cohort.selection_eligible_v2
        ]
        sites = sorted(group_data.site_id.unique())
        expected_sites = 4 if group == SOURCE else 6
        if len(sites) != expected_sites:
            raise AssertionError(f"Unexpected {group} site count: {len(sites)}")
        for site_id in sites:
            picks = diversity_quality_select(
                group_data.loc[group_data.site_id == site_id], QUOTA[group]
            )
            for rank, building_id in enumerate(picks, 1):
                selected_ids.append(building_id)
                selected_rank[building_id] = rank

    selected = cohort.loc[cohort.building_id.isin(selected_ids)].copy()
    expected = {SOURCE: 20, DEVELOPMENT: 12}
    if selected.groupby("alternative_group").size().to_dict() != expected:
        raise AssertionError(f"Selected count mismatch: {selected.groupby('alternative_group').size().to_dict()}")

    # Read exactly the selected 32 electricity columns. Final-test columns are excluded.
    read_columns = ["timestamp", *sorted(selected_ids)]
    header = pd.read_csv(RAW, nrows=0).columns
    missing = sorted(set(read_columns).difference(header))
    if missing:
        raise AssertionError(f"Selected raw columns missing: {missing}")
    raw = pd.read_csv(RAW, usecols=read_columns)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="raise")
    index = pd.DatetimeIndex(raw.timestamp)
    temporal_rows = []
    actual_run_info = {}
    for row in selected.itertuples(index=False):
        run_start, run_end, run_hours = longest_complete_run(raw.timestamp, raw[row.building_id])
        actual_run_info[row.building_id] = (run_start, run_end, run_hours)
        if run_hours < MIN_RUN_DAYS * 24:
            raise AssertionError(f"{row.building_id}: actual run shorter than {MIN_RUN_DAYS} days")
        series = pd.Series(pd.to_numeric(raw[row.building_id], errors="coerce").to_numpy(float), index=index)
        if row.alternative_group == SOURCE:
            val_target_start = run_end - pd.Timedelta(days=28) + pd.Timedelta(hours=1)
            val_target_end = run_end
            train_target_start = run_start + pd.Timedelta(hours=24)
            train_target_end = val_target_start - pd.Timedelta(hours=1)
            train_starts = pd.date_range(
                train_target_start, train_target_end - pd.Timedelta(hours=23), freq="h"
            )
            val_starts = pd.date_range(
                val_target_start, val_target_end - pd.Timedelta(hours=23), freq="h"
            )
            temporal_rows.append(
                {
                    "building_id": row.building_id,
                    "site_id": row.site_id,
                    "primaryspaceusage": row.primaryspaceusage,
                    "alternative_group": row.alternative_group,
                    "role": "source_pretraining",
                    "adaptation_days": np.nan,
                    "actual_longest_run_start": run_start,
                    "actual_longest_run_end": run_end,
                    "actual_longest_run_hours": run_hours,
                    "train_target_start": train_target_start,
                    "train_target_end": train_target_end,
                    "train_sample_count": count_complete_windows(series, train_starts),
                    "validation_target_start": val_target_start,
                    "validation_target_end": val_target_end,
                    "validation_sample_count": count_complete_windows(series, val_starts),
                    "adaptation_target_start": pd.NaT,
                    "adaptation_target_end": pd.NaT,
                    "adaptation_train_sample_count": np.nan,
                    "test_target_start": pd.NaT,
                    "test_target_end": pd.NaT,
                    "daily_test_sample_count": np.nan,
                    "final_test_power_used": False,
                }
            )
        else:
            common_start = run_end - pd.Timedelta(days=57) + pd.Timedelta(hours=1)
            adaptation_start = common_start + pd.Timedelta(hours=24)
            test_start = adaptation_start + pd.Timedelta(days=28)
            test_end = run_end
            test_starts = pd.date_range(test_start, test_end - pd.Timedelta(hours=23), freq="24h")
            if len(test_starts) != 28:
                raise AssertionError(f"{row.building_id}: expected 28 daily test starts")
            for days in ADAPTATION_DAYS:
                if days == 0:
                    adaptation_end = pd.NaT
                    adaptation_count = 0
                else:
                    adaptation_end = adaptation_start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
                    starts = pd.date_range(
                        adaptation_start,
                        adaptation_end - pd.Timedelta(hours=23),
                        freq="h",
                    )
                    adaptation_count = count_complete_windows(series, starts)
                temporal_rows.append(
                    {
                        "building_id": row.building_id,
                        "site_id": row.site_id,
                        "primaryspaceusage": row.primaryspaceusage,
                        "alternative_group": row.alternative_group,
                        "role": "development_adaptation_evaluation",
                        "adaptation_days": days,
                        "actual_longest_run_start": run_start,
                        "actual_longest_run_end": run_end,
                        "actual_longest_run_hours": run_hours,
                        "train_target_start": pd.NaT,
                        "train_target_end": pd.NaT,
                        "train_sample_count": np.nan,
                        "validation_target_start": pd.NaT,
                        "validation_target_end": pd.NaT,
                        "validation_sample_count": np.nan,
                        "adaptation_target_start": adaptation_start if days else pd.NaT,
                        "adaptation_target_end": adaptation_end,
                        "adaptation_train_sample_count": adaptation_count,
                        "test_target_start": test_start,
                        "test_target_end": test_end,
                        "daily_test_sample_count": 28,
                        "final_test_power_used": False,
                    }
                )

    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(TEMPORAL_MANIFEST, index=False, encoding="utf-8-sig")

    manifest_rows = []
    for row in cohort.itertuples(index=False):
        is_selected = row.building_id in selected_ids
        is_final = row.alternative_group == FINAL
        run = actual_run_info.get(row.building_id)
        manifest_rows.append(
            {
                "building_id": row.building_id,
                "site_id": row.site_id,
                "primaryspaceusage": row.primaryspaceusage,
                "alternative_group": row.alternative_group,
                "selected_for_multibuilding_v2": is_selected,
                "selection_role": (
                    "source_pretraining"
                    if is_selected and row.alternative_group == SOURCE
                    else "development_adaptation_evaluation"
                    if is_selected
                    else "assignment_only_final_power_not_read"
                    if is_final
                    else "not_selected"
                ),
                "selection_eligible_v2": False if is_final else bool(row.selection_eligible_v2),
                "selection_rank_within_site": selected_rank.get(row.building_id, np.nan),
                "missing_rate_all_records": np.nan if is_final else row.missing_rate_all_records,
                "longest_complete_run_days_reported": np.nan if is_final else row.longest_complete_run_days,
                "zero_ge_30_flag": np.nan if is_final else row.zero_ge_30_flag,
                "negative_count": np.nan if is_final else row.negative_count,
                "actual_verified_run_start": run[0] if run else pd.NaT,
                "actual_verified_run_end": run[1] if run else pd.NaT,
                "actual_verified_run_hours": run[2] if run else np.nan,
                "power_column_read_in_v2_preparation": bool(is_selected and not is_final),
                "selection_rule_version": "bdg2_multibuilding_v2_quality_site_usage_v1",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(COHORT_MANIFEST, index=False, encoding="utf-8-sig")

    selected_view = manifest.loc[manifest.selected_for_multibuilding_v2]
    site_counts = selected_view.groupby(["alternative_group", "site_id"]).size()
    use_counts = selected_view.groupby(["alternative_group", "primaryspaceusage"]).size()
    dev_ranges = temporal.loc[temporal.adaptation_days.eq(0)].copy()
    report = f"""# BDG2 다건물 개발 실험 선정 및 시간 분할 v2

## 고정 결과

- Source 출발 학습 건물: **20개**, 4개 사이트에서 각 5개.
- 개발·검증 건물: **12개**, 6개 사이트에서 각 2개.
- 최종 시험 사이트: ID와 기존 `alternative_group` 배정만 새 cohort manifest에 보존했다. 해당 전력 열은 읽지 않았다.
- 선정과 시간 분할은 모델 실행 전에 저장했으며 모델 성능을 사용하지 않았다.

## 사전 선정 기준

1. 기존 `alternative_group`이 Source 또는 개발 그룹일 것.
2. 기존 `preliminary_eligible=True`, 숫자 기록 부재·전부 0 건물이 아닐 것.
3. 보고된 최장 연속 유효 기록이 90일 이상일 것.
4. 30일 이상 연속 0값 검토 대상이 아닐 것.
5. 음수값이 0개이고 전체 결측률이 20% 이하일 것.
6. 사이트별 고정 할당량 안에서 용도별 현재 선택 수가 적은 후보를 우선하고, 동률이면 결측률 오름차순, 최장 연속 기록 내림차순, 0 비율 오름차순, 용도·건물 ID 순으로 선택했다.
7. 선택 후 raw의 해당 32개 열만 읽어 실제 유한값·1시간 연속 최장 구간이 90일 이상인지 다시 확인했다. 실패 시 대체 선정 없이 중단하도록 구현했다.

## 사이트별 선택 수

```
{site_counts.to_string()}
```

## 용도별 선택 수

```
{use_counts.to_string()}
```

## 시간 분할 규칙

- Source: 각 건물의 실제 최장 연속 구간을 사용한다. 마지막 28일 정답을 Source 검증으로 고정하고, 그 이전 완전한 24→24시간 창만 Source 학습에 쓴다.
- 개발: 각 건물 최장 연속 구간의 마지막 57일을 고정한다. 첫 24시간은 이력, 다음 28일은 최대 적응 구간, 마지막 28일은 하루 간격 28개 평가 창이다.
- 따라서 개발 건물마다 달력 날짜는 다를 수 있지만, 각 건물의 모든 적응 조건은 동일한 고정 시험 기간을 사용한다.
- 0·1·3·7·14·28일 적응 샘플 수는 각각 0·1·49·145·313·649개이다.
- 결측 보간과 시간대 변환은 수행하지 않았다.

## 개발 건물별 고정 평가 기간

```
{dev_ranges[['building_id','site_id','test_target_start','test_target_end','actual_longest_run_hours']].to_string(index=False)}
```

## 고정 파일 해시

- cohort manifest SHA-256: `{sha256(COHORT_MANIFEST)}`
- temporal manifest SHA-256: `{sha256(TEMPORAL_MANIFEST)}`
- 입력 alternative cohort SHA-256: `{sha256(COHORT_INPUT)}`
- raw electricity SHA-256: `{sha256(RAW)}`

이 선정은 개발용 다건물 예비 실험을 위한 것이며 최종 시험 사이트 또는 논문 최종 코호트를 확정하지 않는다.
"""
    REVIEW.write_text(report, encoding="utf-8")
    print(selected_view.groupby("alternative_group").size().to_string())
    print(f"wrote {COHORT_MANIFEST}")
    print(f"wrote {TEMPORAL_MANIFEST}")
    print(f"cohort_sha256={sha256(COHORT_MANIFEST)}")
    print(f"temporal_sha256={sha256(TEMPORAL_MANIFEST)}")


if __name__ == "__main__":
    main()
