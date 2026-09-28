"""Freeze the final_v1 cohort and protocol before final-test power access.

This Gate 0 script reads only existing candidate/quality manifests and model
configuration bytes.  It must never open the raw electricity CSV or any final
prediction/metric artifact.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"
SCRIPTS_DIR = ROOT / "scripts" / "final_v1"
MODELS_DIR = ROOT / "models" / "final_v1"
SUPPORT_DIR = REPORTS / "final_v1"

CANDIDATES = REPORTS / "bdg2_cohort_split_alternative.csv"
V5_PROTOCOL = REPORTS / "bdg2_v5_final_analysis_protocol_draft.md"
C0_CONFIG = ROOT / "models" / "bdg2_stage6_v3" / "bdg2_stage6_c0_lstm_config_v3.json"
CHECKPOINTS = [
    ROOT / "models" / "bdg2_stage6_v3" / f"bdg2_stage6_c0_lstm_seed_{seed}_v3.pt"
    for seed in (20260924, 20260925, 20260926)
]

COHORT_OUT = REPORTS / "final_v1_cohort_manifest.csv"
PROTOCOL_OUT = REPORTS / "final_v1_protocol_manifest.md"
HASH_OUT = SUPPORT_DIR / "gate0_frozen_hashes.csv"
LOG_OUT = SUPPORT_DIR / "gate0_execution.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def exclusion_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["meets_preliminary_candidate_condition"]):
        reasons.append("existing_preliminary_candidate_condition_false")
    if not bool(row["meets_90d_complete_run"]):
        reasons.append("longest_complete_run_lt_90d")
    if bool(row["no_numeric_values"]):
        reasons.append("no_numeric_values")
    if bool(row["all_numeric_values_zero"]):
        reasons.append("all_numeric_values_zero")
    if bool(row["zero_ge_30_flag"]):
        reasons.append("continuous_zero_run_ge_30d")
    return ";".join(reasons)


def main() -> None:
    for directory in (SCRIPTS_DIR, MODELS_DIR, SUPPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for output in (COHORT_OUT, PROTOCOL_OUT, HASH_OUT, LOG_OUT):
        if output.exists():
            raise FileExistsError(f"Gate 0 refuses to overwrite existing artifact: {output}")

    source = pd.read_csv(CANDIDATES)
    final_rows = source.loc[source.alternative_group.eq("final_test_proposal")].copy()
    if final_rows.empty:
        raise AssertionError("No final_test_proposal rows")
    if final_rows.building_id.duplicated().any():
        raise AssertionError("Duplicate building_id in final proposal")

    final_rows["meets_preliminary_candidate_condition"] = bool_series(final_rows.preliminary_eligible)
    final_rows["meets_90d_complete_run"] = pd.to_numeric(
        final_rows.longest_complete_run_days, errors="coerce"
    ).ge(90.0)
    final_rows["no_numeric_values"] = bool_series(final_rows.no_numeric_values)
    final_rows["all_numeric_values_zero"] = bool_series(final_rows.all_numeric_values_zero)
    final_rows["zero_ge_30_flag"] = bool_series(final_rows.zero_ge_30_flag)
    final_rows["eligible_by_frozen_quality_rule"] = (
        final_rows.meets_preliminary_candidate_condition
        & final_rows.meets_90d_complete_run
        & ~final_rows.no_numeric_values
        & ~final_rows.all_numeric_values_zero
        & ~final_rows.zero_ge_30_flag
    )
    final_rows["selection_status"] = final_rows.eligible_by_frozen_quality_rule.map(
        {True: "selected_for_final_v1", False: "excluded_pre_access"}
    )
    final_rows["exclusion_reason"] = final_rows.apply(exclusion_reason, axis=1)
    final_rows["selection_used_model_performance"] = False
    final_rows["final_power_read_at_gate0"] = False
    final_rows["selection_rule_version"] = "final_v1_gate0_fixed_quality_all_eligible_v1"
    final_rows["selection_source_file"] = CANDIDATES.relative_to(ROOT).as_posix()

    columns = [
        "building_id", "site_id", "primaryspaceusage", "sqm", "timezone",
        "alternative_group", "missing_rate_all_records", "numeric_records",
        "longest_complete_run_days", "forecast_windows_24h_past_24h_future",
        "longest_zero_run_days", "no_numeric_values", "all_numeric_values_zero",
        "zero_ge_30_flag", "meets_preliminary_candidate_condition",
        "meets_90d_complete_run", "eligible_by_frozen_quality_rule",
        "selection_status", "exclusion_reason", "selection_used_model_performance",
        "final_power_read_at_gate0", "selection_rule_version", "selection_source_file",
    ]
    manifest = final_rows[columns].sort_values(["site_id", "building_id"]).reset_index(drop=True)
    manifest.to_csv(COHORT_OUT, index=False, encoding="utf-8", lineterminator="\n")

    selected = manifest.loc[manifest.selection_status.eq("selected_for_final_v1")]
    excluded = manifest.loc[manifest.selection_status.eq("excluded_pre_access")]
    site_rows = []
    for site_id, frame in manifest.groupby("site_id", sort=True):
        site_rows.append([
            site_id,
            len(frame),
            int(frame.selection_status.eq("selected_for_final_v1").sum()),
            int(frame.selection_status.eq("excluded_pre_access").sum()),
        ])
    reason_rows = (
        excluded.exclusion_reason.value_counts(dropna=False).rename_axis("reason").reset_index(name="count")
    )

    hashes = [
        {"path": CANDIDATES.relative_to(ROOT).as_posix(), "role": "gate0_quality_input", "sha256": sha256(CANDIDATES)},
        {"path": COHORT_OUT.relative_to(ROOT).as_posix(), "role": "frozen_final_cohort", "sha256": sha256(COHORT_OUT)},
        {"path": V5_PROTOCOL.relative_to(ROOT).as_posix(), "role": "approved_v5_protocol_basis", "sha256": sha256(V5_PROTOCOL)},
        {"path": C0_CONFIG.relative_to(ROOT).as_posix(), "role": "frozen_c0_config", "sha256": sha256(C0_CONFIG)},
    ]
    hashes.extend(
        {"path": path.relative_to(ROOT).as_posix(), "role": "frozen_source_checkpoint", "sha256": sha256(path)}
        for path in CHECKPOINTS
    )
    pd.DataFrame(hashes).to_csv(HASH_OUT, index=False, encoding="utf-8", lineterminator="\n")

    protocol_text = f"""# final_v1 독립 최종시험 프로토콜 manifest — Gate 0 동결

## 상태

- Gate 0: **PASS**
- 생성 시각(UTC): `{datetime.now(timezone.utc).isoformat()}`
- 이 단계에서 final 전력값·예측 성능·metric 열람: **없음**
- 선정에 모델 결과 사용: **없음**
- 기존 v2~v5 artifact 수정: **없음**

## 고정 연구 조건

- 입력 C0: 과거 전력 24시간 + 예측 대상 24시간의 사전 가용 달력 정보
- 출력: 직후 24시간 전력
- Source checkpoint: 기존 Source 20개 C0 LSTM, 시드 `20260924, 20260925, 20260926`
- Target 적응: 기존 output-head-only, 예산 `0, 1, 3, 7, 14, 28`일
- 기준선: `previous_day`, `weekly_same_hour`
- H1: 28일 사이트 동일 가중 `ΔMAE_source`; 10,000회 site-cluster percentile bootstrap, seed `20260925`
- H2/H3: effect size와 같은 bootstrap CI만 보고하며 p-value/Holm 검정 없음
- F1~F4: v5 정의 및 `EPS=1e-12`
- 성능 확인 뒤 cohort·기간·설정 변경 및 재튜닝 금지

## 사전 고정 cohort 규칙

1. `alternative_group == final_test_proposal`인 기존 배정만 고려한다.
2. 기존 `preliminary_eligible` 조건과 최장 완전연속 기록 90일 이상을 요구한다.
3. 숫자 기록 없음, 숫자 기록 전체 0, 연속 0값 30일 이상 flag를 제외한다.
4. 위 조건을 만족하는 건물을 일부 샘플링하지 않고 모두 포함한다.
5. Gate 1 무결성 실패 건물은 교체하거나 다른 기간을 탐색하지 않고 `data_integrity_failure`로 기록한다.

## Gate 0 결과

- final proposal: **{len(manifest)}개 건물, {manifest.site_id.nunique()}개 사이트**
- 사전 적격: **{len(selected)}개 건물, {selected.site_id.nunique()}개 사이트**
- 사전 제외: **{len(excluded)}개 건물**

{md_table(["site_id", "proposal", "selected", "excluded"], site_rows)}

### 제외 사유

{md_table(["사유", "건물 수"], reason_rows[["reason", "count"]].values.tolist())}

Mouse와 Shrew는 고정 품질 규칙을 통과한 건물이 0개이다. 이를 성능과 무관하게 그대로 유지한다.

## 동결 해시

{md_table(["경로", "역할", "SHA-256"], [[row['path'], row['role'], row['sha256']] for row in hashes])}

`final_v1_cohort_manifest.csv`의 SHA-256은 `{sha256(COHORT_OUT)}`이다. 이 manifest 이후 final 전력 접근이 허용된다. 소프트웨어 결함 수정 시 최초 실행을 보존하고 addendum을 작성한다.
"""
    PROTOCOL_OUT.write_text(protocol_text, encoding="utf-8")
    protocol_hash = sha256(PROTOCOL_OUT)

    LOG_OUT.write_text(
        "\n".join([
            f"completed_utc={datetime.now(timezone.utc).isoformat()}",
            "gate=0",
            "status=PASS",
            "raw_final_power_read=False",
            "performance_metric_read=False",
            f"proposal_buildings={len(manifest)}",
            f"proposal_sites={manifest.site_id.nunique()}",
            f"selected_buildings={len(selected)}",
            f"selected_sites={selected.site_id.nunique()}",
            f"excluded_buildings={len(excluded)}",
            f"cohort_sha256={sha256(COHORT_OUT)}",
            f"v5_protocol_sha256={sha256(V5_PROTOCOL)}",
            f"final_v1_protocol_manifest_sha256={protocol_hash}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"selected={len(selected)} sites={selected.site_id.nunique()} excluded={len(excluded)}")
    print(f"cohort_sha256={sha256(COHORT_OUT)}")
    print(f"v5_protocol_sha256={sha256(V5_PROTOCOL)}")
    print(f"protocol_manifest_sha256={protocol_hash}")


if __name__ == "__main__":
    main()
