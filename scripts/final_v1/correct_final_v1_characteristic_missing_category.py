"""Documented addendum: retain missing usage as a category in exploration."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(HERE.parents[1]))).resolve()
REPORTS = ROOT / "reports"
SUPPORT = REPORTS / "final_v1"
PRESERVED = SUPPORT / "addendum_pre_characteristic_fix"
sys.path.insert(0, str(HERE))

from analyze_final_v1_gate3 import characteristic_analysis, md_table  # noqa: E402


CHARACTERISTICS = SUPPORT / "final_v1_building_characteristics.csv"
EVENTS = REPORTS / "final_v1_failure_events.csv"
ASSOCIATIONS = SUPPORT / "final_v1_characteristic_associations.csv"
GROUPS = SUPPORT / "final_v1_characteristic_group_summary.csv"
CHAR_REPORT = REPORTS / "final_v1_building_characteristics_exploratory.md"
FINAL_REPORT = REPORTS / "final_v1_final_test_report.md"
ADDENDUM = REPORTS / "final_v1_software_addendum_001.md"
LOG = SUPPORT / "software_addendum_001.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if ADDENDUM.exists() or LOG.exists():
        raise FileExistsError("Addendum 001 already exists")
    preserved_map = {
        "association": PRESERVED / "final_v1_characteristic_associations_pre_fix.csv",
        "group": PRESERVED / "final_v1_characteristic_group_summary_pre_fix.csv",
        "characteristic_report": PRESERVED / "final_v1_building_characteristics_exploratory_pre_fix.md",
        "final_report": PRESERVED / "final_v1_final_test_report_pre_fix.md",
        "analysis_code": ROOT / "scripts" / "final_v1" / "addendum_pre_characteristic_fix" / "analyze_final_v1_gate3_pre_characteristic_fix.py",
    }
    if not all(path.exists() for path in preserved_map.values()):
        raise FileNotFoundError("Pre-correction artifacts were not fully preserved")
    before = {name: sha256(path) for name, path in preserved_map.items()}

    characteristics = pd.read_csv(CHARACTERISTICS)
    events = pd.read_csv(EVENTS)
    associations, groups = characteristic_analysis(characteristics, events)
    check = associations.loc[
        associations.response.eq("seed_mean_mae_delta")
        & associations.feature.eq("context_24h_absolute_mean_gap_in_source_config_std")
    ]
    if len(check) != 5 or not check.n_complete.eq(453).all():
        raise AssertionError("Correction failed to retain all 453 buildings")
    associations.to_csv(ASSOCIATIONS, index=False, encoding="utf-8", float_format="%.17g")
    groups.to_csv(GROUPS, index=False, encoding="utf-8", float_format="%.17g")

    strongest = associations.loc[associations.response.eq("seed_mean_mae_delta")].dropna(subset=["spearman_rho"]).copy()
    strongest["abs_rho"] = strongest.spearman_rho.abs()
    strongest = strongest.sort_values(
        ["adaptation_days", "abs_rho", "feature"], ascending=[True, False, True]
    ).groupby("adaptation_days").head(3)
    missing_buildings = int(characteristics.loc[characteristics.adaptation_days.eq(0), "primaryspaceusage"].isna().sum())
    CHAR_REPORT.write_text(
        f"""# final_v1 건물 특성 탐색 분석

## 범위와 정보 시점

- 모든 결과는 **탐색적**이며 건물 제외, 모델 선택, H1~H3 판정에 사용하지 않았다.
- 0일은 첫 예측 직전 24시간 context만 사용했고 장기 Target 통계는 `NA`로 유지했다.
- 비제로 예산은 초기 적응 직전 24시간과 허용된 적응 관측 종료까지만 사용했다.
- 모든 `feature_information_cutoff`는 시험 시작보다 이르다. 시험 정답 실제값은 특성 계산에 사용하지 않았다.
- 일간 반복성은 최소 24개 lag pair, 주간 반복성은 최소 168개 lag pair가 있고 양쪽이 비상수일 때만 Pearson 상관을 계산했다.
- Source 규모 기준은 기존 Source 20개 학습 구간 표를 그대로 사용했다.

## 자료 규모

- 특성 행: {len(characteristics)} = 453개 건물 × 6개 예산.
- 사이트: 7개; 공식 용도값이 있는 범주 13개; 용도 미기재 건물 {missing_buildings}개는 `<missing>` 범주로 보존했다.
- 적응 기반 장기 특성을 사용할 수 있는 행: {int(characteristics.available_pretest_hours.gt(0).sum())}개.
- 주간 반복성 정의 가능 행: {int(characteristics.weekly_repeatability_pearson.notna().sum())}개.

## 예산별 `ΔMAE_source`와 절댓값 상관이 큰 탐색 항목

{md_table(strongest, ['adaptation_days','feature','n_complete','spearman_rho','status'], {'spearman_rho':'{:.4f}'})}

이 표는 작은 수의 독립 사이트와 반복 예산 관측에서 얻은 탐색적 상관이다. 인과관계, 사전 실패 판별 규칙, 전체 BDG2의 일반 법칙으로 해석하지 않는다. 다중 탐색에 대한 확증적 유의성 검정을 수행하지 않았다.

## Software addendum 001

최초 Gate 3 탐색 코드가 `primaryspaceusage` 결측 18개(Swan 사이트)를 pandas `groupby` 기본 동작으로 제외했다. 예측·metric·F1~F4·H1~H3에는 영향이 없었다. 최초 파일은 `reports/final_v1/addendum_pre_characteristic_fix/`에 보존했고, 결측을 임의 용도로 대체하지 않고 `<missing>`이라는 명시적 범주로 유지해 이 보고서와 탐색 표만 수정했다. 자세한 해시는 `reports/final_v1_software_addendum_001.md`에 있다.
""",
        encoding="utf-8",
    )

    with FINAL_REPORT.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## Software addendum 001\n\n"
            "Gate 3 후속 점검에서 공식 건물 용도값이 비어 있는 Swan 사이트 18개가 탐색적 특성 상관의 `groupby`에서 제외된 것을 발견했다. "
            "핵심 예측, 독립 metric, F1~F4, H1~H3에는 영향이 없다. 최초 탐색 산출물과 코드는 보존했으며, 결측 용도를 `<missing>` 범주로 유지해 탐색 상관·그룹표·특성 보고서만 수정했다. "
            "모델, cohort, 시간 경계, 예측, 가설 정의 및 결과는 변경하지 않았다. 자세한 내용은 `reports/final_v1_software_addendum_001.md`에 기록했다.\n"
        )

    after = {
        "association": sha256(ASSOCIATIONS),
        "group": sha256(GROUPS),
        "characteristic_report": sha256(CHAR_REPORT),
        "final_report": sha256(FINAL_REPORT),
    }
    ADDENDUM.write_text(
        f"""# final_v1 Software addendum 001

## 발견 시각과 오류

- 기록 시각(UTC): `{datetime.now(timezone.utc).isoformat()}`
- 오류: 공식 `primaryspaceusage`가 비어 있는 Swan 사이트 18개가 탐색 특성 분석의 pandas `groupby` 기본 `dropna=True` 때문에 제외됨.
- 최초 표본: 해당 탐색 상관의 완전 사례 435개.
- 수정 표본: 453개. 결측 용도는 값을 추정하지 않고 문자열 범주 `<missing>`으로 명시.

## 영향 범위

- 영향 있음: 탐색적 특성 상관표, 용도별 탐색 그룹표, 탐색 보고서의 상관 수치와 표본 수.
- 영향 없음: raw prediction, 독립 MAE/RMSE, 데이터 효율성, F1~F4, H1~H3, bootstrap, cohort, temporal split, 모델 checkpoint.
- 모델 재실행·재학습: 없음.
- 가설·설정·기간·건물 변경: 없음.

## 보존·수정

최초 파일과 코드는 `reports/final_v1/addendum_pre_characteristic_fix/` 및 `scripts/final_v1/addendum_pre_characteristic_fix/`에 보존했다. 분석 함수는 결측 용도를 `<missing>` 범주로 보존하도록 수정했고 탐색 파일만 재계산했다.

| artifact | before SHA-256 | after SHA-256 |
|---|---|---|
| characteristic associations | `{before['association']}` | `{after['association']}` |
| characteristic group summary | `{before['group']}` | `{after['group']}` |
| characteristic report | `{before['characteristic_report']}` | `{after['characteristic_report']}` |
| final report | `{before['final_report']}` | `{after['final_report']}` |
| pre-fix analysis code | `{before['analysis_code']}` | `{sha256(ROOT / 'scripts' / 'final_v1' / 'analyze_final_v1_gate3.py')}` |

이 수정은 전향적으로 고정된 최종 endpoint의 값이나 판정을 바꾸지 않았다.
""",
        encoding="utf-8",
    )
    LOG.write_text(
        "\n".join(
            [
                f"completed_utc={datetime.now(timezone.utc).isoformat()}",
                "software_error=primaryspaceusage_missing_dropped_from_exploratory_groupby",
                "affected_buildings=18",
                "affected_site=Swan",
                "confirmatory_endpoints_affected=False",
                "model_rerun=False",
                f"addendum_sha256={sha256(ADDENDUM)}",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print("Software addendum 001 complete: exploratory n corrected 435 -> 453; confirmatory endpoints unchanged")


if __name__ == "__main__":
    main()
