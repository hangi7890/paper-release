"""Freeze C0 feature and evaluation rules before C0 model training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parent.parent))).resolve()
REPORTS = ROOT / "reports"
COHORT = REPORTS / "bdg2_multibuilding_cohort_manifest_v2.csv"
TEMPORAL = REPORTS / "bdg2_multibuilding_temporal_manifest_v2.csv"
PROTOCOL = REPORTS / "bdg2_stage6_c0_protocol_manifest_v3.csv"
REVIEW = REPORTS / "bdg2_stage6_c0_protocol_v3.md"
SOURCE = "source_train_proposal"
DEVELOPMENT = "development_validation_proposal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cohort = pd.read_csv(COHORT)
    temporal = pd.read_csv(TEMPORAL)
    selected = cohort.loc[cohort.selected_for_multibuilding_v2].copy()
    rows = []
    for building in selected.itertuples(index=False):
        building_temporal = temporal.loc[temporal.building_id == building.building_id]
        if building.alternative_group == SOURCE:
            if len(building_temporal) != 1:
                raise AssertionError(f"{building.building_id}: invalid source temporal rows")
            time_row = building_temporal.iloc[0]
            rows.append(
                {
                    "building_id": building.building_id,
                    "site_id": building.site_id,
                    "alternative_group": building.alternative_group,
                    "role": "source_train_validation",
                    "adaptation_days": "",
                    "train_samples": int(time_row.train_sample_count),
                    "validation_samples": int(time_row.validation_sample_count),
                    "test_windows": "",
                    "test_target_start": "",
                    "test_target_end": "",
                    "input_condition": "C0",
                    "sequence_length": 24,
                    "input_features_per_step": 6,
                    "output_hours": 24,
                    "future_observed_values_used": False,
                    "final_test_power_used": False,
                }
            )
        elif building.alternative_group == DEVELOPMENT:
            for time_row in building_temporal.itertuples(index=False):
                rows.append(
                    {
                        "building_id": building.building_id,
                        "site_id": building.site_id,
                        "alternative_group": building.alternative_group,
                        "role": "development_adaptation_evaluation",
                        "adaptation_days": int(time_row.adaptation_days),
                        "train_samples": int(time_row.adaptation_train_sample_count),
                        "validation_samples": 0,
                        "test_windows": int(time_row.daily_test_sample_count),
                        "test_target_start": time_row.test_target_start,
                        "test_target_end": time_row.test_target_end,
                        "input_condition": "C0",
                        "sequence_length": 24,
                        "input_features_per_step": 6,
                        "output_hours": 24,
                        "future_observed_values_used": False,
                        "final_test_power_used": False,
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(PROTOCOL, index=False, encoding="utf-8-sig")
    report = f"""# BDG2 Stage 6 C0 고정 프로토콜 v3

## C0 입력 정의

각 예측 샘플은 길이 24의 시퀀스이다. 시퀀스의 horizon `h=1..24` 행에는 다음 6개 값이 들어간다.

1. 예측 시작 전 과거 전력의 대응 lag 값 1개: `t-24 .. t-1`.
2. 대응하는 목표 시각 `t .. t+23`의 `sin(2π·hour/24)`.
3. 대응하는 목표 시각의 `cos(2π·hour/24)`.
4. 대응하는 목표 요일의 `sin(2π·weekday/7)`, Monday=0.
5. 대응하는 목표 요일의 `cos(2π·weekday/7)`.
6. 목표 요일이 Saturday/Sunday이면 1인 weekend indicator.

목표 시각의 달력은 예측 시점에 확정된 결정적 정보이다. 미래 실제 전력, 미래 실제 기상, 사후 관측 센서는 사용하지 않는다.

## 모델·평가 고정 규칙

- LSTM sequence length 24, hidden size 32, 1 layer, 선형 24시간 출력.
- v2와 비교해 바뀌는 구조는 input size 1→6뿐이다.
- Source Adam 0.001, batch 256, 최대 100 epoch, patience 10, min delta 1e-5.
- Source 20개 건물 학습·정규화와 Source 검증만으로 checkpoint를 선택한다.
- 개발 12개 건물은 Source 학습·정규화·checkpoint 선택에서 제외한다.
- 개발 주 평가는 v2와 동일한 건물별 고정 28일, 하루 간격 28개 비중첩 창이다.
- 0일은 Source-only이다. 1·3·7·14·28일은 LSTM을 고정하고 출력층만 Adam 0.0005, 50 epoch 적응한다.
- 사전 고정 시드: 20260924, 20260925, 20260926.
- 시험 결과를 보고 구조·학습률·epoch·분할을 변경하지 않는다.

## 고정 입력 해시

- v2 cohort manifest: `{sha256(COHORT)}`
- v2 temporal manifest: `{sha256(TEMPORAL)}`
- C0 protocol manifest: `{sha256(PROTOCOL)}`

최종 시험 사이트의 전력값은 읽지 않으며 최종 시험은 실행하지 않는다.
"""
    REVIEW.write_text(report, encoding="utf-8")
    print(f"wrote {PROTOCOL}")
    print(f"wrote {REVIEW}")
    print(f"protocol_sha256={sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
