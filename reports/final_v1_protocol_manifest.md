# final_v1 독립 최종시험 프로토콜 manifest — Gate 0 동결

## 상태

- Gate 0: **PASS**
- 생성 시각(UTC): `2026-09-25T02:38:25.518154+00:00`
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

- final proposal: **510개 건물, 9개 사이트**
- 사전 적격: **453개 건물, 7개 사이트**
- 사전 제외: **57개 건물**

| site_id | proposal | selected | excluded |
|---|---|---|---|
| Bobcat | 23 | 22 | 1 |
| Bull | 122 | 111 | 11 |
| Eagle | 104 | 94 | 10 |
| Gator | 67 | 63 | 4 |
| Hog | 147 | 135 | 12 |
| Moose | 13 | 10 | 3 |
| Mouse | 7 | 0 | 7 |
| Shrew | 9 | 0 | 9 |
| Swan | 18 | 18 | 0 |

### 제외 사유

| 사유 | 건물 수 |
|---|---|
| continuous_zero_run_ge_30d | 57 |

Mouse와 Shrew는 고정 품질 규칙을 통과한 건물이 0개이다. 이를 성능과 무관하게 그대로 유지한다.

## 동결 해시

| 경로 | 역할 | SHA-256 |
|---|---|---|
| reports/bdg2_cohort_split_alternative.csv | gate0_quality_input | 6c77b0d5b07d0c1a16279a5100cb7ef092526265afc55734778e42fdc74ff902 |
| reports/final_v1_cohort_manifest.csv | frozen_final_cohort | 8216847545f7814c12f3e2edb31523ac11da0ae19cc142985f736bbd7185d3ec |
| reports/bdg2_v5_final_analysis_protocol_draft.md | approved_v5_protocol_basis | 351fbbe6b2aac178d557bbcb31865fbd642739c560ae02a71db910bc645a83de |
| models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json | frozen_c0_config | 9eccb94fcf2682e9a7a5ee205aa1ce250082dd6e7f7750337cb61ffed0ebe7d1 |
| models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260924_v3.pt | frozen_source_checkpoint | 88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b |
| models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260925_v3.pt | frozen_source_checkpoint | 202a610bc6f4c89c32335b674e80f372c03699c3df33c22e30bfae80b8f951f5 |
| models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260926_v3.pt | frozen_source_checkpoint | 708e3321db56f8828a5c59b8cfdbbf02c4b31feff9d1659d1be55d7a61e20d89 |

`final_v1_cohort_manifest.csv`의 SHA-256은 `8216847545f7814c12f3e2edb31523ac11da0ae19cc142985f736bbd7185d3ec`이다. 이 manifest 이후 final 전력 접근이 허용된다. 소프트웨어 결함 수정 시 최초 실행을 보존하고 addendum을 작성한다.
