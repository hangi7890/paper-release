# final_v1 C0 독립 최종시험 보고서

## 완료 상태

- Gate 0 cohort 동결: 완료.
- Gate 1 구조 검증: **PASS**, 453개 건물·7개 사이트 모두 통과.
- Gate 2 단일 모델 실행: 완료. 저장 raw prediction 6,088,320행, 실행 오류 0개.
- Gate 3 독립 metric 재계산: **PASS**. 실행 metric과 최대 차이 `1.71e-13` (`≤1e-10`).
- 최종 실행 뒤 설정·건물·기간·시드·학습률·epoch 변경: **없음**.

## cohort와 제외

- 기존 `final_test_proposal`: 510개 건물, 9개 사이트.
- 품질 규칙 적격·최종 평가: 453개, 7개 사이트.
- Gate 0 제외: 57개. 전부 정확한 사유 `continuous_zero_run_ge_30d`이며 manifest에 건물별로 보존했다.
- Gate 1 `data_integrity_failure`: 0개. 대체 건물·대체 기간 탐색 없음.

| site_id | exclusion_reason | buildings |
|---|---|---|
| Bobcat | continuous_zero_run_ge_30d | 1 |
| Bull | continuous_zero_run_ge_30d | 11 |
| Eagle | continuous_zero_run_ge_30d | 10 |
| Gator | continuous_zero_run_ge_30d | 4 |
| Hog | continuous_zero_run_ge_30d | 12 |
| Moose | continuous_zero_run_ge_30d | 3 |
| Mouse | continuous_zero_run_ge_30d | 7 |
| Shrew | continuous_zero_run_ge_30d | 9 |

## 핵심·부가 결과

- **H1: `supported`**. 28일 사이트 동일 가중 `ΔMAE_source` = **-16.482888133**, 95% site-cluster bootstrap CI **[-30.907569879, -6.56297796855]**.
- H2 효과크기 `F1 rate(28d)-F1 rate(1d)` = **-0.551526864698**, 95% CI **[-0.679617292789, -0.455237391104]**.
- H3 효과크기 `rate(A)-rate(B)` = **0.679884877048**, 95% CI **[0.583579951647, 0.783540480704]**.
- H2/H3 p-value·Holm 검정은 수행하지 않았다.

## 전체 0·1·3·7·14·28일 결과

| adaptation_days | mae_site_equal | rmse_site_equal | delta_mae_source_site_equal | delta_rmse_source_site_equal | a_rate_site_equal | b_rate_site_equal |
|---|---|---|---|---|---|---|
| 0 | 40.65 | 54.6597 | 0 | 0 | 0.0000 | 0.0682 |
| 1 | 38.4238 | 52.1976 | -2.22624 | -2.46208 | 0.3968 | 0.0267 |
| 3 | 34.9867 | 48.2985 | -5.66333 | -6.36123 | 0.6923 | 0.0756 |
| 7 | 29.6476 | 42.1623 | -11.0024 | -12.4975 | 0.8983 | 0.1587 |
| 14 | 27.4135 | 39.3044 | -13.2365 | -15.3554 | 0.9192 | 0.1993 |
| 28 | 24.1671 | 35.2364 | -16.4829 | -19.4233 | 0.9484 | 0.2685 |

## 단순 기준선

| model | mae_site_equal | rmse_site_equal |
|---|---|---|
| previous_day | 20.874 | 39.5247 |
| weekly_same_hour | 18.1359 | 30.0582 |

## F1~F4 실패

- 전체 비제로 건물×예산×시드 사건 6,795개 중 F1 1,508, F2 1,399, F3-PD 4,829, F3-W 4,505, F3-any 5,828, F3-both 3,506개.
- 건물×예산 2,265개 중 F4 448개.
- MAE 개선·RMSE 악화 불일치: 128개.

| adaptation_days | seed_event_denominator | f1_count | f2_count | f3_previous_day_count | f3_weekly_same_hour_count | f3_any_count | f3_both_count | f4_count |
|---|---|---|---|---|---|---|---|---|
| 1 | 1359 | 813 | 792 | 1211 | 1119 | 1316 | 1014 | 164 |
| 3 | 1359 | 377 | 350 | 1097 | 1000 | 1256 | 841 | 149 |
| 7 | 1359 | 134 | 101 | 921 | 892 | 1161 | 652 | 61 |
| 14 | 1359 | 104 | 87 | 852 | 803 | 1092 | 563 | 44 |
| 28 | 1359 | 80 | 69 | 748 | 691 | 1003 | 436 | 30 |
| ALL_NONZERO | 6795 | 1508 | 1399 | 4829 | 4505 | 5828 | 3506 | 448 |

## 데이터 누수·동일성 검사

- 모든 모델·예산·시드에서 건물별 `forecast_origin + target timestamp + horizon + y_true` digest가 동일했다.
- 적응 정답과 시험 정답 교집합은 Gate 1에서 0으로 확인했다.
- C0 입력은 직전 24시간 전력과 예측 대상의 결정적 달력 정보만 사용했다.
- Source 정규화 mean/std와 세 기존 checkpoint 해시는 실행 전 동결값과 일치했다.
- 독립 metric 재계산에서 예측 불가 값 0개, 비정상 status 0개였다.
- 특성 계산은 시험 시작 이전 관측까지만 사용했다.

## 고정 대표 실패 사례

| case_type | building_id | site_id | adaptation_days | seed | selection_value | selection_rule |
|---|---|---|---|---|---|---|
| repeated_f1 | Hog_office_Roger | Hog | 1 | 20260926 | 5.0 | most seed-mean F1 budgets; tie max mean delta then building_id |
| f4_seed_direction_mixed | Eagle_education_Brooke | Eagle | 14 | 20260924 | 226.53299509003176 | largest seed delta range; tie building_id then budget |
| mae_improved_rmse_worsened | Hog_office_Gustavo | Hog | 1 | 20260926 | 6.484847373417267 | largest RMSE worsening; tie building_id, budget, seed |

## 실행 정보

- Gate 2 장치: NVIDIA GeForce RTX 3080, 고정 세 시드.
- Gate 2 실행 시간: 2512.3초.
- Gate 3 분석 시간: 27.3초.
- 원본 prediction은 성능 확인 뒤 재생성하지 않았다.

## 한계

- 평가 가능한 사이트는 7개로 cluster bootstrap의 군집 수가 작다.
- final cohort는 고정 품질 기준을 통과한 건물에 한정되며, 30일 이상 연속 0값 건물과 적격 건물이 없는 2개 사이트는 포함하지 않는다.
- BDG2 개별 계량기의 물리적 범위가 건물마다 완전히 동일하다는 공식 근거는 부족하다.
- C0 LSTM과 head-only 적응만 검증했다. C1 기상, 실내 센서, 신규 Transformer, HVAC 제어는 검증하지 않았다.
- 특성 분석은 탐색적 상관이며 인과 추론이나 건물 제외 규칙으로 사용할 수 없다.
