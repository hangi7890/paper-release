# final_v1 건물 특성 탐색 분석

## 범위와 정보 시점

- 모든 결과는 **탐색적**이며 건물 제외, 모델 선택, H1~H3 판정에 사용하지 않았다.
- 0일은 첫 예측 직전 24시간 context만 사용했고 장기 Target 통계는 `NA`로 유지했다.
- 비제로 예산은 초기 적응 직전 24시간과 허용된 적응 관측 종료까지만 사용했다.
- 모든 `feature_information_cutoff`는 시험 시작보다 이르다. 시험 정답 실제값은 특성 계산에 사용하지 않았다.
- 일간 반복성은 최소 24개 lag pair, 주간 반복성은 최소 168개 lag pair가 있고 양쪽이 비상수일 때만 Pearson 상관을 계산했다.
- Source 규모 기준은 기존 Source 20개 학습 구간 표를 그대로 사용했다.

## 자료 규모

- 특성 행: 2718 = 453개 건물 × 6개 예산.
- 사이트: 7개; 공식 용도값이 있는 범주 13개; 용도 미기재 건물 18개는 `<missing>` 범주로 보존했다.
- 적응 기반 장기 특성을 사용할 수 있는 행: 2265개.
- 주간 반복성 정의 가능 행: 905개.

## 예산별 `ΔMAE_source`와 절댓값 상관이 큰 탐색 항목

| adaptation_days | feature | n_complete | spearman_rho | status |
|---|---|---|---|---|
| 1 | daily_repeatability_pearson | 389 | -0.4063 | exploratory_descriptive |
| 1 | context_24h_absolute_mean_gap_in_source_config_std | 453 | -0.3221 | exploratory_descriptive |
| 1 | absolute_mean_gap_in_source_config_std | 453 | -0.3194 | exploratory_descriptive |
| 3 | absolute_mean_gap_in_source_config_std | 453 | -0.4128 | exploratory_descriptive |
| 3 | context_24h_absolute_mean_gap_in_source_config_std | 453 | -0.3891 | exploratory_descriptive |
| 3 | available_cv_abs_mean | 453 | 0.2979 | exploratory_descriptive |
| 7 | available_cv_abs_mean | 453 | 0.3185 | exploratory_descriptive |
| 7 | absolute_mean_gap_in_source_config_std | 453 | -0.2990 | exploratory_descriptive |
| 7 | context_24h_absolute_mean_gap_in_source_config_std | 453 | -0.2837 | exploratory_descriptive |
| 14 | context_24h_mean | 453 | -0.3225 | exploratory_descriptive |
| 14 | mean_percentile_among_20_source_building_means | 453 | -0.3186 | exploratory_descriptive |
| 14 | available_mean | 453 | -0.3153 | exploratory_descriptive |
| 28 | mean_percentile_among_20_source_building_means | 453 | -0.3513 | exploratory_descriptive |
| 28 | available_mean | 453 | -0.3510 | exploratory_descriptive |
| 28 | context_24h_mean | 453 | -0.3509 | exploratory_descriptive |

이 표는 작은 수의 독립 사이트와 반복 예산 관측에서 얻은 탐색적 상관이다. 인과관계, 사전 실패 판별 규칙, 전체 BDG2의 일반 법칙으로 해석하지 않는다. 다중 탐색에 대한 확증적 유의성 검정을 수행하지 않았다.

## Software addendum 001

최초 Gate 3 탐색 코드가 `primaryspaceusage` 결측 18개(Swan 사이트)를 pandas `groupby` 기본 동작으로 제외했다. 예측·metric·F1~F4·H1~H3에는 영향이 없었다. 최초 파일은 `reports/final_v1/addendum_pre_characteristic_fix/`에 보존했고, 결측을 임의 용도로 대체하지 않고 `<missing>`이라는 명시적 범주로 유지해 이 보고서와 탐색 표만 수정했다. 자세한 해시는 `reports/final_v1_software_addendum_001.md`에 있다.
