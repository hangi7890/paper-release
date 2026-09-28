# Figure 8 extraction validation report

## 입력과 방법

- 원시 입력: `.\reports\final_v1_raw_predictions.csv`
- 원시 입력 SHA-256: `72732c5dc0b0ad9cb7440f326476cd8c2ae57dd9267f7e07398452ae5b636e43`
- 추출 방식: pandas chunked CSV scan (`chunksize=250000`, selected columns only)
- 모델 학습·적응·추론·prediction 재생성: 수행하지 않음
- metric: 저장된 672개 `y_true`, `y_pred`에서 `MAE = mean(abs(y_pred-y_true))`, `RMSE = sqrt(mean((y_pred-y_true)^2))`로 독립 재계산

## 검증 결과

| 검사 | 결과 |
|---|---|
| total_rows_6720 | PASS |
| ten_cases_present | PASS |
| each_case_672_rows | PASS |
| each_case_28_origins | PASS |
| each_origin_horizons_1_24 | PASS |
| source_adapted_axes_and_y_true_exact | PASS |
| finite_y_true_y_pred | PASS |
| no_duplicate_key | PASS |
| raw_status_ok | PASS |
| prediction_available_true | PASS |
| roger_delta_mae_positive | PASS |
| brooke_mixed_delta_mae_directions | PASS |
| brooke_delta_mae_range_matches | PASS |
| gustavo_delta_mae_negative | PASS |
| gustavo_delta_rmse_matches | PASS |
| all_case_metric_status_pass | PASS |
| all_validation_checks_pass | PASS |

## 사례별 metric

| building_id | seed | days | MAE | RMSE | ΔMAE | ΔRMSE | rows | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hog_office_Roger | 20260926 | 0 | 14.8095951143863 | 24.5989407896069 | 0 | 0 | 672 | PASS |
| Hog_office_Roger | 20260926 | 1 | 61.0981266752253 | 66.8269919057528 | 46.2885315608389 | 42.2280511161459 | 672 | PASS |
| Eagle_education_Brooke | 20260924 | 0 | 187.584873801124 | 232.530755088807 | 0 | 0 | 672 | PASS |
| Eagle_education_Brooke | 20260924 | 14 | 198.415448488538 | 225.051551665825 | 10.830574687414 | -7.47920342298164 | 672 | PASS |
| Eagle_education_Brooke | 20260925 | 0 | 131.920074213226 | 167.801571778849 | 0 | 0 | 672 | PASS |
| Eagle_education_Brooke | 20260925 | 14 | 94.6816611432406 | 135.669833039292 | -37.2384130699849 | -32.131738739557 | 672 | PASS |
| Eagle_education_Brooke | 20260926 | 0 | 388.389170668556 | 476.764106063401 | 0 | 0 | 672 | PASS |
| Eagle_education_Brooke | 20260926 | 14 | 172.686750265938 | 224.591103540604 | -215.702420402618 | -252.173002522797 | 672 | PASS |
| Hog_office_Gustavo | 20260926 | 0 | 208.288076049696 | 266.737023845396 | 0 | 0 | 672 | PASS |
| Hog_office_Gustavo | 20260926 | 1 | 174.181016186378 | 273.221871218814 | -34.1070598633177 | 6.48484737341727 | 672 | PASS |

ΔMAE와 ΔRMSE는 `adapted - source-only`이다. 0일 행의 delta는 0이다.

## 고정 결과 재현

- `Hog_office_Roger`, seed 20260926, 1일: ΔMAE = `46.28853156083894` (> 0), ΔRMSE = `42.22805111614588`.
- `Eagle_education_Brooke`, 14일 ΔMAE: seed 20260924 = `10.830574687413957`, 20260925 = `-37.238413069984915`, 20260926 = `-215.70242040261792`. 개선·악화 방향이 모두 존재한다.
- Brooke 최대–최소 ΔMAE = `226.53299509003187`; 기준값 `226.53299509003176`과 절대오차 1e-9 이내로 일치한다.
- `Hog_office_Gustavo`, seed 20260926, 1일: ΔMAE = `-34.107059863317716` (< 0), ΔRMSE = `6.484847373417267`; 기준값 `6.484847373417267`과 절대오차 1e-9 이내로 일치한다.

## Figure 생성 여부

프로젝트에서 지정된 `Figure_Table_Package_v1.zip` 또는 `code/make_figure8_from_predictions.py`를 찾지 못했다. 요청 조건에 따라 검증된 CSV와 metric 보고서까지만 생성했으며 Figure 시계열이나 seed 값을 새로 만들지 않았다.
