# Figure 8 raw-prediction search report

## 권위 있는 원시 예측 파일

- 절대경로: `.\reports\final_v1_raw_predictions.csv`
- 형식: CSV, UTF-8 text, one header row
- 파일 크기: 1,216,425,185 bytes
- 수정 시각: 2026-09-25T12:35:30.188359+09:00 (local), 2026-09-25T03:35:30.188359+00:00 (UTC)
- 전체 데이터 행 수: 6,088,320
- 현재 파일 SHA-256: `72732c5dc0b0ad9cb7440f326476cd8c2ae57dd9267f7e07398452ae5b636e43`
- 기존 artifact manifest SHA-256: `72732c5dc0b0ad9cb7440f326476cd8c2ae57dd9267f7e07398452ae5b636e43`
- manifest 기록 크기: 1,216,425,185 bytes
- manifest 기록 수정 시각: 2026-09-25T03:35:30.188359+00:00
- 전체 열: `building_id, site_id, model, seed, adaptation_days, forecast_origin, input_start, input_end, target_timestamp, horizon_hour, y_true, y_pred, prediction_available, evaluation_mode, status, error`

CSV 자체에는 물리적 dtype metadata가 없다. 아래 자료형은 최종 producer 코드와 저장 텍스트를 함께 확인한 논리 자료형이다.

| 원본 열 | 논리 자료형 | 근거 |
|---|---|---|
| building_id | string | producer schema and stored text |
| site_id | string | producer schema and stored text |
| model | string | producer schema and stored text |
| seed | nullable integer | neural rows contain integer seed; baseline rows are blank |
| adaptation_days | nullable integer | neural rows contain integer budget; baseline rows are blank |
| forecast_origin | offset-free datetime text | producer writes timestamp text; no UTC offset field is stored |
| input_start | offset-free datetime text | producer writes timestamp text |
| input_end | offset-free datetime text | producer writes timestamp text |
| target_timestamp | offset-free datetime text | producer formats `%Y-%m-%d %H:%M:%S` |
| horizon_hour | integer | producer tiles 1 through 24 |
| y_true | float64 decimal text | producer flattens NumPy float arrays |
| y_pred | float64 decimal text | producer flattens NumPy float arrays |
| prediction_available | boolean | producer uses `np.isfinite(y_pred)` |
| evaluation_mode | string | `daily_nonoverlap` |
| status | string | execution status |
| error | string | blank on success |

## 최종 실행 결과로 판정한 근거

1. `reports/final_v1_final_test_report.md`가 Gate 2 raw prediction을 6,088,320행으로 기록하며, 독립 metric 재계산 최대 차이는 `1.71e-13`이다.
2. `reports/final_v1/GATE2_COMPLETE.json`이 완료 상태, 453개 건물, 오류 0개, 6,088,320행과 이 파일의 SHA-256을 기록한다.
3. `scripts/final_v1/run_final_v1_gate2.py`의 `FINAL_PREDICTIONS`가 이 경로를 지정하고, `prediction_frame`이 위 schema를 생성한다.
4. `scripts/final_v1/analyze_final_v1_gate3.py`가 이 파일의 `y_true`와 `y_pred`를 직접 읽어 독립 metric을 재계산한다.
5. 실제 스트리밍 행 수, 파일 크기, 현재 SHA-256이 완료 기록과 artifact manifest에 모두 일치한다.

## 열 대응

| 의미 | 원본 열 | 표준 출력 열 | 근거 |
|---|---|---|---|
| 건물 식별자 | `building_id` | `building_id` | producer schema |
| random seed | `seed` | `seed` | final neural execution loops over fixed seeds |
| adaptation budget | `adaptation_days` | `adaptation_days` | Gate 2 fixed budget field |
| forecast origin | `forecast_origin` | `forecast_origin` | `prediction_frame` metadata |
| target timestamp | `target_timestamp` | `target_timestamp` | `prediction_frame` hourly target timestamps |
| horizon | `horizon_hour` | `horizon` | producer explicitly tiles 1–24 |
| observed value | `y_true` | `y_true` | metric code parses this field as float |
| prediction | `y_pred` | `y_pred` | metric code parses this field as float |

`y_true`와 `y_pred`는 `prediction_frame`에서 실제 평가 배열을 직접 펼친 값이고, Gate 3 metric 코드도 같은 두 열을 float64로 읽는다. 추출 CSV에는 원본 숫자 문자열을 반올림하거나 재서식하지 않고 그대로 기록했다.

원본 timestamp에는 UTC offset 또는 timezone 이름이 저장되어 있지 않다. 따라서 추출 파일은 offset을 발명하지 않고 구분자만 `T`로 바꾼 offset-free ISO 8601 형식으로 보존했다.

## 제외한 후보

| 절대경로 | 크기(bytes) | 제외 근거 |
|---|---:|---|
| `.\reports\bdg2_stage6_c0_adaptation_predictions_v3.csv` | 27,977,967 | Stage-6 development adaptation output; not final_v1 |
| `.\reports\bdg2_stage6_c0_source_only_predictions_v3.csv` | 4,473,559 | Stage-6 development source-only output; not final_v1 |
| `.\reports\bdg2_multibuilding_adaptation_predictions_v2.csv` | 27,542,236 | v2 development output; not final_v1 |
| `.\reports\bdg2_multibuilding_source_only_predictions_v2.csv` | 4,400,802 | v2 development output; not final_v1 |
| `.\reports\bdg2_adaptation_predictions_20260924.csv` | 3,582,246 | four-building preliminary adaptation output |
| `.\reports\bdg2_source_only_predictions_20260924.csv` | 565,529 | four-building preliminary source-only output |
| `.\reports\final_v1\final_v1_representative_failure_predictions.csv` | 825,412 | derived representative-case subset; not the full raw table |

프로젝트 재귀 검색에서 `Figure_Table_Package_v1.zip`과 `code/make_figure8_from_predictions.py`는 발견되지 않았다. 프로젝트의 ZIP 파일은 Edge Deployment 전송 번들이며 Figure 8 생성 코드가 아니다.
