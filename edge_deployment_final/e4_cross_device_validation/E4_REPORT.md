# E4 — Canonical Model Transfer & Cross-Device Numerical Consistency Test

## 판정

**E4 STATUS: PASS**

동결된 C0 LSTM 체크포인트를 Jetson Orin Nano로 전송했고, Jetson CUDA FP32 출력은 E2에서 결과 확인 전에 고정한 수치 일관성 허용 기준 안에서 RTX 3080 기준 출력을 재현했다.

이 판정은 **FP32 교차 장치 수치 일관성**에만 해당한다. 지연시간, 처리량, 전력, 에너지, 온도, 메모리 성능은 측정하지 않았으며 실시간 적합성, 저전력 우수성, 예측 정확도 향상 또는 TensorRT 준비 상태를 뜻하지 않는다.

## 실행 범위와 경로

- 실행일: 2026-09-26 (Asia/Seoul)
- Windows 프로젝트: `research_stage1`
- E2 원본 압축파일: `edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32.zip`
- Jetson 접속 대상: `USER@JETSON_HOST`
- Jetson 전송 위치: `/home/USER/research/edge_deployment/edge_deployment_jetson_fp32.zip`
- Jetson 압축 해제 위치: `/home/USER/research/edge_deployment/e4_runtime/edge_deployment_jetson_fp32`
- Jetson 추론 출력 위치: `/home/USER/research/edge_deployment/e4_outputs`
- Jetson 가상환경 Python: `/home/USER/venvs/edge_c0_fp32/bin/python`
- 성능 벤치마크 수행: 아니요
- 학습 또는 적응 수행: 아니요
- ONNX, TensorRT, FP16, INT8 사용: 아니요

세부 명령과 반환 코드는 `e4_command_log.txt`에 보존했다.

## Jetson 런타임

| 항목 | 확인값 |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.11.0 |
| PyTorch CUDA | 12.6 |
| NumPy | 1.26.4 |
| CUDA available | `True` |
| CUDA device count | 1 |
| CUDA device | Orin |
| Compute capability | 8.7 |

원문 실행 결과는 `jetson_runtime_info.txt`에 기록했다.

## 전송 및 번들 무결성

| 검사 | 기대값 | 확인값 | 결과 |
|---|---:|---:|---|
| 압축파일 크기 | 61,171 bytes | 61,171 bytes | PASS |
| 압축파일 SHA-256 | `479b67791b6770e41ddecb88e72b28d7ebeecaadfa67196e48621778b7a6d4d8` | 동일 | PASS |
| 번들 manifest 상태 | `FROZEN` | `FROZEN` | PASS |
| manifest 파일 수 | 20 | 20 | PASS |
| 파일별 크기·SHA-256 실패 | 0 | 0 | PASS |
| payload digest | `9cc1aaec5b7198199e491c6efe2616137e4f4bf4152f30b5b8d46ccd5c878880` | 동일 | PASS |
| 체크포인트 SHA-256 | `88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b` | 동일 | PASS |

원격 압축파일은 크기와 SHA-256이 일치한 뒤에만 새 `e4_runtime` 디렉터리에 해제했다. 원격 번들 파일 20개는 E2 `TRANSFER_MANIFEST.json`의 크기와 SHA-256에 모두 일치했다.

## 모델 로드 및 실행 조건

| 항목 | 확인값 | 결과 |
|---|---:|---|
| 모델 | C0 LSTM, seed 20260924 | PASS |
| `state_dict` | `strict=True` | PASS |
| missing keys | 0 | PASS |
| unexpected keys | 0 | PASS |
| 파라미터 수 | 5,912 | PASS |
| 입력 차원 | 6 | PASS |
| 입력 시퀀스 길이 | 24 | PASS |
| 출력 horizon | 24 | PASS |
| 파라미터 dtype | `torch.float32` | PASS |
| 평가 모드 | `model.eval()` | PASS |
| gradient 비활성화 | `torch.no_grad()` | PASS |
| autocast/FP16 | 사용 안 함 | PASS |
| TF32 설정 강제 변경 | 없음 | PASS |
| 모델·가중치 수정 | 없음 | PASS |

## CPU benchmark vector sanity test

`vector_id=0`, 입력 shape `(1, 24, 6)`을 Jetson CPU에서 실행했다. 출력 shape는 `(1, 24)`였고 NaN과 Inf는 없었다.

| 스케일 | max abs diff | mean abs diff | RMSE diff |
|---|---:|---:|---:|
| Normalized | 1.3113021850585938e-06 | 6.755193074544271e-07 | 7.740002222788663e-07 |
| Physical | 0.00046463195864987483 | 0.00023935585748032415 | 0.0002742504690085579 |

**CPU benchmark-vector: PASS**

## CUDA benchmark vector sanity test

동일한 `vector_id=0`을 Jetson CUDA FP32에서 batch size 1로 실행했다. 출력 shape는 `(1, 24)`였고 NaN과 Inf는 없었다. 사전 고정 기준을 명시적 assertion으로 적용한 benchmark gate를 통과한 뒤에만 64개 벡터 추론을 진행했다.

| 스케일 | max abs diff | mean abs diff | RMSE diff |
|---|---:|---:|---:|
| Normalized | 1.2814998626708984e-06 | 6.916622320810953e-07 | 7.867803703047205e-07 |
| Physical | 0.0004540721413945903 | 0.00024507575848590574 | 0.0002787788418554443 |

**CUDA benchmark-vector: PASS**

## 64 validation vectors CUDA FP32 결과

- 입력 shape: `(64, 24, 6)`
- 기대 출력 shape: `(64, 24)`
- 실제 출력 shape: `(64, 24)`
- 비교한 출력 원소: 1,536
- NaN: 0
- Inf: 0
- 독립 재계산: Windows에 회수한 `.npy`와 동결된 RTX reference `.npy`를 `analyze_e4_outputs.py`로 다시 비교

| 스케일 | max abs diff | mean abs diff | RMSE diff |
|---|---:|---:|---:|
| Normalized | 1.4781951904296875e-05 | 1.4946805701280634e-06 | 2.39523791840085e-06 |
| Physical | 0.0052376693520272966 | 0.0005296081778475107 | 0.0008487014649326075 |

**64-vector CUDA inference: PASS**

## 최대 오차 원소

인덱스는 0부터 시작한다.

| 항목 | Normalized | Physical |
|---|---:|---:|
| vector_id | 51 | 51 |
| horizon index | 16 | 16 |
| RTX reference | 6.002083778381348 | 2293.775557483601 |
| Jetson | 6.002098560333252 | 2293.780795152953 |
| absolute difference | 1.4781951904296875e-05 | 0.0052376693520272966 |

## 사전 고정 허용 기준 판정

| 조건 | 기준 | 관측값 | 판정 |
|---|---:|---:|---|
| Shape | exact match | `(64,24)` = `(64,24)` | PASS |
| NaN | 0 | 0 | PASS |
| Inf | 0 | 0 | PASS |
| Normalized max abs diff | ≤ 1e-4 | 1.4781951904296875e-05 | PASS |
| Normalized mean abs diff | ≤ 2e-5 | 1.4946805701280634e-06 | PASS |
| Physical max abs diff | ≤ 0.05 | 0.0052376693520272966 | PASS |
| Physical mean abs diff | ≤ 0.01 | 0.0005296081778475107 | PASS |

**Frozen acceptance criteria: PASS**

번들에 포함된 `verify_output.py`의 JSON에는 `tolerance_enforced=false`라는 E1 설명 필드가 남아 있다. E4 판정에서는 이를 허용 기준 판정으로 사용하지 않았다. CUDA benchmark gate가 사전 기준을 명시적으로 검사했고, 회수한 64개 출력에는 별도 코드 `analyze_e4_outputs.py`가 같은 고정 기준을 독립 적용했다. 기준값은 결과를 확인한 뒤 변경하지 않았다.

## 재현 자료

- `run_e4_transfer_validation.py`: 전송, 원격 해시 검사, 압축 해제, 번들 무결성 검사, 모델 감사, CPU·CUDA 추론, 출력 회수
- `analyze_e4_outputs.py`: 회수 출력과 동결 RTX reference의 독립 비교 및 고정 기준 판정
- `benchmark_vector_comparison.csv`: benchmark vector의 24개 원소 비교
- `validation_64_comparison_summary.json`: 1,536개 출력의 요약 통계와 개별 기준 판정
- `max_error_case.json`: 최대 오차 위치와 값
- `jetson_validation_output_normalized.npy`, `jetson_validation_output_physical.npy`: Jetson CUDA 결과
- `artifact_manifest.csv`: E4 디렉터리 산출물의 파일 크기와 SHA-256. manifest 자신은 순환 해시를 피하기 위해 목록에서 제외한다.

## 최종 해석

> The frozen C0 LSTM checkpoint was transferred to the Jetson Orin Nano and reproduced the RTX reference predictions within the pre-specified FP32 numerical consistency tolerance.

- Model/checkpoint modified: **NO**
- Performance benchmark performed: **NO**
- E4 READY FOR E5: **YES**
- E5 started: **NO**

