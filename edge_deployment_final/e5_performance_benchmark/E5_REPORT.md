# E5 — Edge Performance Benchmark Report

## 최종 판정

**E5 STATUS: PASS**

동결된 C0 LSTM FP32 모델을 Jetson Orin Nano와 RTX 3080에서 동일한 batch-1 benchmark vector로 실행했다. 양 장치 모두 사전·사후 수치 일관성 gate를 통과했고, 장치별 두 latency mode에서 각각 5,000개의 관측값을 완료했다. 모델 학습, 적응, 가중치 변경, FP16, ONNX, TensorRT, `torch.compile`은 수행하지 않았다.

RTX process RSS 수집은 Windows API collector 오류로 실패했으나 지연시간, CUDA 메모리, telemetry 및 수치 일관성에는 영향을 주지 않았다. 최초 실행 코드와 결과를 보존하고 별도 addendum에 원인과 수정사항을 기록했다.

## 모델 및 동결 조건

| 항목 | 값 |
|---|---|
| Model | C0 LSTM |
| Seed | 20260924 |
| Parameters | 5,912 |
| Precision | FP32 |
| Batch | 1 |
| Input | `(1, 24, 6)` |
| Output | `(1, 24)` |
| Checkpoint SHA-256 | `88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b` |
| `state_dict` | `strict=True`, missing 0, unexpected 0 |
| Inference | `model.eval()`, `torch.no_grad()` |
| Model/checkpoint modified | NO |

고정 프로토콜은 실제 성능값 확인 전에 `protocol/E5_FROZEN_PROTOCOL.md`에 저장했다. 각 latency mode는 warm-up 200회 후 5 runs × 1,000회로 측정했으며 outlier를 삭제하지 않았다.

## 환경

### Jetson Orin Nano

| 항목 | 값 |
|---|---|
| Host | `JETSON_HOST`, user `han` |
| Device | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| L4T | R36.4.3 |
| Python | 3.10.12 |
| PyTorch | 2.11.0 |
| PyTorch CUDA | 12.6 |
| NumPy | 1.26.4 |
| Power mode | 25W, `nvpmodel -q` 확인 |
| `jetson_clocks` | 상태 확인 불가: `--show`가 root를 요구함. 새로 활성화하거나 변경하지 않음 |
| Available RAM at environment capture | 6,627,717,120 bytes |
| Initial runtime GPU temperature | 49.406°C |

### RTX 3080 host

| 항목 | 값 |
|---|---|
| OS | Windows 11, build 26200 |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| Driver | 591.86 |
| Driver-reported CUDA | 13.1 |
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA | 12.8 |
| NumPy | 2.3.5 |
| Initial runtime GPU temperature | 62°C |
| Initial runtime GPU power | 122.96 W |

측정 직전 `nvidia-smi` process table에는 compute-only `C` context가 없고 Windows 화면용 `C+G` context만 있었다. 그러나 WDDM 데스크톱 환경의 GUI 부하는 유지됐으며, 높은 idle GPU utilization과 전력 변동이 RTX 전력 및 tail latency의 제한 사항이다.

## 주요 latency 및 CUDA memory 비교

단위는 latency `ms`, memory `MiB`이다. Memory는 `torch.cuda.reset_peak_memory_stats()` 이후 두 latency mode 전체의 peak다.

| Device | Runtime | Precision | Batch | Parameters | Mean model-only | Median model-only | P95 model-only | P99 model-only | Mean end-to-end | Median end-to-end | P95 end-to-end | P99 end-to-end | Peak CUDA allocated | Peak CUDA reserved |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jetson Orin Nano | PyTorch 2.11.0 / CUDA 12.6 | FP32 | 1 | 5,912 | 0.690587 | 0.675552 | 0.731138 | 0.960481 | 0.932956 | 0.883082 | 0.937253 | 0.976195 | 17.24 | 22.00 |
| RTX 3080 | PyTorch 2.11.0+cu128 / CUDA 12.8 | FP32 | 1 | 5,912 | 0.205226 | 0.171072 | 0.354354 | 0.885665 | 0.337380 | 0.232600 | 0.912830 | 2.373438 | 17.24 | 22.00 |

- Jetson/RTX mean latency ratio: model-only **3.3650**, end-to-end **2.7653**
- Cross-device power ratio: 계산하지 않음
- Raw observations: 장치별 model-only 5,000, end-to-end 5,000
- 독립 raw CSV 재계산과 저장 통계의 최대 차이: 0.0

## 전체 latency 분포

| Device | Mode | Mean | Std | Median | P90 | P95 | P99 | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jetson | Model-only | 0.690587 | 0.080474 | 0.675552 | 0.715875 | 0.731138 | 0.960481 | 0.393472 | 4.994784 |
| Jetson | End-to-end | 0.932956 | 2.758206 | 0.883082 | 0.927259 | 0.937253 | 0.976195 | 0.841401 | 195.884742 |
| RTX 3080 | Model-only | 0.205226 | 0.121196 | 0.171072 | 0.243907 | 0.354354 | 0.885665 | 0.068608 | 1.303552 |
| RTX 3080 | End-to-end | 0.337380 | 0.374808 | 0.232600 | 0.377610 | 0.912830 | 2.373438 | 0.218300 | 4.413000 |

Jetson end-to-end run 5에 195.884742 ms 관측값이 한 번 포함되어 mean과 standard deviation이 증가했다. 사전 규칙에 따라 삭제하거나 재측정하지 않았다. Median, P95, P99와 maximum을 함께 보고한다.

## Run별 latency — Jetson

| Mode | Run | Mean | Std | Median | P90 | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model-only | 1 | 0.730606 | 0.168721 | 0.682464 | 0.955046 | 0.960480 | 0.975460 | 0.393472 | 4.994784 |
| Model-only | 2 | 0.681297 | 0.020170 | 0.675088 | 0.711648 | 0.721858 | 0.738541 | 0.639584 | 0.836928 |
| Model-only | 3 | 0.681780 | 0.026791 | 0.675584 | 0.711866 | 0.719554 | 0.732749 | 0.634272 | 1.287296 |
| Model-only | 4 | 0.678660 | 0.018401 | 0.673488 | 0.707229 | 0.714752 | 0.738982 | 0.642496 | 0.773568 |
| Model-only | 5 | 0.680590 | 0.021034 | 0.674528 | 0.711146 | 0.719043 | 0.747385 | 0.646240 | 0.866656 |
| End-to-end | 1 | 0.890922 | 0.021944 | 0.884090 | 0.923425 | 0.933054 | 0.953020 | 0.845977 | 1.010493 |
| End-to-end | 2 | 0.893200 | 0.080710 | 0.882634 | 0.925086 | 0.933790 | 0.968600 | 0.847641 | 3.325057 |
| End-to-end | 3 | 0.895157 | 0.071042 | 0.882106 | 0.930342 | 0.940769 | 0.980740 | 0.841401 | 2.867764 |
| End-to-end | 4 | 0.890377 | 0.066931 | 0.881226 | 0.922894 | 0.931008 | 0.958734 | 0.844280 | 2.879860 |
| End-to-end | 5 | 1.095126 | 6.163529 | 0.885418 | 0.934565 | 0.946153 | 0.998202 | 0.853657 | 195.884742 |

## Run별 latency — RTX 3080

| Mode | Run | Mean | Std | Median | P90 | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model-only | 1 | 0.201781 | 0.105084 | 0.172032 | 0.237482 | 0.336427 | 0.692634 | 0.069632 | 1.243136 |
| Model-only | 2 | 0.212133 | 0.117919 | 0.174016 | 0.277558 | 0.392448 | 0.844921 | 0.070656 | 1.303552 |
| Model-only | 3 | 0.206249 | 0.130694 | 0.172032 | 0.235523 | 0.329478 | 0.916675 | 0.068608 | 1.243136 |
| Model-only | 4 | 0.201746 | 0.129679 | 0.169984 | 0.220278 | 0.330130 | 0.973116 | 0.071680 | 1.227488 |
| Model-only | 5 | 0.204224 | 0.120512 | 0.170224 | 0.245834 | 0.329832 | 0.861347 | 0.071680 | 1.222656 |
| End-to-end | 1 | 0.273171 | 0.152960 | 0.223800 | 0.331420 | 0.572220 | 1.020718 | 0.218300 | 1.351600 |
| End-to-end | 2 | 0.400316 | 0.489213 | 0.236450 | 0.613650 | 1.447840 | 2.680274 | 0.220700 | 3.417500 |
| End-to-end | 3 | 0.306894 | 0.249665 | 0.232350 | 0.343610 | 0.741910 | 1.629787 | 0.220100 | 1.898600 |
| End-to-end | 4 | 0.386228 | 0.539522 | 0.233700 | 0.382650 | 1.564830 | 3.245411 | 0.220900 | 4.413000 |
| End-to-end | 5 | 0.320290 | 0.273078 | 0.234400 | 0.367650 | 0.853585 | 1.794797 | 0.223600 | 1.984700 |

## CUDA 및 process memory

| Device | Current CUDA allocated | Current CUDA reserved | Peak CUDA allocated | Peak CUDA reserved | Process current RSS | Process peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Jetson | 9,596,928 B | 23,068,672 B | 18,076,672 B | 23,068,672 B | 585,945,088 B | 585,945,088 B |
| RTX 3080 | 9,596,928 B | 23,068,672 B | 18,076,672 B | 23,068,672 B | 측정 불가 | 측정 불가 |

RTX process RSS 누락은 `rtx3080/PROCESS_RAM_COLLECTION_ADDENDUM.md`에 기록했다. 수정 후 별도 API probe 값은 benchmark 결과에 대입하지 않았다.

## Sustained power 및 thermal

| Device | Power measurement source | Idle reported power | Sustained reported power | Peak reported power | Change from idle | Initial GPU temp | Sustained mean GPU temp | Peak GPU temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Jetson Orin Nano | `tegrastats` VDD_IN input rail | 3.146 W | 4.031 W | 4.032 W | +0.884 W | 49.750°C | 50.329°C | 50.843°C |
| RTX 3080 | `nvidia-smi` GPU `power.draw` | 63.373 W | 70.422 W | 123.660 W | +7.049 W | 61.000°C | 60.600°C | 63.000°C |

Jetson VDD_IN과 RTX GPU board power는 측정 경계가 다르다. 두 값을 이용한 cross-device power ratio 또는 energy-per-inference ratio를 계산하지 않았다.

- Jetson samples: idle 59, active 59, cooldown 59
- RTX samples: idle 60, active 60, cooldown 60
- Sustained workload completed iterations: Jetson 91,022, RTX 214,724
- 위 반복 수는 전력·온도 관찰용 controlled workload이며 forecasting throughput 결과가 아니다.
- Raw telemetry에서 thermal shutdown, throttling 또는 overheat 경고 문자열은 발견되지 않았다. Raw field가 모든 hardware throttling 상태를 제공하는 것은 아니므로 경고가 전혀 발생하지 않았다는 포괄적 증명으로 해석하지 않는다.
- RTX active 평균 온도가 idle 평균보다 0.333°C 낮은 값은 WDDM desktop 부하와 냉각 상태가 변동한 실제 관측 결과이며 보정하지 않았다.

## 사전·사후 numerical consistency

| Device | Gate | Normalized max abs diff | Normalized mean abs diff | Physical max abs diff | Physical mean abs diff | NaN | Inf | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Jetson | Pre | 1.2814998627e-06 | 6.9166223208e-07 | 0.0004540721 | 0.0002450758 | 0 | 0 | PASS |
| Jetson | Post | 1.2814998627e-06 | 6.9166223208e-07 | 0.0004540721 | 0.0002450758 | 0 | 0 | PASS |
| RTX 3080 | Pre | 5.9604644775e-08 | 2.4835268656e-08 | 0.0000211196 | 0.0000087998 | 0 | 0 | PASS |
| RTX 3080 | Post | 5.9604644775e-08 | 2.4835268656e-08 | 0.0000211196 | 0.0000087998 | 0 | 0 | PASS |

Shape는 모두 `(1,24)`로 정확히 일치했다. E4의 사전 고정 normalized 및 physical 기준을 모두 통과했다.

## Operational interpretation

주 연구 workload는 과거 24시간 전력과 target calendar feature로 향후 24시간을 예측하는 batch-1 호출이다. 측정된 median 및 P95 latency는 양 장치 모두 1 ms 이내였고, 보존한 최대 outlier를 포함해도 24시간 예측 운영 간격보다 충분히 작다. 이 결과는 해당 고정 모델 호출의 시간 규모만 설명하며, 일반적인 실시간 시스템 적합성이나 전체 애플리케이션 지연을 뜻하지 않는다.

## 제한 사항 및 미해결 항목

1. RTX process RSS/peak RSS는 collector 오류로 측정 불가다. 실행 원본과 수정본을 모두 보존했고 benchmark는 반복하지 않았다.
2. 현재 Jetson image에서 `jetson_clocks --show`가 root를 요구해 상태를 직접 확인하지 못했다. E5에서 활성화하거나 변경하지 않았다.
3. RTX는 WDDM desktop의 다수 `C+G` context와 높은 idle GPU utilization이 존재했다. compute-only context는 없었으나 RTX 전력 및 tail latency는 전용 무부하 서버 조건이 아니다.
4. Jetson과 RTX의 PyTorch CUDA 버전과 운영체제가 다르다.
5. Power rail 경계가 다르므로 장치 간 전력 효율 우열을 판단하지 않았다.
6. 단일 소형 모델, 단일 입력 vector, FP32, batch 1 결과이며 다른 모델·precision·batch에 일반화하지 않는다.

## 실행 및 재현 자료

- 실제 실행 코드: `benchmark_fp32_executed_v1.py`, SHA-256 `6d069a5411fcc98c3557c36e6aabe96df13b414af217c9eac55f91f188d0d118`
- Process RAM collector 수정본: `benchmark_fp32.py`, SHA-256 `b3f6b44935e3a5e0ed32111a174fb076e9dda8ec5e13aa835f28a793445a41d`
- 실행 오케스트레이터: `run_e5_benchmark.py`
- 독립 분석: `analyze_e5_results.py`
- 실행 단계: 30, 모두 return code 0
- 기록된 실행 단계 총 시간: 451.457초
- E6 TensorRT/FP16: 시작하지 않음

## 완료 상태

- Pre-benchmark numerical consistency: **PASS**
- Post-benchmark numerical consistency: **PASS**
- NaN: **0**
- Inf: **0**
- Artifacts SHA-256 manifest: `artifact_manifest.csv`
- E5 COMPLETE: **YES**

