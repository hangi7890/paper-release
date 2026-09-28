# E5 Frozen Protocol — Edge Performance Benchmark

## 동결 시점과 목적

- 동결일: 2026-09-26 (Asia/Seoul)
- 동결 시점: E5 지연시간·전력·온도 결과 확인 전
- 목적: 동일한 canonical C0 LSTM FP32 모델의 Jetson Orin Nano와 RTX 3080 실행 특성을 측정한다.
- 제외: 예측 정확도 재평가, 학습, 적응, 가중치 변경, 모델 선택, TensorRT, ONNX, FP16, INT8, `torch.compile`, pruning, quantization

## 동결 모델과 입력

- 모델: C0 LSTM
- seed: 20260924
- parameter count: 5,912
- checkpoint SHA-256: `88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b`
- archive SHA-256: `479b67791b6770e41ddecb88e72b28d7ebeecaadfa67196e48621778b7a6d4d8`
- precision: FP32
- batch size: 1
- benchmark vector: E1/E2의 `vector_id=0`
- input shape: `(1, 24, 6)`
- output shape: `(1, 24)`
- 실행: `model.eval()`, `torch.no_grad()`
- autocast: 사용하지 않음
- TF32: 런타임 기본값을 강제로 변경하지 않음
- 파일 I/O, Python 시작, checkpoint 로드는 latency에서 제외

## Gate E5-0

두 장치에서 성능 측정 전에 다음을 검사한다.

1. checkpoint SHA-256 일치
2. parameter count 5,912
3. 모델 parameter dtype FP32
4. 입력 `(1,24,6)`, 출력 `(1,24)`
5. eval mode 및 no-grad 추론
6. NaN 0, Inf 0
7. E4 RTX reference와 normalized/physical 출력 비교
8. E4 동결 기준 통과

어느 장치든 실패하면 성능 측정을 시작하지 않는다.

## 지연시간 프로토콜

각 장치에서 두 모드를 각각 측정한다.

### Model-only

- 입력 tensor는 CUDA에 미리 존재한다.
- model forward만 포함하고 출력은 CUDA에 둔다.
- CUDA event로 각 inference를 측정한다.
- 각 관측값 종료 event를 synchronize한다.

### Deployment end-to-end

- CPU tensor는 RAM에 미리 존재한다.
- CPU→CUDA 복사, model forward, CUDA→CPU 복사를 포함한다.
- 디스크 I/O는 포함하지 않는다.
- 각 관측값 시작 전과 출력 복사 후 `torch.cuda.synchronize()`를 적용한다.
- `time.perf_counter_ns()`로 wall-clock 시간을 측정한다.

### 반복 수

- 각 모드 시작 전 warm-up: 200회
- 독립 run: 5회
- run별 measured inference: 1,000회
- 모드별 총 관측값: 5,000개
- run 사이 안정화 시간: 1초
- warm-up은 raw 결과와 통계에서 제외한다.
- outlier는 삭제하지 않는다.

### 통계

- 단위: millisecond
- mean
- population standard deviation (`ddof=0`)
- median
- P90, P95, P99: NumPy 선형 보간 percentile
- minimum, maximum
- 각 run과 전체 5,000개를 모두 요약한다.

## 메모리 프로토콜

모델을 CUDA에 로드한 뒤 latency warm-up 전에 `torch.cuda.reset_peak_memory_stats()`를 한 번 호출한다. 두 latency 모드가 끝난 뒤 다음을 기록한다.

- current allocated
- current reserved
- peak allocated
- peak reserved
- process current RSS
- process peak RSS 또는 운영체제가 제공하는 최대 working set

추가 dependency는 설치하지 않는다. Jetson은 `/proc`, Windows는 운영체제 API를 사용한다.

## 장치 상태

- Jetson power mode를 변경하지 않는다. 확인된 25W mode를 유지한다.
- `jetson_clocks`를 새로 활성화하거나 상태를 변경하지 않는다.
- RTX에서 다른 대규모 GPU compute workload가 발견되면 시작하지 않는다.
- Windows WDDM의 화면·GUI `C+G` context는 환경 정보와 제한 사항으로 기록한다.

## Sustained power/thermal 프로토콜

장치별로 latency 측정과 분리해 순서대로 수행한다.

1. idle: 60초
2. sustained canonical batch-1 FP32 inference: 60초
3. cooldown: 60초

각 phase는 약 1초 간격으로 수집한다. Active phase에서는 모델과 입력을 미리 로드하고 200회 warm-up과 synchronize를 마친 다음 telemetry 수집과 60초 연속 추론을 시작한다. 각 추론은 synchronize하여 실제 완료된 반복만 센다.

- Jetson: `tegrastats`, `VDD_IN` 또는 실제 제공 rail, RAM, CPU, GR3D, CPU/GPU/SoC/TJ 온도
- RTX: `nvidia-smi`, GPU power, GPU temperature, utilization, memory used, SM/memory clocks

Jetson 입력 rail과 RTX GPU board power는 측정 경계가 다르다. cross-device power ratio와 energy-per-inference ratio는 계산하지 않는다.

## 사후 Gate

모든 성능 측정 후 두 장치에서 benchmark vector를 다시 추론하고 E4 RTX reference 및 동일 동결 기준과 비교한다. shape, NaN, Inf, normalized/physical 차이를 검사한다.

## E4 수치 일관성 기준

- shape: exact
- NaN: 0
- Inf: 0
- normalized max absolute difference ≤ 1e-4
- normalized mean absolute difference ≤ 2e-5
- physical max absolute difference ≤ 0.05
- physical mean absolute difference ≤ 0.01

## 중단 기준

- checkpoint hash mismatch
- output consistency 실패
- NaN 또는 Inf
- CUDA OOM
- thermal shutdown 또는 throttling 경고
- Jetson runtime crash
- RTX의 다른 대규모 compute workload
- 의도하지 않은 모델 또는 precision 변경

결과 확인 후 반복 수, 입력, 모델, precision, outlier 정책을 변경하지 않는다.

