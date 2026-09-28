# Final Stage-6 C0 LSTM Training Configuration — Read-Only Audit

## 감사 목적과 범위

이 감사의 목적은 논문 Methods/Table 2에 필요한 Source loss, target-adaptation loss, Source batch size, target-adaptation의 실제 batch 규칙을 최종 Stage-6 실행 Python 코드에서 직접 확인하는 것이다.

- 새 실험: 수행하지 않음
- 모델 실행·재학습·적응: 수행하지 않음
- prediction 재생성: 수행하지 않음
- 기존 코드·config·checkpoint·prediction·metric·report 수정: 없음
- 감사 방식: 소스 코드의 import/call path, 함수 구현, 기존 write-once 실행 기록을 읽기 전용으로 대조

## 감사 대상 최종 execution path

1. Source checkpoint 생성 진입점: `scripts/run_bdg2_stage6_c0_lstm_v3.py::main`
   - lines 359–362: `fit_source(...)` 호출
   - lines 365–380: `bdg2_stage6_c0_lstm_seed_{seed}_v3.pt` 저장
2. 최종시험 진입점: `scripts/final_v1/run_final_v1_gate2.py::main`
   - lines 32–37: `run_bdg2_stage6_c0_lstm_v3`에서 `C0LSTM`, `Config`, `adapt_head`, `predict` 직접 import
   - lines 73–76, 143–146: 세 Source checkpoint SHA-256 고정 및 검사
   - lines 159–169: 동결 Source config로 `Config` 생성 후 기본 `Config()`와 동일성 검사
   - lines 427–441: 동결 checkpoint 로드 및 Source model 구성
   - lines 476–490: 최종 건물별 adaptation window를 구성하고 import한 `adapt_head(...)` 호출
3. `scripts/bdg2_stage6_c0_common_v3.py`는 C0 feature 구성을 담당하며 loss, optimizer 또는 batching은 정의하지 않는다.

감사한 주요 소스 SHA-256:

- `scripts/run_bdg2_stage6_c0_lstm_v3.py`: `2c5f8204d37ca1b58efcf11086f90d5b91a134fb2a886fcb43e4e06e3037df7c`
- `scripts/bdg2_stage6_c0_common_v3.py`: `85414e839953c2c93b95571c854dcb2a6350c23677af1ebe769bd17a03de3699`
- `scripts/final_v1/run_final_v1_gate2.py`: `ea3941000cba7f5a72c8333652aff8f6b549e679af2ee780c188448eb09f9384`
- `models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260924_v3.pt`: `88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b`

---

# [FINAL AUDIT RESULT]

## 1. Source training loss

- **Result:** Source-normalized 24시간 target에 대한 mean squared error.
- **Exact implementation:** `torch.nn.MSELoss()`의 기본 mean reduction.
- **File / function / line:** `scripts/run_bdg2_stage6_c0_lstm_v3.py`, `fit_source_once`, lines 150–174; criterion line 154, 실제 forward→loss line 172.
- **Evidence:**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=config.source_learning_rate)
criterion = nn.MSELoss()
...
loss = criterion(model(features), targets)
loss.backward()
optimizer.step()
```

`make_dataset` lines 132–135에서 `targets = ((y - mean) / std).astype(np.float32)`로 target을 Source 통계로 정규화한다. Source validation/checkpoint 선택 loss는 `validation_loss` lines 138–147에서 squared error 전체 합을 `targets.numel()`로 나누므로 element-level MSE이다.

실제 실행 경로 근거: `main` lines 359–362 → `fit_source` lines 209–221 → `fit_source_once` lines 150–206. 선택된 state는 lines 365–380에서 최종 Stage-6 checkpoint로 저장된다.

## 2. Target adaptation loss

- **Result:** Source-normalized 24시간 target에 대한 mean squared error.
- **Same as source training?:** **Yes**
- **Exact implementation:** `torch.nn.MSELoss()`의 기본 mean reduction.
- **File / function / line:** `scripts/run_bdg2_stage6_c0_lstm_v3.py`, `adapt_head_once`, lines 234–260; criterion line 249, 실제 forward→loss line 256.
- **Evidence:**

```python
optimizer = torch.optim.Adam(model.head.parameters(), lr=config.adaptation_learning_rate)
criterion = nn.MSELoss()
...
loss = criterion(model(features), targets)
loss.backward()
optimizer.step()
```

`Config` lines 67–69은 adaptation learning rate `5e-4`, epochs `50`, nominal batch `64`를 정의한다. 최종 `final_v1` 실행은 `run_final_v1_gate2.py` lines 32–37에서 이 `adapt_head`를 직접 import하고 lines 476–490에서 호출한다.

## 3. Source batch size

- **Result:** 완료된 세 Source seed 모두 **256** 사용.
- **Exact rule:** 최초 후보는 `config.source_batch_size=256`; CUDA OOM인 경우에만 128, 이후 64로 fallback. 실제 완료 기록에는 세 seed 모두 batch 256, `oom_errors=[]`이다. `drop_last`가 지정되지 않아 마지막 partial batch를 유지한다.
- **File / function / line:**
  - `scripts/run_bdg2_stage6_c0_lstm_v3.py::Config`, line 63
  - `fit_source_once`, lines 150–162
  - `fit_source`, lines 209–221
  - `main`, lines 359–390
- **Evidence:**

```python
source_batch_size: int = 256
...
loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed),
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
...
for batch_size in (config.source_batch_size, 128, 64):
    try:
        result = fit_source_once(..., batch_size)
        return (*result, batch_size, oom_errors)
    except torch.cuda.OutOfMemoryError as exc:
        ...
```

기존 완료 기록 대조:

- `models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json`의 세 checkpoint 모두 `batch_size=256`, `oom_errors=[]`.
- `reports/bdg2_stage6_c0_training_history_v3.csv`의 모든 seed/epoch `batch_size=256`.
- Source training sample 218,206개이므로 epoch당 256개 full batch 852개와 94개 partial batch 1개가 생성된다.

## 4. Target adaptation batch size

- **Result:** 명목/최초 시도 batch는 64이나, 실제 DataLoader batch는 **`min(성공한 후보 batch, len(dataset))`**이다. 최종 실행에서는 모든 비제로 예산에서 후보 64가 성공했고 OOM fallback은 없었다.
- **Exact rule:** `adapt_head`가 `(64, 32, 16)`을 OOM fallback 후보로 순서대로 시도하고, `adapt_head_once`가 `batch_size=min(batch_size, len(dataset))`로 DataLoader를 생성한다.
- **Effective behavior for 1-day / 3-day / 7-day / 14-day / 28-day budgets:**

| Budget | Adaptation samples | Actual DataLoader batch size | Epoch당 실제 batch 크기 | Epochs |
|---:|---:|---:|---|---:|
| 1일 | 1 | 1 | `[1]` | 50 |
| 3일 | 49 | 49 | `[49]` | 50 |
| 7일 | 145 | 64 | `[64, 64, 17]` | 50 |
| 14일 | 313 | 64 | `[64, 64, 64, 64, 57]` | 50 |
| 28일 | 649 | 64 | `[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 9]` | 50 |

0일은 Source-only이므로 adaptation DataLoader와 optimizer update가 없다.

- **File / function / line:**
  - `scripts/run_bdg2_stage6_c0_lstm_v3.py::Config`, lines 67–69
  - `adapt_head_once`, lines 234–260; dynamic rule line 243
  - `adapt_head`, lines 263–275; fallback candidates line 265
  - `scripts/final_v1/run_final_v1_gate2.py`, expected samples line 61, assertion lines 210–225, actual call lines 476–490
- **Evidence:**

```python
adaptation_learning_rate: float = 5e-4
adaptation_epochs: int = 50
adaptation_batch_size: int = 64
...
loader = DataLoader(
    dataset,
    batch_size=min(batch_size, len(dataset)),
    shuffle=True,
    generator=torch.Generator().manual_seed(seed),
    num_workers=0,
)
...
for batch_size in (config.adaptation_batch_size, 32, 16):
```

최종 실행의 `EXPECTED_SAMPLES = {0: 0, 1: 1, 3: 49, 7: 145, 14: 313, 28: 649}`와 `min(...)` 규칙으로 위 batch 구성이 직접 결정된다. `reports/final_v1/gate2_execution_metrics.csv`는 비제로 예산 전체에서 성공 후보값 64, `oom_fallback_count=0`, `status=ok`를 기록한다.

## 5. Related batching behavior

- **shuffle:** Source `True`; adaptation `True`. 두 loader 모두 seed가 지정된 `torch.Generator()`를 전달한다.
- **drop_last:** Source `False`; adaptation `False`. 최종 코드에서 `drop_last` 인수를 전달하지 않으므로 PyTorch DataLoader 기본값 `False`가 적용되고 partial batch가 유지된다.
- **other relevant behavior:**
  - Source loader: `num_workers=0`, CUDA 가능 시 `pin_memory=True`.
  - Adaptation loader: `num_workers=0`, `pin_memory` 미지정.
  - Source batch fallback은 CUDA OOM에만 반응하며 256→128→64 순서다.
  - Adaptation candidate fallback은 CUDA OOM에만 반응하며 64→32→16 순서다.
  - 최종 완료 실행에서는 Source와 adaptation 모두 OOM fallback이 발생하지 않았다.

## 6. Head-only adaptation verification

- **LSTM frozen?:** **Yes**
- **Updated parameter set:** `model.head.parameters()`만 Adam optimizer에 전달된다. LSTM parameter는 모두 `requires_grad=False`로 설정된다.
- **File / function / line:** `scripts/run_bdg2_stage6_c0_lstm_v3.py::adapt_head_once`, lines 234–249; freeze lines 238–239, optimizer line 248.

```python
for parameter in model.lstm.parameters():
    parameter.requires_grad = False
...
optimizer = torch.optim.Adam(model.head.parameters(), lr=config.adaptation_learning_rate)
```

최종 호출 근거는 `scripts/final_v1/run_final_v1_gate2.py` lines 32–37과 476–490이다. 최종 head bundle은 lines 499–503에서 `head.weight`와 `head.bias`만 저장한다.

## 7. Consistency check

- **Does the final code agree with the existing Stage-6 report/config?** 부분적으로 Yes. Source Adam 0.001, Source batch 256, adaptation Adam 0.0005, 50 epochs, head-only 범위와 nominal adaptation batch 64는 코드·config·protocol이 일치한다. 기존 narrative report/protocol에는 loss class와 dynamic adaptation batch 규칙이 직접 명시되어 있지 않다.
- **Any discrepancy:** `adaptation_batch_size=64`는 실제 DataLoader batch가 아니라 성공한 candidate/maximum batch다. 최종 metric 파일의 같은 이름 필드도 1일 및 3일에 64를 기록하지만 실제 loader batch는 각각 1과 49다. 이는 기록 필드의 의미/문서화 discrepancy이며 학습 코드 충돌은 아니다.
- **Does any discrepancy affect previously generated predictions/results?:** **No**, 재실행 없이 코드와 기록으로 판단 가능한 범위에서 수치 결과에는 영향이 없다. 기존 prediction은 감사한 dynamic `min(...)` 규칙으로 이미 생성되었다. 논문 Table 2에서 단순히 “adaptation batch size=64”라고만 쓰면 1일·3일 실제 behavior를 잘못 설명하므로 규칙으로 수정해 기술해야 한다.

## 8. Manuscript-ready values

Source training loss = Mean squared error (`torch.nn.MSELoss`, mean reduction) on source-normalized targets  
Target adaptation loss = Mean squared error (`torch.nn.MSELoss`, mean reduction) on source-normalized targets; same as Source training  
Source batch size = 256 (`drop_last=False`; OOM-only fallback 128→64; completed three-seed run used 256)  
Target adaptation batch size/rule = `min(64, number of adaptation samples)`, `drop_last=False`; OOM-only candidate fallback 32→16, unused in the completed run

## 9. Audit artifact preservation

- Markdown: `reports/final_v1/final_stage6_training_config_audit.md`
- JSON: `reports/final_v1/final_stage6_training_config_audit.json`
- JSON contains the required structured fields and audited file hashes.
- Unverified items: 없음.

## Artifact checksums

- JSON SHA-256: `c2bb57bd076791d1db5bc4cdde3d7af260251d99599463697105e5e083fe1bb1`
- Markdown SHA-256: self-referential value cannot be embedded in the same file without changing that file. The exact finalized Markdown SHA-256 is calculated after this file is closed and reported with the completion result.

