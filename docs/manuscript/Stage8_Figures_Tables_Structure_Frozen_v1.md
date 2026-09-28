# Figures/Tables 구조 확정안 v1 — 8 Figures + 7 Tables

확정일: 2026-09-28 (Asia/Seoul)  
기준: `Stage8_Integrated_Manuscript_v2.md`의 현재 본문 및 앞서 합의한 8 Figures + 7 Tables 역할  
작업 범위: 번호·역할·패널·본문 삽입 문단·기존 항목 처리 확정. 실제 그림/표 제작, 본문 편집, Word 배치는 다음 단계. v2는 수정하지 않았다.

## 1. 구조 검토 결과

| 검토 항목 | 결정 |
|---|---|
| Main 구성 | Figure 1–8, Table 1–7 유지 |
| 문단 순서 | 기존 2.1–2.10 / 3.1–3.6 / 4.1–4.6 유지 |
| 그림 중복 | 기존 Edge workflow와 Edge latency를 새 Figure 7의 패널로 통합 |
| 표 중복 | 기존 Table 1 placeholder와 Table 2 cohort-by-site를 새 Table 1로 통합 |
| 무번호 configuration table | 새 Table 2로 승격 |
| 무번호 failure definition table | Methods 2.7의 compact definition list로 전환하여 내용 보존; 추가 Main table로 세지 않음 |
| 기존 failure-taxonomy 그림 | 독립 그림을 없애고 Methods 2.7 정의와 Table 5 주석으로 내용 보존 |
| Figure 8 첫 인용 | Discussion 4.3. Results 3.4의 이미 보고된 사례를 해석하는 위치에 배치 |
| Table 7 첫 인용 | Results 3.6의 끝. 최종 예측평가의 감사사항을 짧은 연결 문장으로 요약 |
| 통계 수준 | Site-equal estimates와 pooled event rates를 축·열·주석에서 구분 |
| Results 수치 | 기존 동결 결과 사용. 구조 변경으로 재계산·재학습하지 않음 |

### 첫 인용 순서

Figures: 2.1 F1 → 2.5 F2 → 3.2 F3 → 3.3 F4 → 3.4 F5 → 3.5 F6 → 3.6 F7 → 4.3 F8.

Tables: 2.2 T1 → 2.5 T2 → 3.2 T3 → 3.3 T4 → 3.4 T5 → 3.6 T6 → 3.6 끝 T7.

Figure와 Table은 각각 독립된 번호 체계다. Figure 8을 3.4에서 먼저 인용하거나 Table 7을 3.1에서 먼저 인용하지 않는다. 3.4의 사례명과 3.1의 무결성 결과 문장 자체는 그대로 유지한다.

## 2. Figures 확정 배치표

| 번호 | 영어 제목 | 역할 / 질문 | 구성 | 첫 인용 및 삽입 위치 |
|---|---|---|---|---|
| Figure 1 | Study design and fixed-test target-adaptation protocol | 어떤 데이터 분리·시간 경계로 비교했는가? | (a) source/development/final 단계; (b) adaptation budget과 공통 test 시간축 | 2.1 마지막 문단 뒤; 기존 Figure 1 위치 |
| Figure 2 | Forecasting architecture and output-head-only target adaptation | 입력·모델·업데이트 대상은 무엇인가? | (a) 24×6 → single-layer LSTM(32) → linear head → 24 outputs; (b) source full-network training vs frozen LSTM + trainable head | 2.5 마지막 본문 문단 뒤, Table 2 앞 |
| Figure 3 | Forecasting performance across target-adaptation budgets | 데이터가 늘면 평균 오차가 어떻게 변하는가? | (a) site-equal MAE; (b) site-equal RMSE; 각 패널에 두 baseline reference lines | 3.2 baseline 비교를 끝내는 마지막 문단 뒤, Table 3 앞 |
| Figure 4 | Primary and secondary hypothesis results | H1–H3의 효과·불확실성은 무엇인가? | (a) H1 site points + overall CI; (b) H2 site-equal F1 rates; (c) H3 A/B site-equal rates | 3.3 마지막 결과 문단 뒤, Table 4 앞 |
| Figure 5 | Adaptation degradation, baseline underperformance, and seed instability | 각 실패 유형이 얼마나 자주 나타나는가? | (a) pooled F1/F2; (b) pooled F3-PD/W/any/both; (c) pooled F4 | 3.4 “Across all nonzero budgets…” 문단 뒤, Table 5 앞 |
| Figure 6 | Exploratory associations between building characteristics and adaptation outcomes | 사전 관측 특성과 적응 효과는 어떻게 연관되는가? | 고정된 탐색 분석의 continuous characteristic × budget Spearman heatmap, outcome=ΔMAE_source; 각 cell에 ρ/n | 3.5 Swan correction 설명 문단 뒤; 기존 exploratory figure 위치 |
| Figure 7 | Edge numerical consistency and inference latency | 같은 모델이 장치에서 재현되며 얼마나 빨리 실행되는가? | (a) validation workflow; (b) normalized/original-scale errors vs tolerances; (c) model-only latency; (d) end-to-end latency | 3.6 기존 마지막 본문 문단(RTX RSS 한계) 뒤, Table 6 앞 |
| Figure 8 | Representative adaptation failure patterns | 이미 보고된 실패가 실제 예측에서 어떻게 나타나는가? | (a) repeated F1 시계열; (b) F4 seed별 ΔMAE 점도표; (c) MAE↓/RMSE↑ 시계열 | 4.3 첫 문단 뒤; 정의·해석 문단 앞 |

### Figure 1 제작 규칙

- Site disjointness는 final cohort와 source/development 사이에 표시한다. Source와 development끼리도 site-disjoint라고 새로 가정하지 않는다.
- Source: 20 buildings / 4 sites; development: 12 / 6; final proposal: 510 / 9 → retained: 453 / 7.
- Final exclusion: 57 records, continuous zero runs ≥30 d. Mouse/Shrew가 0 retained임은 Table 1 주석으로 보존한다.
- 0 d는 adaptation 없는 branch. 1/3/7/14/28 d는 모두 동일 test interval로 연결한다.
- 28 daily origins × 24 h = 672 targets를 명시한다.
- Adaptation budget을 ‘입력 이력까지 포함한 모든 관측 기간’으로 단순화하지 않는다. 1 d에 supervised window 1개가 만들어지는 시간 경계는 동결 window 규칙과 대조하여 표시한다.
- 매 hour input과 horizon을 모두 한 덩어리로 그려 target labels/test overlap이 있는 것처럼 보이게 하지 않는다.

### Figure 2 제작 규칙

- Six variables는 hourly electricity 1개와 deterministic calendar 5개이다.
- Calendar는 horizon에 대응하는 known information임을 표시한다. Future measured load/weather를 입력으로 추가하지 않는다.
- Source-normalization 유지, 5,912 parameters, head-only adaptation을 표시한다.
- Frozen LSTM / trainable head를 색상과 텍스트로 함께 구분한다.
- 모든 LR/epochs/batch 숫자를 그림에 반복하지 않는다. 상세 설정은 Table 2가 담당한다.

### Figure 3–5 통계·표시 규칙

| 대상 | 통계 단위 | 금지할 혼동 |
|---|---|---|
| Figure 3 / Table 3 | 기존 site-equal MAE/RMSE 및 rate | pooled 건물 평균으로 대체하지 않음 |
| Figure 4 / Table 4 | H1 primary; H2/H3 secondary; site-cluster CI | H2/H3에 p-value, 별표, Holm-adjusted 유의성 판정 추가 금지 |
| Figure 5(a,b) / Table 5 | budget별 pooled building × seed; 분모 453×3=1359 | F1 raw 59.82%와 site-equal 약60.3%를 같은 비율로 표기하지 않음 |
| Figure 5(c) / Table 5 | budget별 pooled building; 분모453 | F4에 1359 분모 사용 금지 |

- Figure 3 x축은 0/1/3/7/14/28 d의 실제 간격을 사용한다. 이산 관측점을 선으로 연결한 것으로 설명하고 연속 학습곡선으로 과장하지 않는다.
- Figure 3의 baseline은 기존 MAE/RMSE를 그대로 사용한다. Weekly baseline이 168 h 과거값을 사용하는 사실은 caption 또는 Table 3 주석에 유지한다.
- Figure 4(a): 7 site point estimates와 overall H1 95% CI. 확보되지 않은 site별 CI를 새로 그리지 않는다.
- Figure 4(b): 1 d/28 d site-equal F1와 Δ=-0.5515, CI[-0.6796,-0.4552]. 정확한 endpoint는 동결 자료에서 읽는다.
- Figure 4(c): A=0.9484, B=0.2685, Δ=0.6799, CI[0.5836,0.7835]. A−B CI를 A/B 개별 error bar처럼 사용하지 않는다.
- Figure 5는 기존 event count/기존 분모를 percentage로 표시하는 것에 한정한다. 별도 threshold·magnitude analysis·추가 CI를 도입하지 않는다.
- F1–F3 값이 0-day에 정의된 실패율인 것처럼 연결하지 않는다. Figure 5의 x축은 1/3/7/14/28 d.

### Figure 6 제작 규칙

- 기존 frozen exploratory output에서 ΔMAE_source에 해당하는 전체 continuous-characteristic 행을 사용한다. 큰 |ρ| 순으로 유리한 행만 선택하지 않는다.
- 열: 1/3/7/14/28 d. 0 d는 source 대비 변화가 정의상 0이므로 상관 열을 별도로 만들지 않는다.
- Missing/undefined correlations는 회색 N/A, 0으로 채우지 않는다. 표본 n이 budget/feature별로 달라짐을 표시한다.
- 색 범위는 −1~1, 0 중심. Causal interpretation 및 significance stars 없음.
- Swan 누락 수정 이후 output만 사용한다. Site/use grouping 및 다른 outcome의 상세 association은 Supplementary.
- Figure 6을 새 예측변수 선별·건물 배제 규칙으로 해석하지 않는다.

### Figure 7 제작 규칙

- 2×2 패널. (a)는 workflow, (b)는 acceptance 비교, (c,d)는 실제 latency distributions.
- (b): 네 error summary를 각자의 tolerance로 나눈 비율로 표시하면 공통 기준선1을 쓸 수 있다. 이 비율은 수치일치 허용치 대비 비율이며 성능·전력효율 비율이 아니다. 원래 error/tolerance도 label 또는 Table 6에 보존한다.
- (c,d): 실제 저장된 latency observations로 ECDF, x축 log(ms); device를 선으로 구분. Run 수5, 각1000회, 총5000회/device/mode를 caption에 명시한다. 이를 5000개의 독립 실험으로 해석하지 않는다.
- Jetson end-to-end 195.8847 ms 관측을 포함한다. Crop/outlier deletion 금지.
- 원시 latency가 확보되지 않으면 median/P95로 분포를 합성하지 않는다. 분포 패널 제작은 원자료 확보 후 진행한다.
- Power/thermal은 Table 6에만 수록하고 본 그림의 별도 효율 비교 패널로 확장하지 않는다.

### Figure 8 제작 규칙

| Panel | Frozen representative case | 표시 |
|---|---|---|
| (a) | Hog_office_Roger / 1 d / seed20260926 | Observed, source-only, adapted의 672 h; repeated F1 label |
| (b) | Eagle_education_Brooke / 14 d / seeds20260924–20260926 | seed별 ΔMAE_source 점; 0 기준선; negative=improvement, positive=degradation |
| (c) | Hog_office_Gustavo / 1 d / seed20260926 | Observed, source-only, adapted의 672 h; ΔMAE<0, ΔRMSE=+6.485 |

- 현재 대화의 기존 caption을 기본으로 사용하되 실제 원자료와 226.533 range / +6.48485 원값을 대조한 뒤 표시한다.
- 3-panel 가로 배치, (a,c)는 넓게 (b)는 좁게. 최종 Word 폭에서 672 h 선의 판독이 어렵다면 상·하단 패널 배치로 변경 가능하나 panel 내용·선택 사례는 유지한다.
- (a)의 한 budget 시계열만으로 ‘모든 budget에서 반복’을 입증했다고 표현하지 않는다. Repeated F1은 frozen selection record에 근거한 annotation이며 시계열은 1 d 사례를 보여준다.
- 대표 사례 선택은 동결 규칙을 따르고 plot 후 시각적으로 다시 선택하지 않는다.
- 본문3.4에 이미 보고된 사례를 Discussion4.3에서 시각적으로 해석하는 역할이다. 새로운 가설검정이나 새 모델 결과를 Discussion에 추가하지 않는다.

## 3. Tables 확정 배치표

| 번호 | 영어 제목 | 구성 | 첫 인용 및 삽입 위치 |
|---|---|---|---|
| Table 1 | Dataset cohorts and retained final-evaluation sites | Panel A: source/development/proposed/retained cohort 요약; Panel B: final 7site별 건물 수 | 2.2 마지막 문단 뒤 |
| Table 2 | Forecasting model and training/adaptation configuration | 기존 v2 configuration table + train/validation windows + adaptation windows 설명; MSE/mean reduction, normalization, shuffle/drop_last | 2.5 끝 Figure 2 다음 |
| Table 3 | Site-equal forecasting performance across adaptation budgets | 0/1/3/7/14/28 d MAE/RMSE/ΔMAE/ΔRMSE/A/B + baseline 2행 | 3.2 끝 Figure 3 다음 |
| Table 4 | Primary and secondary hypothesis effect estimates | H1/H2/H3, estimand, point estimate,95%CI, interpretation/analysis status | 3.3 끝 Figure 4 다음 |
| Table 5 | Pooled adaptation-outcome event counts and rates | 행=F1,F2,F3-PD,W,any,both,F4; 열=budget별 n(%), unit/denominator 주석 | 3.4 Figure 5 다음 |
| Table 6 | Cross-device FP32 consistency and edge benchmark results | Panel A: tolerance/error; Panel B: latency/CUDA memory/RSS; Panel C: device-specific telemetry | 3.6 Figure 7 다음 |
| Table 7 | Final-evaluation reproducibility, integrity, and protocol-freeze checks | 기존 합의의 15항목 + training-configuration audit 1행; full hash는 Supplementary | 3.6 Table 6 뒤의 마지막 요약 문단 |

### Table 1

- Panel A 열: Cohort / Purpose / Buildings / Sites. Source20/4, development12/6, proposed final510/9, retained final453/7.
- Proposed와 retained는 독립 cohort인 것처럼 합산하지 않는다. Source/development site 수도 단순 합산하여 unique sites라고 쓰지 않는다.
- Panel B: Bobcat22, Bull111, Eagle94, Gator63, Hog135, Moose10, Swan18, total453.
- 57개 exclusion 및 Mouse/Shrew zero retained는 footnote에 둔다.
- 기존 Results3.1의 duplicate cohort table은 제거하되 관련 문단은 유지하고 Table1을 다시 인용한다.

### Table 2

- v2의 완료된 MSE·batch 정보를 그대로 사용한다. Source batch256은 loader setting이며 마지막 partial batch까지256이라는 의미가 아님을 drop_last footnote로 분명히 한다.
- Source train/validation:218,206/12,980. Adaptation windows:1/49/145/313/649.
- Nominal maximum64; actual loader setting=min(64,N). Effective setting1/49/64/64/64.
- MSE mean reduction / source-normalized targets, Source/adaptation shuffle=True, drop_last=False.
- 0 d는 no adaptation. Source maximum100 + patience10 vs adaptation50 + no early stopping을 구분.
- OOM fallback 상세, batch별 마지막 잔여 sample 수는 Supplementary; Main 표에는 unused fallback 후보를 실제 실행 batch로 넣지 않는다.
- Full normalization precision은 Supplementary에, 본문/표에는 기존6자리 표현을 사용한다.

### Table 3

- 기존 Table3의 6 budget 결과와 수치를 보존한다.
- Previous-day MAE20.8740/RMSE39.5247, weekly MAE18.1359/RMSE30.0582를 baseline rows로 추가한다.
- Baseline rows의 adaptation budget/Δ/A/B는 —. Baseline을 adaptation condition으로 분류하지 않는다.
- A와 B는 site-equal rates, 0–1 scale임을 footnote에 명시한다. Figure5의 pooled percent와 구분한다.
- MAE/RMSE는 original meter units. Weekly168 h vs neural24 h information distinction 유지.

### Table 4

| Row | Estimand | Estimate | 95% site-cluster CI | Status 표기 |
|---|---|---:|---|---|
| H1 | MAE(28 d)−MAE(source-only) | −16.4829 | [−30.9076,−6.5630] | Primary; predefined support criterion met |
| H2 | F1 rate(28 d)−F1 rate(1 d) | −0.5515 | [−0.6796,−0.4552] | Secondary effect-size analysis |
| H3 | A rate(28 d)−B rate(28 d) | 0.6799 | [0.5836,0.7835] | Secondary effect-size analysis |

- 이 표는 기존 Results3.3 수치의 배치 specification이다. p-value 열을 만들지 않는다.
- Site clusters7, bootstrap resamples10,000, bootstrap seed20260925를 footnote에 둔다.
- H2/H3의 rate difference와 percentage points를 혼용하지 않는다. 표에는0–1 rate difference를 유지한다.

### Table 5

- n(%)는 기존 count를 정해진 분모로 표시하며 반올림만 수행한다.
- F1–F3:1359 per budget; F4:453 per budget. Total across budgets는 Supplementary로 보내 표 폭을 줄인다.
- 표에 short outcome label을 포함하거나 footnote에 F1/F2/F3/F4 비교기준을 기록한다.
- Numerical tie tolerance10^-12를 footnote에 명시한다. F3의 strict worse-than event와 ‘beat하지 못함’은 tie에서 같지 않을 수 있으므로 수학적 event 정의와 동결 tie 처리를 기준으로 label을 작성한다.
- 128 MAE↓/RMSE↑ 사건은 독립 failure category row로 신설하지 않는다. 기존 본문과 Figure8(c)에서 유지한다.

### Table 6

- Panel A: max/mean normalized error, max/mean original-scale error, 각각의 tolerance, shape/NaN/Inf acceptance.
- Panel B: 기존 mean/median/P95 latency, allocated/reserved CUDA memory, process RSS. Model-only와 deployment end-to-end 구분.
- RTX RSS는 unavailable; 기존 collection error footnote 유지. CUDA memory를 process RAM으로 표기하지 않는다.
- Panel C: power/thermal telemetry는 device-specific boundary를 각 열에 기재. Jetson VDD_IN과 RTX board power 간 ratio/energy-per-inference 열 없음.
- 극단값195.8847 ms를 footnote 또는 maximum 항목으로 보존. 모든 장치의 미확보 max 값을 임의 작성하지 않는다.
- E2E는 transfer+forward 범위이며 sensing/network/disk/startup/checkpoint load를 포함하는 실제 서비스 총지연으로 표현하지 않는다.

### Table 7

- 기존 합의의 verification/result/status 구조를 유지하며 감사 완료행을 추가한다.
- 결과 문구: `No result-affecting discrepancy identified`. Nominal64와 actual loader1/49의 metadata semantics 차이를 ‘불일치 전혀 없음’으로 덮지 않는다.
- Swan exploratory correction: `Documented / resolved`. Raw predictions·MAE/RMSE·H1–H3·F1–F4 unchanged 범위 유지.
- Full hashes, exact code locations 및 artifact IDs는 Supplementary로 이동.
- Main 표의 PASS는 수행된 해당 검증의 통과를 의미하며 모든 과학적 타당성의 포괄 증명으로 표현하지 않는다.

## 4. 기존 요소의 이동·통합 매핑

| v2 요소 | 확정 처리 |
|---|---|
| Figure1 protocol | 새 Figure1 |
| Figure2 failure taxonomy | Methods2.7 definition list + Table5 footnote로 의미 보존; 독립 그림 제거 |
| Figure3 edge workflow | 새 Figure7(a) |
| Figure4 adaptation curve | 새 Figure3 |
| Figure5 H1–H3 | 새 Figure4 |
| Figure6 failure rates | 새 Figure5 |
| Figure7 exploratory associations | 새 Figure6 |
| Figure8 edge latency | 새 Figure7(c,d) |
| 새 architecture/adaptation schematic | 새 Figure2 |
| Results3.4 대표 사례 | 새 Figure8, Discussion4.3에 시각화 |
| Table1 cohort placeholder + Table2 cohort-by-site | 새 Table1 |
| 무번호 model configuration | 새 Table2 |
| 무번호 failure definition table | Methods2.7 compact list; 추가 Main table 없음 |
| Table3 main performance | 새 Table3 |
| Results3.3 H1–H3 문단 | 새 Table4; 기존 문단 유지 |
| Table4 failure counts | 새 Table5 |
| Table5 edge benchmark | 새 Table6 |
| 후속 integrity/training audit | 새 Table7 |

따라서 완료 후 무번호 데이터표를 별도로 남겨 ‘7 Tables’ 외 추가 표가 생기지 않도록 한다. A/B panels가 있는 표는 하나의 caption과 번호 아래 묶는다.

## 5. 본문 반영 시 사용할 정확한 앵커와 연결 문장

아래 문장은 이번 배치 결정을 위해 새로 작성한 편집 연결 문장이다. 과거 대화의 원문 인용이라고 취급하지 않는다. 아직 v2에 삽입하지 않았다.

| 위치 | v2 앵커 | 첫 인용용 연결 문장 / 처리 |
|---|---|---|
| 2.1 끝 | “...rather than treating individual hourly observations or repeated seeds as independent samples.” | The study design and fixed-test adaptation protocol are summarized in Figure 1. |
| 2.2 끝 | “This limitation is retained when interpreting differences in absolute load magnitude across buildings.” | The study cohorts and retained final-evaluation sites are summarized in Table 1. |
| 2.5 끝 | “...exactly one hour.”로 끝나는 reproducibility 문단 | The forecasting architecture and head-only adaptation procedure are illustrated in Figure 2, and the model and training settings are summarized in Table 2. |
| 3.1 | 기존 Table2 placeholder 및 cohort 표 자리 | The retained site composition is listed in Table 1. 기존 duplicate table 삭제 |
| 3.2 끝 | “...both MAE and RMSE.”로 끝나는 baseline 문단 | Figure 3 shows the adaptation curves, and Table 3 reports the corresponding site-equal estimates and temporal-baseline results. |
| 3.3 끝 | “...nevertheless outperformed both temporal baselines.” | The primary and secondary effects are summarized in Figure 4 and Table 4. |
| 3.4 | “...not always identical across the two error metrics.” | Pooled event rates are shown in Figure 5, with the event counts, percentages, and denominators reported in Table 5. |
| 3.5 끝 | “...or H1–H3 outcomes.” | Figure 6 summarizes the exploratory associations using the corrected characteristic outputs. |
| 3.6 기존 끝 | “...the benchmark was not rerun to replace the missing measurement.” | Figure 7 summarizes the deployment workflow, numerical consistency, and latency distributions; the benchmark measurements are reported in Table 6. |
| 3.6 최종 | Figure7/Table6 배치 뒤, 4.Discussion 시작 전 | For the final forecasting evaluation, the integrity and training-configuration checks are summarized in Table 7. |
| 4.3 첫 문단 뒤 | “In addition, 128 cases improved in MAE while worsening in RMSE.” | Figure 8 illustrates the representative cases reported in Section 3.4, using the frozen case-selection rules. |

Figure8의 내용은 Results3.4에 이미 보고됐으므로 4.3에는 새로운 수치 주장 문단을 만들 필요가 없다. Table7은 Table6 뒤에서 예측평가 감사로 주제를 명확히 전환하는 한 문장만 추가한다.

## 6. 제작 자료 준비 상태

현재 작업에서는 원고·기존 대화의 수치를 구조 설계에 사용했다. 아래 raw artifacts의 현재 파일 존재/내용을 새로 검사한 것은 아니다. 다음 제작 단계에서 해당 자료를 읽어 대조한다.

| 대상 | 제작 입력 | 현재 작업에서 가능한 범위 |
|---|---|---|
| F1/F2, T1/T2 | 동결 protocol / v2 / training audit | 구조·문구·설정값 확보 |
| F3, T3 | 최종 site-equal metrics와 baselines | 원고 요약값 확보; 제작은 full precision 원자료 사용 |
| F4, T4 | hypothesis report / site-level effects / CI | 전체효과와 CI 확보; site별7개 값 및 endpoint 원자료 대조 필요 |
| F5, T5 | budget별 frozen counts | 원고에 count와 분모 확보 |
| F6 | correction 이후 exploratory correlations + n | 원고에 일부 요약값; 전체 matrix 필요 |
| F7, T6 | consistency output / raw latency / telemetry | summary값 확보; 분포용 latency 원자료 필요 |
| F8 | 세 사례 frozen predictions / seed metrics / selection record | 사례 ID·일부 annotation 확보; 곡선·seed별 값 필요 |
| T7 | final integrity / software addendum / training audit | 합의된 항목 확보; artifact별 최종 대조 필요 |

원자료가 필요한 항목은 제작 준비 조건이며 이번 번호·역할·삽입 위치 확정을 막는 미결정사항은 아니다. 실제 제작에서 요약값만으로 시계열·분포·correlation matrix를 합성하지 않는다.

## 7. Supplementary로 보낼 상세정보

| 정보 | Main 연결 | Supplementary 처리 |
|---|---|---|
| 전체 건물·site·기간 manifest | T1/F1 | 상세 cohort/time boundaries |
| 정확한 normalization constants / code versions / budgets별 partial batches | T2 | 재현성 세부 설정 |
| 전체 building/seed/budget metrics 및 site-level effects | T3/T4/F3/F4 | 상세 수치 |
| 모든 탐색 association / category summaries / missingness | F6 | 전체 exploratory outputs |
| run별 latency / telemetry / 환경 및 RSS 누락 설명 | F7/T6 | Edge 상세 자료 |
| 대표 사례 선택 규칙 / 전체 budget의 repeated-F1 근거 | F8 | Frozen selection 근거 |
| full SHA-256 / integrity audit / training audit / addendum | T7 | 감사자료; 제안된 S4 번호는 Supplementary 전체 구조에서 최종 확인 |

## 8. 완료 조건과 다음 단계

- Main8 figures/7 tables로 모든 기존 표·그림의 역할을 배정했다.
- 첫 인용 번호는 각 체계 내에서1→8,1→7 순서다.
- 현재 원고의 문단 순서·결과값은 수정하지 않았다.
- 다음 단계: 이 확정안에 따라 실제 Figure/Table 제작에 필요한 동결 자료를 확보·대조하고 제작한다. Word 반영 때 위 앵커와 인용 문장을 적용한다.
- 최종 그림/표 크기·해상도·페이지 나눔은 실제 제작 및 MDPI template QA에서 검증한다.
