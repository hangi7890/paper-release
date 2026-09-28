# Manuscript tables 1–7


## Table 1. Study cohorts, site composition, and roles in the cross-building transfer-learning experiment.



| Cohort | Sites, n | Buildings, n | Site IDs | Role in the study |
| --- | --- | --- | --- | --- |
| Source | 4 | 20 | Crow, Lamb, Rat, Robin | Source-model pretraining and source-side model selection |
| Development | 6 | 12 | Bear, Cockatoo, Fox, Panther, Peacock, Wolf | Protocol development, adaptation analysis, and formulation of final-evaluation hypotheses |
| Final-evaluation proposal | 9 | 510 | Bobcat, Bull, Eagle, Gator, Hog, Moose, Mouse, Shrew, Swan | Prespecified candidate pool for final evaluation |
| Final evaluation | 7 | 453 | Bobcat, Bull, Eagle, Gator, Hog, Moose, Swan | Frozen final evaluation of the predefined protocol |

Note: The source, development, and final-evaluation cohorts used non-overlapping site sets. Fifty-seven buildings were excluded from the final proposal for continuous zero runs of at least 30 days. No eligible buildings remained from Mouse or Shrew. The development cohort was used to finalize the experimental and analysis protocol before final evaluation and was excluded from the final confirmatory estimates.


## Table 2. Forecasting-model architecture, source-training configuration, and target-adaptation protocol.

A  Model and optimization

| Configuration | Source pretraining | Target adaptation |
| --- | --- | --- |
| Historical electricity input / forecast horizon | 24 h / 24 h | 24 h / 24 h |
| Input tensor per sample | 24 × 6 | 24 × 6 |
| Input representation | Historical electricity plus five deterministic target-horizon calendar variables | Same representation |
| Calendar variables | Hour sine/cosine; weekday sine/cosine; weekend indicator | Same representation |
| Architecture | Single-layer LSTM, hidden size 32; linear 24-output head | Same architecture |
| Updated parameters | Full network | Linear forecasting head only; LSTM frozen |
| Parameters at inference | 5,912 | 5,912 |
| Loss | MSE on source-normalized targets; mean reduction | Same loss |
| Optimizer / learning rate | Adam / 0.001 | Adam / 0.0005 |
| Epoch limit / early stopping | 100 / patience 10 | 50 / none |
| Model selection / validation | Chronologically separated source-validation data | No target-specific validation split |
| Training / validation windows | 218,206 / 12,980 | Budget-dependent / not applicable |
| Batch size | 256 in the completed run | Nominal maximum 64; effective size min(64, N) |
| DataLoader settings | shuffle=True; drop_last=False | shuffle=True; drop_last=False |
| Random seeds | 20260924, 20260925, 20260926 | Same three seeds |

Note: MSE denotes mean squared error. N is the number of available adaptation samples. Incomplete final batches were retained. Calendar inputs are deterministic; no future measured electricity or other future-observed covariates were used.


## Table 2. Forecasting-model architecture, source-training configuration, and target-adaptation protocol.

B  Normalization and target-data budgets

| Setting | Value |
| --- | --- |
| Normalization mean | 167.065198 |
| Normalization population standard deviation | 354.328670 |
| Normalization data | Source-training inputs and targets only |
| Use during adaptation and evaluation | Frozen source statistics retained |
| Target adaptation budget (days) | 0 / 1 / 3 / 7 / 14 / 28 |
| Available adaptation windows | 0 / 1 / 49 / 145 / 313 / 649 |
| Effective DataLoader batch size | Not applicable / 1 / 49 / 64 / 64 / 64 |
| 0-day condition | Source-only model without target-label-based updates |
| Evaluation period | Common fixed 28-day interval for every budget within each building |
| Evaluation origins | 28 non-overlapping daily origins |
| Evaluation targets | 672 hourly values per building, seed, and budget |
| Adaptation/evaluation target overlap | 0 h |

Note: Budget, sample-count, and batch-size sequences use the same order. The 1-day condition provides one complete supervised input–target window. No target-test-based early stopping, normalization, or checkpoint selection was used.


## Table 3. Final-evaluation performance across target-adaptation budgets and temporal baselines.



| Model / budget | Adaptation windows | MAE | RMSE | ΔMAE | ΔRMSE | A (%) | B (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source-only (0 d) | 0 | 40.6500 | 54.6597 | — | — | — | 6.82 |
| 1 d adaptation | 1 | 38.4238 | 52.1976 | -2.2262 | -2.4621 | 39.68 | 2.67 |
| 3 d adaptation | 49 | 34.9867 | 48.2985 | -5.6633 | -6.3612 | 69.23 | 7.56 |
| 7 d adaptation | 145 | 29.6476 | 42.1623 | -11.0024 | -12.4975 | 89.83 | 15.87 |
| 14 d adaptation | 313 | 27.4135 | 39.3044 | -13.2365 | -15.3554 | 91.92 | 19.93 |
| 28 d adaptation | 649 | 24.1671 | 35.2364 | -16.4829 | -19.4233 | 94.84 | 26.85 |
| Previous-day | — | 20.8740 | 39.5247 | — | — | — | — |
| Weekly-same-hour | — | 18.1359 | 30.0582 | — | — | — | — |

Note: All estimates use equal weighting of the seven sites after seed averaging within buildings and building averaging within sites. Errors are in original electricity-meter units. Δ denotes adapted minus source-only error. A denotes lower MAE than source-only; B denotes lower MAE than both temporal baselines. A is not applicable to source-only. Previous-day repeats the preceding 24 h; weekly-same-hour uses the observation 168 h before each target. The latter is a strong operational comparator with a longer history than the LSTM’s 24-h electricity input.


## Table 4. Primary and secondary final-evaluation hypotheses and site-equal effect estimates.



| Hypothesis | Prespecified estimand | Estimate | 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| H1 Primary | ΔMAE at 28 d relative to source-only | −16.4829 | [−30.9076, −6.5630] | Supported under the prespecified criterion |
| H2 Secondary | F1 rate(28 d) − F1 rate(1 d) | −55.15 pp | [−67.96, −45.52] pp | Lower frequency of adaptation-induced MAE degradation |
| H3 Secondary | Rate(A) − rate(B) at 28 d | +67.99 pp | [+58.36, +78.35] pp | Source-only improvement more frequent than beating both baselines |

Note: Confidence intervals are 95% percentile intervals from 10,000 site-cluster bootstrap resamples (seed 20260925; seven sites). ΔMAE is adapted minus source-only MAE, in original meter units. F1 denotes adaptation-induced MAE degradation. A denotes source-only improvement; B denotes beating both temporal baselines. pp denotes percentage points. H2 and H3 are secondary effect-size analyses; no p-values or Holm-adjusted tests were calculated.


## Table 5. Predefined adaptation-outcome and failure-event frequencies across target-data budgets.



| Outcome | 1 d | 3 d | 7 d | 14 d | 28 d |
| --- | --- | --- | --- | --- | --- |
| F1 | 813 (59.82%) | 377 (27.74%) | 134 (9.86%) | 104 (7.65%) | 80 (5.89%) |
| F2 | 792 (58.28%) | 350 (25.75%) | 101 (7.43%) | 87 (6.40%) | 69 (5.08%) |
| F3-PD | 1,211 (89.11%) | 1,097 (80.72%) | 921 (67.77%) | 852 (62.69%) | 748 (55.04%) |
| F3-W | 1,119 (82.34%) | 1,000 (73.58%) | 892 (65.64%) | 803 (59.09%) | 691 (50.85%) |
| F3-any | 1,316 (96.84%) | 1,256 (92.42%) | 1,161 (85.43%) | 1,092 (80.35%) | 1,003 (73.80%) |
| F3-both | 1,014 (74.61%) | 841 (61.88%) | 652 (47.98%) | 563 (41.43%) | 436 (32.08%) |
| F4 | 164 (36.20%) | 149 (32.89%) | 61 (13.47%) | 44 (9.71%) | 30 (6.62%) |

Note: F1: MAE degradation relative to source-only. F2: RMSE degradation relative to source-only. F3-PD and F3-W: MAE worse than the previous-day and weekly-same-hour baseline, respectively. F3-any: worse than at least one baseline. F3-both: worse than both. F4: both improving and degrading seeds within the same building and budget. F1–F3 use 1,359 building–seed events per budget; F4 uses 453 buildings per budget. EPS = 10⁻¹² resolves numerical ties only. Across all nonzero budgets, 128/6,795 events (1.88%) improved MAE but worsened RMSE. Rates were recomputed directly from the reported integer counts.


## Table 6. Cross-device FP32 numerical consistency and edge inference benchmark results.

A  Numerical consistency

| Consistency check | Observed value | Prespecified limit | Result |
| --- | --- | --- | --- |
| 64-vector output shape | 64 × 24 | Exact match | PASS |
| Output elements / NaN / Inf | 1,536 / 0 / 0 | No NaN or Inf | PASS |
| Normalized maximum absolute difference | 1.4782 × 10⁻⁵ | ≤1 × 10⁻⁴ | PASS |
| Normalized mean absolute difference | 1.4947 × 10⁻⁶ | ≤2 × 10⁻⁵ | PASS |
| Original-scale maximum absolute difference | 0.005238 | ≤0.05 | PASS |
| Original-scale mean absolute difference | 0.000530 | ≤0.01 | PASS |
| Pre-/post-benchmark gates on each device | Identical acceptance outcomes | Frozen criteria | PASS |

Note: The 64-vector checks compare Jetson CUDA FP32 outputs with the frozen RTX reference. The pre-/post-benchmark gates use the fixed benchmark vector on each device and are distinct from the 64-vector validation. The canonical source-only checkpoint uses seed 20260924. These checks concern numerical consistency, not forecasting superiority.


## Table 6. Cross-device FP32 numerical consistency and edge inference benchmark results.

B  Inference latency and memory

| Category | Metric | Jetson Orin Nano | RTX 3080 |
| --- | --- | --- | --- |
| Model | Parameters / precision / batch | 5,912 / FP32 / 1 | 5,912 / FP32 / 1 |
| Runtime | PyTorch / PyTorch CUDA | 2.11.0 / 12.6 | 2.11.0+cu128 / 12.8 |
| Model-only | Mean (ms) | 0.6906 | 0.2052 |
| Model-only | Median (ms) | 0.6756 | 0.1711 |
| Model-only | P95 (ms) | 0.7311 | 0.3544 |
| Model-only | P99 (ms) | 0.9605 | 0.8857 |
| End-to-end | Mean (ms) | 0.9330 | 0.3374 |
| End-to-end | Median (ms) | 0.8831 | 0.2326 |
| End-to-end | P95 (ms) | 0.9373 | 0.9128 |
| End-to-end | P99 (ms) | 0.9762 | 2.3734 |
| CUDA memory | Peak allocated (MiB) | 17.24 | 17.24 |
| CUDA memory | Peak reserved (MiB) | 22.00 | 22.00 |
| Process memory | RSS (MiB) | 558.80 | Unavailable |

Note: Each mode used 200 warm-up iterations followed by five runs of 1,000 measured inferences. Model-only measures the GPU forward pass with GPU-resident input/output. End-to-end adds CPU-to-GPU and GPU-to-CPU transfers; disk I/O, Python startup, and checkpoint loading are excluded. One Jetson end-to-end observation of 195.8847 ms was retained. RSS converts 585,945,088 bytes to MiB. RTX RSS was unavailable because of a Windows collector error; the benchmark was not rerun.


## Table 6. Cross-device FP32 numerical consistency and edge inference benchmark results.

C  Device-specific power and thermal telemetry

| Category | Metric | Jetson Orin Nano | RTX 3080 |
| --- | --- | --- | --- |
| Power | Measurement boundary | tegrastats VDD_IN input rail | nvidia-smi GPU board power |
| Power | Idle reported power (W) | 3.146 | 63.373 |
| Power | Sustained reported power (W) | 4.031 | 70.422 |
| Power | Peak reported power (W) | 4.032 | 123.660 |
| GPU temperature | Initial telemetry value (°C) | 49.750 | 61.000 |
| GPU temperature | Sustained mean (°C) | 50.329 | 60.600 |
| GPU temperature | Peak (°C) | 50.843 | 63.000 |

Note: Telemetry was measured during a separate sustained-inference workload. Power measurement boundaries differ across devices; no cross-device power or energy-per-inference ratio was calculated. Jetson jetson_clocks status could not be verified. RTX measurements were collected under Windows WDDM with background graphical contexts. The benchmark does not include sensor acquisition, networking, database access, or building-control actions.


## Table 7. Final-evaluation reproducibility, integrity, and protocol-freeze checks.

A  Cohort and input integrity

| Category | Verification | Result | Status |
| --- | --- | --- | --- |
| Cohort freeze | Final cohort after predefined quality screening | 453 buildings / 7 sites | PASS |
| Quality-gate traceability | Exclusions from the 510-building proposal | 57; continuous zero runs ≥30 d | PASS |
| Structural integrity | Buildings passing the structural gate | 453/453; 0 integrity failures | PASS |
| Substitution control | Replacement buildings or alternative test periods after the structural gate | None | PASS |
| Adaptation/test separation | Overlap of adaptation and evaluation target timestamps | 0 h | PASS |
| Common evaluation truth | Forecast origin, target timestamp, horizon, and observed truth across models, budgets, and seeds | Identical building-level digest | PASS |
| Normalization integrity | Source normalization statistics versus frozen values | Match | PASS |
| Checkpoint integrity | Three source checkpoint hashes versus frozen values | Match | PASS |

Note: PASS indicates successful completion of the specified integrity check. It is not a general claim of scientific validity. Full artifact hashes and software identifiers belong in the Supplementary Materials.


## Table 7. Final-evaluation reproducibility, integrity, and protocol-freeze checks.

B  Execution and analysis integrity

| Category | Verification | Result | Status |
| --- | --- | --- | --- |
| Execution completeness | Stored final predictions and execution errors | 6,088,320 rows; 0 errors | PASS |
| Independent metric verification | Maximum stored-versus-recalculated metric difference | 1.71 × 10⁻¹³; tolerance 10⁻¹⁰ | PASS |
| Prediction validity | Unavailable predictions / abnormal metric status | 0 / 0 | PASS |
| Information boundary | Evaluation-period observations used for exploratory characteristics | None; all cutoffs preceded evaluation | PASS |
| Protocol stability | Post-run changes to settings, buildings, periods, seeds, learning rate, or epochs | None | PASS |
| Prediction preservation | Raw predictions regenerated after inspecting performance | No | PASS |
| Exploratory addendum | Missing usage metadata for 18 Swan buildings | Exploratory outputs corrected; predictions, metrics, F1–F4 and H1–H3 unchanged | Documented / resolved |
| Training-configuration audit | Executed loss and batching versus reported configuration | No result-affecting discrepancy identified | PASS |

Note: The exploratory software correction retained 18 Swan buildings with missing usage metadata and changed only the descriptive characteristic outputs. The training audit distinguished the nominal adaptation batch maximum of 64 from the effective DataLoader size min(64, N); no result-affecting discrepancy was identified. Raw predictions and frozen final-evaluation results were preserved.
