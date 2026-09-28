# Supplementary Master Structure

This structure supplements the frozen eight-figure and seven-table main-text plan. Main figures and tables are not duplicated. Methods, tables, figures, and data files use independent numbering sequences.

## Supplementary Methods

| number | title | data basis | first main-text citation |
|---|---|---|---|
| Methods S1 | Scope, frozen artifacts, and Supplementary assembly | Final Gate 0-3 reports; Stage-6 audit; E4/E5 reports | Methods 2.1 and Table 7 integrity sentence |
| Methods S2 | Building eligibility, cohort construction, and quality control | final_v1_cohort_manifest.csv | Methods 2.2 after cohort description |
| Methods S3 | Temporal boundaries and 24-to-24 window construction | final_v1_temporal_manifest.csv | Methods 2.3 after forecasting task |
| Methods S4 | C0 features, normalization, architecture, and source training | Stage-6 code, config, and audit | Methods 2.4-2.5 |
| Methods S5 | Output-head-only adaptation and actual batching | Stage-6 audit and Gate 2 execution metrics | Methods 2.5 after adaptation budgets |
| Methods S6 | Metrics, site-equal aggregation, bootstrap, and F1-F4 events | Gate 3 code and final summary tables | Methods 2.6-2.8 |
| Methods S7 | Leakage-controlled exploratory characteristics and Swan correction | corrected final_v1 characteristic outputs | Methods 2.9 and Results 3.5 |
| Methods S8 | Cross-device numerical validation and edge benchmarking | E4/E5 frozen outputs | Methods 2.10 and Results 3.6 |
| Methods S9 | Representative-case selection, extraction, and reproducibility | Figure 8 validation and selection records | Discussion 4.3 after first paragraph |

## Supplementary Tables

| number | title | data basis | first main-text citation |
|---|---|---|---|
| Table S1 | Cohorts and final-site retention | Table_S1_A/B | Methods 2.2 |
| Table S2 | Exact training, adaptation, and batching configuration | Table_S2 and Stage-6 audit | Methods 2.5 |
| Table S3 | Complete site-equal performance and temporal baselines | final_v1_site_equal_adaptation_curve | Results 3.2 |
| Table S4 | H1-H3 overall and site-level effects | H1-H3 site effect CSVs | Results 3.3 |
| Table S5 | Complete failure-event counts and rates | final_v1_failure_summary | Results 3.4 |
| Table S6 | Exploratory association coverage and corrected missing-use category | corrected characteristic outputs | Results 3.5 |
| Table S7 | E4 FP32 cross-device numerical consistency | validation_64_comparison_summary | Results 3.6 |
| Table S8 | E5 latency, memory, and device-specific telemetry | E5 raw and summary outputs | Results 3.6 |
| Table S9 | Key reproducibility hashes | artifact manifests and current files | Table 7 reproducibility summary |
| Table S10 | Frozen Figure 8 cases and recalculated metrics | validated 6,720-row extraction | Discussion 4.3 |

## Supplementary Figures

| number | title | data basis | first main-text citation |
|---|---|---|---|
| Figure S1 | Adaptation windows and effective batching | Table S2B | Methods 2.5 |
| Figure S2 | Site-level ΔMAE across budgets | Data S5 | Results 3.2 |
| Figure S3 | Complete pooled failure-event rates | Table S5 | Results 3.4 |
| Figure S4 | Full exploratory Spearman matrix | Data S8 | Results 3.5 |
| Figure S5 | Run-level latency distributions | Data S11 | Results 3.6 |
| Figure S6 | Device-specific power and temperature phases | Data S12 | Results 3.6 |
| Figure S7 | Budget-by-seed patterns for frozen Figure 8 cases | Data S4 | Discussion 4.3 |

## Supplementary Data

| number | file | rows | content | primary_key |
|---|---|---|---|---|
| Data S1 | Data_S1_final_cohort_manifest.csv | 510 | All proposed final buildings, quality fields, retention status, and exclusion reason | building_id |
| Data S2 | Data_S2_temporal_manifest.csv | 2718 | Six frozen temporal conditions for every retained building | building_id + adaptation_days |
| Data S3 | Data_S3_all_model_metrics.csv | 9060 | Baselines and all neural final metrics | building_id + model + seed + adaptation_days |
| Data S4 | Data_S4_neural_data_efficiency.csv | 8154 | 453 × 3 seed × 6 budget neural results and deltas | building_id + seed + adaptation_days |
| Data S5 | Data_S5_site_budget_seed_summary.csv | 126 | 7 site × 6 budget × 3 seed building-equal summaries | site_id + adaptation_days + seed |
| Data S6 | Data_S6_failure_events.csv | 6795 | F1-F3 and discordance at building × nonzero budget × seed | building_id + adaptation_days + seed |
| Data S7 | Data_S7_failure_by_building_budget.csv | 2265 | Building × nonzero budget summaries and F4 | building_id + adaptation_days |
| Data S8 | Data_S8_exploratory_associations.csv | 350 | 14 features × 5 responses × 5 nonzero budgets | feature + response + adaptation_days |
| Data S9 | Data_S9_exploratory_group_summary.csv | 105 | Site and use-category descriptive failure summaries | group_field + group_value + adaptation_days |
| Data S10 | Data_S10_building_characteristics.csv | 2718 | Leakage-controlled pre-test characteristics for six budgets | building_id + adaptation_days |
| Data S11 | Data_S11_edge_latency_observations.csv | 20000 | 2 devices × 2 modes × 5 runs × 1000 observations | device + mode + run_id + repetition |
| Data S12 | Data_S12_edge_telemetry_summary.csv | 6 | Device-specific idle, active, and cooldown summaries | device + phase |
| Data S13 | Data_S13_e4_validation_64_comparison.csv | 1536 | 64 vectors × 24 horizons RTX reference versus Jetson output | vector_id + horizon_index |
| Data S14 | Data_S14_figure8_case_predictions.csv | 6720 | Ten validated Figure 8 building-seed-budget combinations | building_id + seed + adaptation_days + forecast_origin + horizon |
| Data S15 | Data_S15_figure8_case_metrics.csv | 10 | Recalculated Figure 8 case MAE, RMSE, and deltas | building_id + seed + adaptation_days |
| Data S16 | Data_S16_metric_recalculation_checks.csv | 9060 | Execution metric versus independent recalculation comparison | building_id + model + seed_key + days_key |
| Data S17 | Data_S17_source_checkpoint_hashes.csv | 3 | Frozen source checkpoint hashes | seed |

## Numbering decision

- Supplementary Methods: S1-S9.
- Supplementary Tables: S1-S10. Lettered A/B/C files are panels under one table number.
- Supplementary Figures: S1-S7. Every figure is stored as editable SVG, vector PDF, and 600 dpi PNG.
- Supplementary Data: S1-S17. These CSV files contain the filterable full-detail records behind the compact tables and figures.
