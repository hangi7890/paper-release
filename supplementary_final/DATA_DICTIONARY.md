# Supplementary Data Dictionary

All files are UTF-8 CSV files. Numeric values retain the source precision written by the frozen analyses. Boolean columns are stored as `True` or `False`. Empty cells denote unavailable or inapplicable values and are not silently replaced by zero.

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

## Interpretation controls

- Data S3 includes two deterministic baselines and six neural budgets. Baseline rows have no neural seed or adaptation budget.
- Data S6 contains F1-F3 events at the building × budget × seed level. Data S7 contains F4 once per building × budget and prevents the threefold duplication that would result from counting seed rows.
- Data S8-S10 use the corrected post-addendum outputs. The 18 Swan buildings with missing `primaryspaceusage` are retained as `<missing>` in group summaries.
- Data S11 retains all latency observations, including the Jetson end-to-end maximum of 195.884742 ms.
- Data S12 keeps Jetson VDD_IN and RTX GPU board power as separate measurement domains. It must not be used to form a cross-device power ratio.
