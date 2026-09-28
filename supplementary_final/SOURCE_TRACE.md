# Source Trace

## Source files and transformations

| source | path | size_bytes | sha256 | used_fields | use |
|---|---|---|---|---|---|
| Final raw predictions | .\reports\final_v1_raw_predictions.csv | 1216425185 | 72732c5dc0b0ad9cb7440f326476cd8c2ae57dd9267f7e07398452ae5b636e43 | not reread; recorded hash/rows and validated Figure 8 subset only | authoritative provenance; full results use preserved aggregate tables |
| Final cohort | .\reports\final_v1_cohort_manifest.csv | 141297 | 8216847545f7814c12f3e2edb31523ac11da0ae19cc142985f736bbd7185d3ec | building/site/use/quality/status/reason | Data S1, Table S1 |
| Final temporal split | .\reports\final_v1_temporal_manifest.csv | 1187861 | 1cb76d4cc7cef687fc747b2e0015112d85aec4e0f9e76271b28284f54421bc08 | time boundaries, counts, digests, status | Data S2, Methods S3 |
| Final metrics | .\reports\final_v1_metrics_by_building_seed_budget.csv | 2253122 | 8540b3280a2e97935f36d9c03a3e8fc5c1de2901184fff774cae9a0e0dd9ad2d | building/site/model/seed/budget/MAE/RMSE | Data S3 and compact performance tables |
| Data efficiency | .\reports\final_v1_data_efficiency.csv | 3034742 | a0cff65891e839c3c86e7a88e19865f3d856ec526c788472e7a8fc02a770f914 | deltas, baselines, A/B flags | Data S4-S5 and Figures S2/S7 |
| Failure events | .\reports\final_v1_failure_events.csv | 2914693 | 2875ae251fb07f9d46bd7d13c9ba239642c1a79d2efa225713cdfdcf53d453b9 | F1-F4 and metric discordance | Data S6-S7, Table S5, Figure S3 |
| Corrected associations | .\reports\final_v1\final_v1_characteristic_associations.csv | 37560 | 823f4fc353902d2536555203e74b337ace7f83023346c5dcb1d4adb0a9cec627 | feature/response/budget/n/rho/status | Data S8, Table S6, Figure S4 |
| Corrected characteristic groups | .\reports\final_v1\final_v1_characteristic_group_summary.csv | 11447 | 27817ff1add9f5318d6da13cde6eff405138058393b7a9f8d3ede56bf923a6a4 | site/use groups and rates | Data S9 and Swan correction check |
| Corrected characteristics | .\reports\final_v1\final_v1_building_characteristics.csv | 1602434 | 6f7312c06157b9566db604b287b227b9092695d6ae02911f0c9d378d85bbabcb | pre-test characteristics and cutoffs | Data S10 |
| Stage-6 audit | .\reports\final_v1\final_stage6_training_config_audit.json | 7615 | c2bb57bd076791d1db5bc4cdde3d7af260251d99599463697105e5e083fe1bb1 | loss, batches, shuffle, drop_last, head parameters, hashes | Methods S4-S5, Table S2 |
| E4 validation summary | .\edge_deployment_final\e4_cross_device_validation\validation_64_comparison_summary.json | 3526 | aef172d9957c860c89c53fc602b3cd8b7c2cb3b6e1d6c7ca86633daf8b668cb1 | criteria and normalized/physical differences | Table S7 |
| E5 Jetson latency | .\edge_deployment_final\e5_performance_benchmark\jetson\latency_model_only_raw.csv | 128861 | 64ac0ff62f5c4f638aa5f7d36e8bddb8698316fc0313c0f589c66e96b605a34d | run/repetition/latency | Data S11, Figure S5 |
| E5 Jetson end-to-end latency | .\edge_deployment_final\e5_performance_benchmark\jetson\latency_end_to_end_raw.csv | 79031 | 2d02ece7abeb8bfc39f1ab6895d91a19cd4a3fd55aff7d237181fd4bf6ae8249 | run/repetition/latency | Data S11, Figure S5 |
| E5 RTX latency | .\edge_deployment_final\e5_performance_benchmark\rtx3080\latency_model_only_raw.csv | 133043 | ae6f96fd4d9e95a9384b6abb11bec63c8d856a02920f85f61caa0390e9fcc8c3 | run/repetition/latency | Data S11, Figure S5 |
| E5 RTX end-to-end latency | .\edge_deployment_final\e5_performance_benchmark\rtx3080\latency_end_to_end_raw.csv | 68906 | 57b439ea26bf88551b66235dc5f4bde94626cf98061be1283b7cb3a9bff4be8b | run/repetition/latency | Data S11, Figure S5 |
| Figure 8 predictions | .\figure8_extraction\case_predictions.csv | 709012 | 5cdab6287bc60f0bcc40f0f5e810114fc1c1ad955629f72f80522f9bcf449537 | 10 cases, timestamps, horizons, truth, prediction | Data S14, Table S10 |
| Figure 8 metrics | .\figure8_extraction\case_metrics.csv | 1501 | ca45fd1dbed241a8b2f5de4e5f4f9941716eb351bfc3b2f08dfd60d30504d677 | MAE/RMSE/source/deltas/status | Data S15, Table S10 |

## Reproduction commands

Run from the `research_stage1` project root with the existing Anaconda Python environment:

```powershell
python supplementary_final\code\build_supplementary_data_figures.py
python supplementary_final\code\compose_supplementary_markdown.py
python supplementary_final\code\build_supplementary_docx.py
python supplementary_final\code\build_supplementary_manifest.py
python supplementary_final\code\verify_supplementary_package.py
```

The document builder and render commands are recorded in `qa/docx_build_and_render_commands.txt`. None of these commands trains a model or regenerates final predictions.

## Aggregation controls

- Building-level neural result rows: 453 buildings × 3 seeds × 6 budgets = 8,154.
- Site × budget × seed summary rows: 7 × 6 × 3 = 126.
- F1-F3 event rows: 453 × 5 nonzero budgets × 3 seeds = 6,795.
- F4 building-budget rows: 453 × 5 = 2,265.
- Final target values per building-condition: 28 origins × 24 horizons = 672.
- E5 latency observations: 2 devices × 2 modes × 5 runs × 1,000 = 20,000.
- E4 validation elements: 64 vectors × 24 horizons = 1,536.

## Known unavailable or unperformed items

- RTX process RSS during E5 is unavailable because the Windows collector failed. A later probe is not substituted.
- Jetson `jetson_clocks` state was not confirmed because `--show` required root; E5 did not enable or change it.
- Jetson VDD_IN and RTX board power have different boundaries; no cross-device power or energy ratio is computed.
- TensorRT, ONNX, FP16, INT8, pruning, quantization, C1 weather, indoor sensing, and occupancy inputs were not executed in the final scope.
- No required Supplementary source file was missing. `MISSING_SOURCE_DATA.md` was therefore not created.
