# FP32 deployment and benchmark reproduction

The included saved-output checks require no device execution. Run `python tools/smoke_check.py --core` for the E4 64-vector normalized/physical consistency and all 20,000 E5 latency observations. This invokes the original comparison/statistic functions on saved arrays/CSVs.

## Inputs and exact model

Bundle path: `edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32/`.
It contains the actual `scripts/model_definition.py`, `preprocessing.py`, `run_inference.py`, `verify_output.py`, three model/feature/preprocessing configs, numerical criteria and small frozen validation vectors/reference arrays. The external checkpoint must be placed under `model/` with the filename in `config/model_config.json`; its hash is verified before loading.

Architecture: FP32 one-layer C0 LSTM, input (*,24,6), hidden 32, output (*,24), 5,912 parameters. The deployment seed is 20260924; this engineering validation is not a new forecasting hypothesis. PyTorch checkpoint loading is only for the exact trusted author checkpoint, never arbitrary downloaded pickle files.

## Future device execution (not executed during release preparation)

Copy the portable bundle to each device and provide the exact external checkpoint. Run from the bundle root, choosing a new output directory:

```bash
python scripts/run_inference.py --input validation/validation_input_tensor.npy --output-dir outputs/validation_cuda --device cuda
```

Compare saved normalized and physical outputs with `scripts/verify_output.py --help` or `analyze_e4_outputs.py:metrics`. Acceptance: shape exact, finite values, normalized maximum/mean absolute difference <=1e-4/2e-5 and physical maximum/mean <=0.05/0.01. Fixed 64 vectors contain 1,536 output values.

Actual historical benchmark entry point:

```bash
python edge_deployment_final/e5_performance_benchmark/benchmark_fp32_executed_v1.py --help
python edge_deployment_final/e5_performance_benchmark/benchmark_fp32_executed_v1.py consistency --bundle-root PATH_TO_BUNDLE --output NEW_PRE_JSON --device cuda
python edge_deployment_final/e5_performance_benchmark/benchmark_fp32_executed_v1.py latency --bundle-root PATH_TO_BUNDLE --output-dir NEW_LATENCY_DIR --device cuda
python edge_deployment_final/e5_performance_benchmark/benchmark_fp32_executed_v1.py consistency --bundle-root PATH_TO_BUNDLE --output NEW_POST_JSON --device cuda
```

The latency subcommand fixes 200 warmups and five runs of 1,000 measurements, batch 1, model-only CUDA-event timing and synchronized end-to-end host/device transfers. Do not replace the original recorded outputs with new runs. A new runtime/device run must have its own report/version. The newer `benchmark_fp32.py` repairs the process-memory collector only; using it must be labeled as a new run, not the original E5 execution.

Installation, SSH credentials and device addresses are intentionally not shipped. Jetson CUDA/cuDSS compatibility is documented in the historical reports; environment setup must be performed for the actual deployment host. The original E4 remote orchestration was machine/account-specific and is omitted; the model execution and comparison functions it called are included.
