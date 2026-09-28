# Buildings paper: BDG2 C0 transfer learning release

This directory assembles the actual code and preserved results for the 20-source / 12-development / 453-final-building C0 study. The final evaluation uses seven sites, three seeds (20260924, 20260925, 20260926), six adaptation budgets (0, 1, 3, 7, 14, 28 days), and 28 daily forecasts per target building. No research experiment was rerun to create this release.

Read **GAPS.md before citing or submitting**. In particular, the manuscript's blanket missingness condition does not describe final Gate 0: 31 retained final buildings have missingness above 20%. Original evidence is preserved; the scientific selection rule has not been changed.

## Contents and authority

| Directory | Purpose |
|---|---|
| `scripts/` | Historical quality audit, deterministic selection, windows, source training and C0 feature code |
| `scripts/final_v1/` | Actual final Gate 0–3 implementation, corrected exploratory analysis, original synthetic tests |
| `models/bdg2_stage6_v3/` | Sanitized original run configuration only; no weights |
| `reports/` | Frozen manifests and existing compact final/development result tables and reports |
| `figure8_extraction/` | Original streaming extractor and validated 6,720-row case subset |
| `publication/` | Supplied main Tables 1–7 / Figures 1–8 and plotting scripts |
| `supplementary_final/` | Methods S1–S9, Tables S1–S10, Figures S1–S7, Data S1–S17 and numeric plotting code |
| `edge_deployment_final/` | Portable model/preprocessing code, fixed validation vectors, saved RTX/Jetson outputs, E4/E5 analyses and benchmark implementations |
| `configs/` | Measured external input hashes and historical settings/runtime |
| `provenance/` | Original-to-release mapping, exact source/release hashes, portability changes and preserved-source check |
| `verification/` | Small smoke checks and their logs; these are not new scientific results |
| `tools/` | Isolated workspace preparation, input verification, bounded checks and release inventory |

`RESULT_TRACEABILITY.md` maps claims to functions and saved outputs. `RESULT_FILES.csv` lists existing result files, their rows and generating scripts. All release hashes are calculated from real files; original hashes and changed release hashes are labeled separately. Historical hash strings inside reports remain historical records.

## Environment

Python 3.12 is recommended because it matches the frozen source run. Use a fresh virtual environment and install `requirements.txt`; install the appropriate **official PyTorch build** separately before importing the training/final Gate 2 modules. `requirements-training.txt` records scientific package versions but does not select an NVIDIA Jetson wheel or install CUDA/cuDSS. Historical runtime details are in `configs/runtime_record.json` and the E4/E5 reports. No packages are installed automatically.

```bash
python -m venv .venv
# Activate your platform's virtual environment.
python -m pip install -r requirements.txt
# Select the official PyTorch wheel for your device; use the historical versions for strict replication.
python tools/smoke_check.py --core
python tools/smoke_check.py --figures
python tools/smoke_check.py --ancillary
```

The release audit used the existing training environment for core checks and the existing analysis environment for plotting checks. The optional table-import check reused the already installed pure-Python python-docx package. It did not claim that one newly installed combined environment was tested. Full training and hardware benchmarks were not rerun. Archived reports inside subfolders retain historical references to Word/ZIP/private authoring artifacts not shipped here; this top-level README and GAPS.md describe the actual release contents.

## External data and excluded artifacts

Obtain BDG2 from the official repository: https://github.com/buds-lab/building-data-genome-project-2

- Electricity file: https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/electricity.csv
- Metadata file: https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/metadata/metadata.csv
- Documentation: https://github.com/buds-lab/building-data-genome-project-2/wiki
- Review upstream data licensing/attribution before redistributing derivatives; the retained source-review report documents the prior review. No code license is assigned without the authors' decision.

Use the actual CSV payload (not a Git LFS pointer). Place it in an isolated replay workspace:

```text
data/raw/bdg2/electricity.csv  # timestamp + 1,578 building-ID columns, 17,544 rows
data/raw/bdg2/metadata.csv     # building_id/site_id/primaryspaceusage/sqm/timezone and official metadata
models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260924_v3.pt
models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260925_v3.pt
models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_seed_20260926_v3.pt
reports/final_v1_raw_predictions.csv  # optional author archive for exact 6,088,320-row metric replay/extraction
```

The checkpoints and full prediction archive have no public URL established by this release. They must be obtained from the authors for exact frozen replay; source training can produce new checkpoints, but their hashes and predictions are not guaranteed to reproduce the historical CUDA run bit for bit. The 6,720 saved Figure 8 case rows and all compact metrics are included. No interpolation, rounding, or fresh prediction was applied to that subset.

Verify supplied inputs with `python tools/check_external_inputs.py --workspace ../replay --kind all`. Expected hashes were independently recomputed during packaging. No final electricity values were inspected during this packaging task.

## Isolated execution and order

Never execute result-producing scripts against the packaged evidence directory. Create a new workspace with `tools/prepare_workspace.py`. It copies code and appropriate inputs into a new directory and omits the selected stage's output files; it refuses an existing destination. Commands below are instructions for a future reproduction, **not experiments run during release assembly**. Model scripts can be expensive.

1. **Quality audit and preprocessing**

```bash
python tools/prepare_workspace.py --dest ../replay-quality --mode quality
cd ../replay-quality
# Supply the two raw BDG2 CSVs first.
python scripts/audit_bdg2_electricity.py
python scripts/audit_bdg2_zero_values.py
python scripts/join_bdg2_metadata_quality.py
```

No imputation, zero removal, resampling or timezone conversion belongs to the C0 pipeline. `review_bdg2_experiment_candidates.py`, `prepare_bdg2_cohort_review.py`, and `review_bdg2_cohort_split.py` contain the historical candidate/site construction. For paper replication use the **included frozen alternative_group manifest**; do not optimize a new split. `prepare_bdg2_multibuilding_v2.py` defines source/development quotas and their temporal constructor. The older four-building `build_bdg2_temporal_splits.py` is included for provenance but is not the final temporal rule. Final Gate 1 uses the end of the longest valid run.

2. **Source C0 training and development adaptation**

```bash
python tools/prepare_workspace.py --dest ../replay-source --mode source
cd ../replay-source
# Supply raw data. Frozen cohort/temporal/C0 protocol manifests are already included.
python scripts/run_bdg2_stage6_c0_lstm_v3.py
```

This historical driver trains all three source seeds and runs development source-only/head-only conditions. Source normalization is computed from flattened source-training X and y, including overlapping windows; target validation/test data do not enter those statistics. See `configs/scientific_settings.json`. It is not executed by the smoke tests.

3. **Final protocol and fixed evaluation**

Gate 0 selection and Gate 1 constructors are in `scripts/final_v1/build_final_v1_gate0.py` and `build_final_v1_gate1.py`. The archived outputs, including every exclusion, are supplied. Recreating timestamped protocol documents may not reproduce historical bytes. To replay the frozen final evaluation, retain the supplied frozen cohort/temporal/protocol files and use the exact external source weights:

```bash
python tools/prepare_workspace.py --dest ../replay-final --mode final
cd ../replay-final
python tools/check_external_inputs.py --kind data
python tools/check_external_inputs.py --kind checkpoints
python scripts/final_v1/run_final_v1_gate2.py
```

Gate 2 runs previous-day, weekly-same-hour, and all neural budgets/seeds. It refuses repeat execution and requires CUDA. Raw/weight identity and scientific settings remain strictly checked. Release-only path sanitization changed three metadata file byte hashes; their new, actually measured hashes are pinned in the copied Gate 2 code and documented in `provenance/portability_changes.json`. These are not replacement hashes for raw data or weights. New source-training outputs will not silently bypass the frozen replay guards.

4. **Independent metrics, failure analysis and exploratory characteristics**

```bash
python tools/prepare_workspace.py --dest ../replay-analysis --mode analysis
cd ../replay-analysis
# Supply original raw predictions and electricity; execution metrics/config are included.
python scripts/final_v1/analyze_final_v1_gate3.py
```

This independently recalculates MAE/RMSE, site-equal curves, H1/H2/H3, EPS-based F1–F4, representative cases, and pre-test features. The current function includes the documented Swan missing-category correction. The original correction script is retained for provenance and must not be rerun on already corrected results. All six budgets are reported; F4 uses building-budget rather than seed-event denominators.

5. **Paper tables, figures and Figure 8**

Create a `snapshot` workspace, then run:

```bash
python publication/code/make_publication_figures.py
python publication/code/make_tables.py
python publication/code/make_figure8_from_predictions.py figure8_extraction/case_predictions.csv --out outputs/figure8
python supplementary_final/code/build_supplementary_data_figures.py
```

Main publication scripts retain historical rounded display constants; they are **presentation code, not metric estimators**. Their values are checked against saved numerical results by the release smoke check. Figure 8 is read directly from saved predictions. Full-case re-extraction uses a `figure8` workspace plus the original full prediction archive, then `python figure8_extraction/extract_figure8_cases.py`. Supplementary plotting uses complete saved result tables. Documentation/Word-authoring helpers requiring the private handoff archive were deliberately omitted.

6. **RTX–Jetson FP32 and latency**

See `EDGE_REPRODUCTION.md`. Included functions can independently compare saved device outputs and latency rows without a GPU. Actual device inference requires the external checkpoint and compatible runtime; local SSH aliases, account details, installers, and remote deployment automation were excluded. Both `benchmark_fp32_executed_v1.py` (actual historical run) and the later collector-repaired `benchmark_fp32.py` are labeled; they must not be confused.

## Public release decision

No Git repository/remote was connected in the inspected working project, so no commit or public push was performed. Before upload, authors should approve the code/data license, data-derived fixture redistribution, manuscript draft inclusion, and the manuscript corrections in GAPS.md. Inspect `verification/RELEASE_CHECK_REPORT.md` and `provenance/release_manifest.csv`. The package contains no credentials, full raw BDG2 tables or checkpoints. Automated scans are evidence of the patterns tested, not a guarantee about all possible sensitive content.
