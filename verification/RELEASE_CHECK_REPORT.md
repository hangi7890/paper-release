# Release verification report

## Completed checks

- 48 Python files parsed successfully; static local-import closure contains no unresolved module.
- The three copied final Gate synthetic scripts passed: 12 Gate 1 checks, 15 Gate 2 checks and 15 Gate 3 checks. They use synthetic numeric records, not final-site electricity.
- Core release smoke: 36 recorded checks PASS. Includes actual quality-audit functions on 49 synthetic hourly rows, C0 feature order, normalization and static mean-MSE/head-only/batch rules.
- Saved Figure 8: 6,720 rows, 10 combinations, 672 values per combination; recalculated MAE/RMSE maximum absolute difference from saved metrics was 5.68e-14.
- All F1-F4 counts match the existing results; all six site-equal adaptation summaries match with maximum numeric difference 0.
- H1/H2/H3 effect estimates and the original seven-site analysis were reproduced from the saved metric table. This is arithmetic verification, not another final model run.
- All 350 corrected exploratory association rows match, including missing-data statuses/complete-case counts.
- The 64 saved RTX/Jetson validation vectors pass normalized and physical criteria. All 20,000 latency rows reproduce recorded run/overall statistics with maximum difference 0.
- Figure smoke: 12 checks PASS. Main MAE/RMSE/delta display constants agree at their written rounding precision; all seven failure-count series agree exactly. Figure 8 renders in PNG/SVG/PDF from the existing case CSV in a temporary directory.
- Ancillary smoke: 4 checks PASS. New isolated workspace preparation works; the main table builder imports; Supplementary tables reconstruct from saved artifacts without checkpoint binaries; the existing one-row Data S17 gap remains explicitly documented.
- 28 research components have code/result traceability rows, all pointing to existing files. 186 saved result/evidence files are indexed in RESULT_FILES.csv.
- Source preservation: 272 source-file/archive-member/excluded-input entries retain their independently measured pre-copy hashes. Raw data and checkpoints were hashed, not interpreted or executed.
- Privacy/package scan: no tested private path, private SSH key, common credential token or private address patterns found; no checkpoints, full raw-data table, full prediction archive or >50 MB file included. This is a targeted automated scan, not an exhaustive privacy guarantee.

## Environments and commands

Core checks used the existing Python 3.12.14 / PyTorch 2.11.0+cu128 / pandas 3.0.1 / NumPy 2.3.5 environment with CUDA hidden. Plot checks used the existing Python 3.13.9 analysis environment. No package installation or clean combined environment test was performed.

```text
python tools/smoke_check.py --core
python tools/smoke_check.py --figures
python tools/smoke_check.py --ancillary
python tools/audit_release.py --check-originals
```

The ancillary import check used the existing python-docx library through a temporary module-search path; normal users should install requirements.txt in their chosen environment. All tests run without training, adaptation, checkpoint loading, model forward execution or new hardware measurements. Temporary plots and reconstructed tables are verification output only and were not installed as paper results.

## Error and repair record

The first release-owned smoke runner stopped while reading a Korean-containing Python file with the Windows default CP949 decoder. Explicit UTF-8 reading fixed the runner, and the second core run passed. The first log/status is retained as `core_smoke_attempt1.*`. No experiment code or numeric output was modified for that failure.

The missing publication import and unsupported UTC labeling were repaired only in release copies; the details are in GAPS.md and provenance/portability_changes.json. A later documentation pass also updated the optional v5 helper's sanitized configuration hash. No scientific parameter, loss, optimizer, epoch limit, seed, final-building selection or time boundary was altered.

## Not claimed

No full 6,088,320-row prediction recomputation, source retraining, target adaptation, full-study inference, Jetson benchmark, Git commit or public push was performed during packaging. No connected Git repository was present. Original artifacts remain in the research folder. Author decisions on GAPS.md, licensing and public artifact hosting remain required.
