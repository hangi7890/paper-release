# Confirmed gaps, discrepancies and release limits

This file reports observed differences. No experimental setting or result was changed to match manuscript prose.

## G1 — Final eligibility does not impose the source/development 20% missingness cutoff

`docs/manuscript/Stage8_Integrated_Manuscript_v2.md`, section 2.2, describes candidate records as having no more than 20% missing values and no negative values. `scripts/prepare_bdg2_multibuilding_v2.py:eligibility` applies these extra conditions to source/development candidates. `scripts/final_v1/build_final_v1_gate0.py:exclusion_reason/main` applies preliminary eligibility, >=90d run, nonempty/non-all-zero values and no >=30d zero run, without adding the 20% cutoff. The retained final manifest actually contains **31 buildings with missingness >20%**. Correct the manuscript's wording to distinguish source/development rules and final rules. Do not remove these buildings or change frozen results after evaluation.

## G2 — Main figure/table builders are presentation scripts with rounded constants

The supplied v3 `publication/code/make_publication_figures.py` hardcodes rounded MAE/RMSE, event counts, association excerpts, site effects and latency summaries. It does not calculate metrics from raw predictions. `make_tables.py` combines these with fixed narrative/table content. These are included as the actual existing presentation implementation. The release checks MAE/RMSE/deltas to the original display precision and all F1-F4 counts against saved results. The full statistical chain remains Gate 2 -> Gate 3 -> numerical CSVs. No false claim of fully automatic raw-data-to-typeset-manuscript provenance is made.

The table builder imported `make_figure_table_package`, which is absent from the v3 ZIP. Its release copy now imports `make_publication_figures`, which supplies the same requested constants. Import-time CSV writes were guarded; numerical constants were not changed. Some table narrative and presentation excerpts remain manually specified and require author review before a new dataset/version is used.

## G3 — Historical Figure 8 renderer added UTC to naive timestamps

The supplied renderer used `pd.to_datetime(..., utc=True)`, even though extracted timestamp strings have no offset. The release copy parses without `utc=True`. Stored prediction CSVs are byte-preserved, numeric data are unchanged and plotting uses relative day indices. The source extractor, raw values and original figure artifacts remain traceable. Do not claim physical UTC alignment from this fix.

## G4 — Supplementary Data S17 contains one checkpoint, not three

The existing `supplementary_final/data/Data_S17_source_checkpoint_hashes.csv` has only seed 20260924. Its generator takes rows from the Stage-6 audit's `audited_execution_files`, which lists that checkpoint. The existing Supplementary documentation describes three. This is an inventory/documentation gap, not evidence that only one seed was run: the original source run configuration and final metrics contain all three seeds. The release preserves existing Data S17 and supplies separately **`configs/source_checkpoint_hashes.json` with all three real checkpoint sizes/hashes**, recomputed from the actual excluded checkpoint files. Authors should correct the Supplementary inventory claim or approve an explicit addendum; no old result was silently overwritten.

## G5 — Public inputs required for full historical replay

Full BDG2 raw data, model/adaptation checkpoints and the 6,088,320-row prediction CSV are excluded. The raw CSV has an official upstream source, but no public download URL for the exact frozen weights/full prediction archive was found locally. Consequently, this public folder can verify compact results, saved Figure 8 cases and device outputs immediately; full historical inference/metric replay needs author-supplied artifacts. Retraining source weights may produce different bytes across environments and is not presented as exact frozen replay.

## G6 — Path-bearing frozen metadata hashes differ after redaction

Original configurations/reports contain private absolute filesystem paths. Release copies use relative/redacted paths. These metadata byte hashes differ; original and copied hashes are separately preserved. The release Gate 2 and optional v5 helper pin the sanitized source metadata bytes, while the raw-electricity and checkpoint guards retain their exact original hashes. Timestamped protocol regeneration may not yield the original protocol digest. Use supplied frozen manifests for frozen replay; do not bypass guards or substitute results.

## G7 — Historical versus corrected exploratory/edge code

Gate 3 has a documented missing-category correction preserving 18 Swan buildings. The corrected implementation/results and the pre-correction addendum artifacts are both included and labeled. Predictions, H1-H3 and F1-F4 are not regenerated.

E5 `benchmark_fp32_executed_v1.py` is the actual executed benchmark. `benchmark_fp32.py` contains the later Windows RSS collector fix. Historical RTX process RSS is unavailable; a fresh collector probe cannot fill it retroactively. Jetson `jetson_clocks` state was unverified. Jetson VDD_IN and RTX board power are different boundaries; no comparable energy-per-inference result is supplied. No new E4/E5 device run was performed for release preparation.

## G8 — Bounded portability coverage and publication decisions

Core smoke checks use the existing PyTorch experiment environment; rendering checks use the existing Matplotlib environment. No clean combined install, full training, all-building inference or new Jetson benchmark was attempted. Synthetic tests and saved-result checks cannot guarantee cross-platform GPU reproducibility. OOM fallback code is retained exactly (source 256 ->128 ->64; adaptation 64 ->32 ->16), although the recorded completed run used the first candidates.

The author must choose a code license and confirm upstream data-derived fixture licensing/attribution before GitHub publication. No permissive license is invented. Private SSH automation, install scripts, credentials, virtual environments, checkpoint binaries, unrelated NIST/HouseZero/weather/Transformer experiments and document-authoring helpers are excluded. No Git remote was connected locally, and no public push was performed.
