# Manuscript Supplementary Cross-References

The source manuscript `Stage8_Integrated_Manuscript_v2.md` still contains the older Figure 1-8 and Table 1-5 placeholders. The frozen v3 structure defines eight main figures and seven main tables. The source manuscript is not edited here. The following sentences are proposed insertions for the next manuscript revision.

## Proposed first citations

| main section | insertion anchor | supplementary citation | proposed sentence |
|---|---|---|---|
| 2.2 Dataset and Building Cohort | After the paragraph ending with the meter-scope limitation | Supplementary Methods S2; Table S1; Data S1 | Detailed building eligibility, site retention, and exclusion reasons are provided in Supplementary Methods S2, Table S1, and Data S1. |
| 2.3 Forecasting Task and Input Representation | After the 24-hour input/output definition | Supplementary Methods S3-S4; Data S2 | The exact temporal boundaries, continuity checks, and C0 feature equations are given in Supplementary Methods S3-S4 and Data S2. |
| 2.5 Target Adaptation and Data Budgets | After the architecture/adaptation paragraph | Supplementary Methods S5; Table S2; Figure S1 | The actual supervised-window counts, DataLoader batch rule, and partial-batch behavior are documented in Supplementary Methods S5, Table S2, and Figure S1. |
| 2.6-2.8 Metrics and Hypotheses | After the site-equal/bootstrap definitions | Supplementary Methods S6; Tables S3-S4; Data S3-S5 | Complete building-, seed-, budget-, and site-level results are available in Supplementary Methods S6, Tables S3-S4, and Data S3-S5. |
| 3.4 Adaptation Failure Analysis | After the sentence ending 'not always identical across the two error metrics.' | Table S5; Figure S3; Data S6-S7 | Complete event counts, denominators, and building-level failure records are reported in Table S5, Figure S3, and Data S6-S7. |
| 3.5 Exploratory Building-Characteristic Analysis | After the Swan correction paragraph | Supplementary Methods S7; Table S6; Figure S4; Data S8-S10 | The full corrected exploratory matrix and complete-case counts are provided in Supplementary Methods S7, Table S6, Figure S4, and Data S8-S10. |
| 3.6 Edge Deployment Results | After the RTX RSS limitation sentence | Supplementary Methods S8; Tables S7-S8; Figures S5-S6; Data S11-S13 | Cross-device numerical details, all latency observations, and device-specific telemetry are provided in Supplementary Methods S8, Tables S7-S8, Figures S5-S6, and Data S11-S13. |
| 3.6 Final integrity transition | After main Table 7 | Table S9; Data S16-S17 | Full artifact hashes and independent metric-recalculation records are reported in Supplementary Table S9 and Data S16-S17. |
| 4.3 Failure interpretation | After the first paragraph and main Figure 8 citation | Supplementary Methods S9; Table S10; Figure S7; Data S14-S15 | The frozen selection rules, complete budget-by-seed patterns, and validated case records are provided in Supplementary Methods S9, Table S10, Figure S7, and Data S14-S15. |

## v2-to-v3 numbering differences

| v2 element | frozen v3 element | status |
|---|---|---|
| v2 Figure 1 | v3 Figure 1 | protocol retained |
| v2 Figure 2 | Methods 2.7 + v3 Table 5 notes | standalone taxonomy figure removed |
| v2 Figure 3 | v3 Figure 7a | edge workflow integrated |
| v2 Figure 4 | v3 Figure 3 | adaptation curve |
| v2 Figure 5 | v3 Figure 4 | H1-H3 |
| v2 Figure 6 | v3 Figure 5 | failure rates |
| v2 Figure 7 | v3 Figure 6 | exploratory associations |
| v2 Figure 8 | v3 Figure 7c-d | edge latency; representative cases become v3 Figure 8 |
| v2 Table 1 + Table 2 | v3 Table 1 | cohort summary consolidated |
| v2 unnumbered configuration | v3 Table 2 | configuration promoted |
| v2 Table 3 | v3 Table 3 | performance retained |
| v2 Results 3.3 text | v3 Table 4 | hypothesis effects tabulated |
| v2 Table 4 | v3 Table 5 | failure counts |
| v2 Table 5 | v3 Table 6 | edge benchmark |
| new integrity table | v3 Table 7 | full hashes remain Supplementary Table S9 |

## Cross-reference QA

- Main figures and tables are not renumbered inside this Supplementary package.
- Methods, Tables, Figures, and Data each start at S1 and advance independently.
- Figure S7 supplements main Figure 8 with all budgets and seeds. It does not replace or duplicate the 672-hour main panels.
- Table S5 preserves the distinct statistical units: F1-F3 are building × budget × seed; F4 is building × budget.
- H2/H3 remain effect-size analyses with site-cluster intervals and no p-value or Holm procedure.
