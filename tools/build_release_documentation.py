"""Build provenance/result indices from actual copied artifacts, without inference."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import assemble_release as assembly

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'research_stage1'

MAPPING=[
('Quality and numeric/zero audit','Methods 2.2; S2','scripts/audit_bdg2_electricity.py','timestamp_audit,value_audit','reports/bdg2_electricity_quality.csv','streamed timestamp/numeric/missing/zero/negative/longest-run inspection'),
('Zero-run review','Methods 2.2; S2','scripts/audit_bdg2_zero_values.py','','reports/bdg2_zero_value_audit.csv','zero ratios and longest zero intervals; no zero imputation'),
('Metadata linkage','Methods 2.2; Table 1','scripts/join_bdg2_metadata_quality.py','','reports/bdg2_metadata_quality_join.csv','building_id join; metering scope remains unresolved'),
('Candidate eligibility','Methods 2.2','scripts/review_bdg2_experiment_candidates.py','main','reports/bdg2_experiment_candidates.csv','candidate reports'),
('Site allocation','Table 1; S1','scripts/review_bdg2_cohort_split.py','deterministic_alternative','reports/bdg2_cohort_split_alternative.csv','historical quality/site balancing; preserve frozen alternative_group'),
('Source/development selection','Table 1; S1','scripts/prepare_bdg2_multibuilding_v2.py','eligibility,diversity_quality_select','reports/bdg2_multibuilding_cohort_manifest_v2.csv','20 source / 12 development, distinct sites and quotas'),
('Windows and temporal split','Figure 1/2; Methods 2.3','scripts/bdg2_multibuilding_common_v2.py','build_windows,source_windows,adaptation_windows,daily_test_windows','reports/bdg2_multibuilding_temporal_manifest_v2.csv','24-to-24 input/target structure; rolling adaptation versus daily evaluation'),
('C0 feature layout','Table 2; Methods 2.3','scripts/bdg2_stage6_c0_common_v3.py','make_c0_features,target_calendar_features','models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json','past 24 power with deterministic future-target calendar'),
('Source training/normalization','Table 2; Methods S4','scripts/run_bdg2_stage6_c0_lstm_v3.py','main,collect_source,fit_source_once,validation_loss','reports/bdg2_stage6_c0_training_history_v3.csv','source-only X/y normalization; MSE; Adam; 256 batch; 3 seeds'),
('Head-only target adaptation','Table 2; Methods S5','scripts/run_bdg2_stage6_c0_lstm_v3.py','adapt_head,adapt_head_once','reports/final_v1/gate2_run_config.json','same fixed source weights; min(64,N); 50 epochs; no adaptation at 0d'),
('Final cohort','Table 1; Methods S2','scripts/final_v1/build_final_v1_gate0.py','exclusion_reason,main','reports/final_v1_cohort_manifest.csv','453 retained/57 excluded; see GAPS missingness distinction'),
('Final temporal constructor','Figure 2; Methods S3','scripts/final_v1/build_final_v1_gate1.py','longest_complete_run,frozen_boundaries,validate_building_structure','reports/final_v1_temporal_manifest.csv','longest-run end rule; 28 nonoverlapping 24h targets'),
('Final baseline and neural execution','Table 3; Figure 4','scripts/final_v1/run_final_v1_gate2.py','main,daily_test_windows,weekly_prediction,prediction_frame','reports/final_v1/gate2_execution_metrics.csv','all budgets and seeds; full raw predictions external'),
('Independent MAE/RMSE','Table 3; Figure 4','scripts/final_v1/analyze_final_v1_gate3.py','recompute_metrics,compare_execution_metrics','reports/final_v1_metrics_by_building_seed_budget.csv','9060 metric rows; full prediction archive required for complete recomputation'),
('Site-equal aggregation','Table 3; Figure 4; S3','scripts/final_v1/analyze_final_v1_gate3.py','build_efficiency,build_curves','reports/final_v1/final_v1_site_equal_adaptation_curve.csv','seed/building/site hierarchy'),
('H1/H2/H3 bootstrap','Table 4; Figure 5; S4','scripts/final_v1/analyze_final_v1_gate3.py','bootstrap_ci,build_hypotheses','reports/final_v1_hypothesis_results.md','10000 site resamples, fixed seed; no added p values'),
('F1-F4 failure definitions','Table 5; Figure 6; S5','scripts/final_v1/analyze_final_v1_gate3.py','build_failures','reports/final_v1_failure_events.csv','EPS=1e-12; 6795 seed events; 2265 distinct F4 units'),
('Exploratory features','Table 6; S6','scripts/final_v1/analyze_final_v1_gate3.py','build_characteristics,characteristic_analysis','reports/final_v1/final_v1_characteristic_associations.csv','350 correlations; corrected missing usage grouping'),
('Representative-case rules','Figure 8; S7','scripts/final_v1/analyze_final_v1_gate3.py','choose_representative_cases','reports/final_v1/final_v1_representative_failure_cases.csv','prespecified Roger/Brooke/Gustavo selection rules'),
('Figure 8 extraction','Figure 8','figure8_extraction/extract_figure8_cases.py','','figure8_extraction/case_predictions.csv','10 cases,6720 rows; full archive external for extraction replay'),
('Main figures','Figures 1-7','publication/code/make_publication_figures.py','','publication/data/performance.csv','historical rounded presentation constants; not new metric computation'),
('Main tables','Tables 1-7','publication/code/make_tables.py','create','publication/data/tables.json','fixed manuscript table text with repaired module import'),
('Figure 8 renderer','Figure 8','publication/code/make_figure8_from_predictions.py','main','figure8_extraction/case_metrics.csv','saved numeric predictions; release preserves naive timestamp type'),
('Supplementary numerical outputs','Tables S1-S10 / Figures S1-S7','supplementary_final/code/build_supplementary_data_figures.py','prepare_full_data,build_tables,figure_s1,figure_s2,figure_s3,figure_s4,figure_s5,figure_s6,figure_s7','supplementary_final/data/Data_S3_all_model_metrics.csv','saved tables and vectors; Data S17 coverage gap retained'),
('Cross-device comparison','Table 7; Figure 3; S7','edge_deployment_final/e4_cross_device_validation/analyze_e4_outputs.py','metrics,main','edge_deployment_final/e4_cross_device_validation/validation_64_comparison_summary.json','1536 saved FP32 outputs compared'),
('Device inference','Figure 3','edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32/scripts/run_inference.py','load_bundle_model,main','edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32/config/model_config.json','standalone actual architecture; external frozen checkpoint'),
('Historical latency benchmark','Table 7; Figure 7; S8','edge_deployment_final/e5_performance_benchmark/benchmark_fp32_executed_v1.py','','edge_deployment_final/e5_performance_benchmark/jetson/latency_summary.json','200 warmups,5x1000; preserved executed version including RSS limitation'),
('Latency/telemetry analysis','Table 7; Figure 7; S8','edge_deployment_final/e5_performance_benchmark/analyze_e5_results.py','recalc_latency,compare_recorded','supplementary_final/data/Data_S11_edge_latency_observations.csv','20000 saved observations; keep outliers and power boundaries'),
]

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    # Complete helper inputs located during call-flow inspection.
    extra=['reports/bdg2_multibuilding_baseline_metrics_v2.csv','reports/bdg2_v5_analysis_definitions_frozen.md']
    extra += [p.relative_to(SOURCE).as_posix() for base in ['scripts/final_v1/addendum_pre_characteristic_fix','reports/final_v1/addendum_pre_characteristic_fix'] for p in (SOURCE/base).iterdir() if p.is_file() and p.suffix in {'.py','.md','.csv'}]
    for rel in extra:
        if not (ROOT/rel).exists():assembly.copy_file(rel)
    if assembly.ROWS:
        with (ROOT/'provenance/source_map.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(assembly.ROWS[0]));writer.writerows(assembly.ROWS)
    # This optional v5 helper also pins the now-sanitized source config.
    p=ROOT/'scripts/analyze_bdg2_v5_building_characteristics.py'
    config=ROOT/'models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json'
    text=p.read_text(encoding='utf-8').replace('9eccb94fcf2682e9a7a5ee205aa1ce250082dd6e7f7750337cb61ffed0ebe7d1',hashlib.sha256(config.read_bytes()).hexdigest())
    p.write_text(text,encoding='utf-8')
    cfg=json.loads(config.read_text(encoding='utf-8'))
    settings={k:cfg[k] for k in ['input_condition','feature_names','calendar_alignment','config','seeds','source_buildings','development_buildings','source_normalization','source_training_samples','source_validation_samples']}
    settings.update({'final_buildings':453,'final_sites':7,'adaptation_days':[0,1,3,7,14,28],'daily_test_windows':28,'target_hours_per_building':672,'bootstrap_repetitions':10000,'bootstrap_seed':20260925,'EPS':1e-12})
    (ROOT/'configs/scientific_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    runtime={'source_run_record':cfg['versions'],'source_device':cfg['device_name'],'jetson':{'torch':'2.11.0','cuda':'12.6','device':'Orin','compute_capability':'8.7'},'runtime_source':'source configuration and E4 report; no new hardware probe','portability_checks':'Existing environments used; no clean dependency installation performed'}
    (ROOT/'configs/runtime_record.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    external=json.loads((ROOT/'configs/external_artifacts.json').read_text())
    (ROOT/'configs/source_checkpoint_hashes.json').write_text(json.dumps([x for x in external if x['path'].endswith('.pt')],indent=2),encoding='utf-8')
    traced=[]
    for label,claim,code,functions,result,note in MAPPING:
        path=ROOT/code;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        defs={n.name:n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        lines=', '.join(f'{f}:{defs.get(f,"unverified")}' for f in functions.split(',') if f)
        traced.append({'component':label,'paper_result':claim,'script':code,'functions_lines_release':lines,'saved_result':result,'result_exists':(ROOT/result).exists(),'note':note})
    write_csv(ROOT/'RESULT_TRACEABILITY.csv',traced,list(traced[0]))
    md=['# Result-to-code traceability','','Paths are relative to the release root. Line numbers refer to the release copies. Original hashes are in `provenance/source_map.csv`.','','| Component / paper result | Script and functions | Existing result | Scope |','|---|---|---|---|']
    for x in traced:md.append(f'| {x["component"]} / {x["paper_result"]} | `{x["script"]}`; {x["functions_lines_release"]} | `{x["saved_result"]}` | {x["note"]} |')
    md += ['','The latest locally located integrated manuscript is the supplied Stage8 v2 Markdown in `docs/manuscript/`. A newer integrated full manuscript was not found. Main Table/Figure code is the v3 publication ZIP; Supplementary is the completed local `supplementary_final` version. This distinction is retained rather than treating all names as one version.']
    (ROOT/'RESULT_TRACEABILITY.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    results=[]
    for folder in ['reports','publication','supplementary_final','figure8_extraction','edge_deployment_final']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or p.suffix not in {'.csv','.json','.npy','.png','.pdf','.svg'}:continue
            rel=p.relative_to(ROOT).as_posix();n=''
            if p.suffix=='.csv':
                with p.open(encoding='utf-8-sig',newline='') as f:n=max(0,sum(1 for _ in csv.reader(f))-1)
            direct=[x['script'] for x in traced if x['saved_result']==rel]
            if not direct:
                direct=['supplementary_final/code/build_supplementary_data_figures.py'] if rel.startswith('supplementary_final/') else ['publication/code/make_publication_figures.py or make_tables.py; see traceability'] if rel.startswith('publication/') else ['see RESULT_TRACEABILITY.md and historical report; input/evidence artifact may have no single generator']
            results.append({'path':rel,'size_bytes':p.stat().st_size,'rows_excluding_header':n,'generator_or_role':'; '.join(direct),'included_status':'existing result/evidence, not newly generated by release'})
    write_csv(ROOT/'RESULT_FILES.csv',results,list(results[0]))
    print(f'{len(traced)} component mappings; {len(results)} saved result files indexed.')

if __name__=='__main__':main()
