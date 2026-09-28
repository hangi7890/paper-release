"""Make an isolated, explicitly chosen replay workspace; never execute models."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--mode',choices=['snapshot','quality','source','final','analysis','figure8','edge-analysis'],default='snapshot')
    args=ap.parse_args(); dest=args.dest.resolve()
    if dest.exists(): raise FileExistsError('Choose a new workspace; existing directories are never replaced')
    if dest==ROOT or ROOT in dest.parents: raise ValueError('Workspace must be outside the release directory')
    excluded=set()
    if args.mode=='quality':
        excluded={str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'reports').glob('*') if p.is_file() and p.name.startswith(('bdg2_electricity_quality','bdg2_zero_value_audit','bdg2_metadata_quality_join','bdg2_meter_scope_review'))}
    if args.mode=='source':
        excluded|={str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'reports').glob('bdg2_stage6_c0_*') if p.is_file() and p.name!='bdg2_stage6_c0_protocol_manifest_v3.csv'}
        excluded.add('models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json')
    if args.mode=='final':
        excluded|={'reports/final_v1/'+n for n in ['gate2_execution_metrics.csv','gate2_failures.csv','gate2_run_config.json','gate2_execution.log','GATE2_STARTED.lock','GATE2_COMPLETE.json','raw_predictions.partial.csv']}
        excluded.add('reports/final_v1_raw_predictions.csv')
    if args.mode=='analysis':
        excluded|={'reports/'+n for n in ['final_v1_metrics_by_building_seed_budget.csv','final_v1_data_efficiency.csv','final_v1_failure_events.csv','final_v1_failure_summary.csv','final_v1_hypothesis_results.md','final_v1_building_characteristics_exploratory.md','final_v1_final_test_report.md']}
        excluded|={'reports/final_v1/'+p.name for p in (ROOT/'reports/final_v1').iterdir() if p.name.startswith(('final_v1_','gate3_'))}
    if args.mode=='figure8':
        excluded|={'figure8_extraction/'+n for n in ['case_predictions.csv','case_metrics.csv','SEARCH_REPORT.md','VALIDATION_REPORT.md']}
    if args.mode=='edge-analysis':
        excluded|={p.relative_to(ROOT).as_posix() for p in (ROOT/'edge_deployment_final/e4_cross_device_validation').glob('*') if p.suffix in {'.csv','.json','.md'}}
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(x in rel.parts for x in ['.git','__pycache__','verification']): continue
        if rel.as_posix() in excluded: continue
        target=dest/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    for folder in ['data/raw/bdg2','reports/final_v1','models/bdg2_stage6_v3','models/final_v1','figure8_extraction']:
        (dest/folder).mkdir(parents=True,exist_ok=True)
    print(f'Prepared {args.mode} workspace. No experiment executed. See README for required external inputs.')

if __name__=='__main__':main()
