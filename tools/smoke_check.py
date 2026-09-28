"""Bounded checks on copied code, synthetic records and saved results only.

Never trains/adapts a model, loads a checkpoint, or generates model predictions.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECKS=[]

def check(name,condition,detail=''):
    CHECKS.append({'check':name,'pass':bool(condition),'detail':detail})
    print(('PASS' if condition else 'FAIL')+' | '+name+(' | '+str(detail) if detail else ''))
    if not condition:raise AssertionError(name)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def core():
    import numpy as np
    import pandas as pd
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/final_v1')]
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.relative_to(ROOT).as_posix())
    check('all Python files parse',True)
    # Imports create only empty model directories in a disposable workspace.
    with tempfile.TemporaryDirectory(prefix='paper_release_smoke_') as tmp:
        os.environ['PAPER_WORKSPACE']=tmp
        for name in ['test_final_v1_gate1_synthetic.py','test_final_v1_gate2_synthetic.py','test_final_v1_gate3_synthetic.py']:
            runpy.run_path(str(ROOT/'scripts/final_v1'/name),run_name='__main__')
            check(name,True)
        import bdg2_stage6_c0_common_v3 as c0
        import run_bdg2_stage6_c0_lstm_v3 as train
        features=c0.make_c0_features(np.arange(24)[None,:],pd.DataFrame({'target_start':['2020-01-02']}),10.0,2.0)
        check('C0 shape and power-lag normalization',features.shape==(1,24,6) and np.array_equal(features[0,:,0],(np.arange(24)-10)/2))
        check('target calendar begins at target midnight',features[0,0,1]==0 and features[0,0,2]==1)
        cfg=train.Config()
        check('historical optimizer/batch configuration',cfg.source_batch_size==256 and cfg.adaptation_batch_size==64 and cfg.adaptation_epochs==50 and cfg.source_learning_rate==.001 and cfg.adaptation_learning_rate==.0005)
        # No call to fit_source/adapt_head/predict/model.forward.
        s=(ROOT/'scripts/run_bdg2_stage6_c0_lstm_v3.py').read_text(encoding='utf-8')
        check('mean MSE/head-only implementation retained',s.count('criterion = nn.MSELoss()')==2 and 'torch.optim.Adam(model.head.parameters()' in s and 'parameter.requires_grad = False' in s and 'batch_size=min(batch_size, len(dataset))' in s)
        import audit_bdg2_electricity as audit
        fixture=Path(tmp)/'synthetic.csv'
        pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=49,freq='h'),'Synthetic':np.arange(49,dtype=float)}).to_csv(fixture,index=False)
        meta=audit.timestamp_audit(str(fixture)); rows=audit.value_audit(str(fixture),meta)
        check('actual quality auditor on 49-row fixture',len(rows)==1)
    os.environ.pop('PAPER_WORKSPACE',None)

    pred=pd.read_csv(ROOT/'figure8_extraction/case_predictions.csv',float_precision='round_trip')
    old=pd.read_csv(ROOT/'figure8_extraction/case_metrics.csv',float_precision='round_trip')
    check('saved Figure 8 6720 rows / 10 keys',len(pred)==6720 and pred.groupby(['building_id','seed','adaptation_days']).ngroups==10)
    check('no duplicated case prediction keys',not pred.duplicated(['building_id','seed','adaptation_days','forecast_origin','horizon']).any())
    check('finite saved values',np.isfinite(pred[['y_true','y_pred']]).all().all())
    largest=0.
    for key,g in pred.groupby(['building_id','seed','adaptation_days']):
        check('case '+str(key),len(g)==672 and g.forecast_origin.nunique()==28 and all(sorted(h.horizon)==list(range(1,25)) for _,h in g.groupby('forecast_origin')))
        row=old[(old.building_id==key[0])&(old.seed==key[1])&(old.adaptation_days==key[2])].iloc[0]
        err=g.y_pred.to_numpy()-g.y_true.to_numpy();mae=np.abs(err).mean();rmse=np.sqrt(np.mean(err**2))
        largest=max(largest,abs(mae-row.mae),abs(rmse-row.rmse))
    check('saved Figure 8 metric recomputation',largest<=1e-10,f'max difference {largest:.3g}')
    aligned=True
    for (building,seed),g in pred.groupby(['building_id','seed']):
        base=g[g.adaptation_days==0].sort_values(['forecast_origin','horizon'])
        adapted=g[g.adaptation_days>0].sort_values(['forecast_origin','horizon'])
        key=['forecast_origin','target_timestamp','horizon','y_true']
        aligned &= base[key].reset_index(drop=True).equals(adapted[key].reset_index(drop=True))
    check('Figure 8 source/adapted target keys and observations identical',aligned)

    import analyze_final_v1_gate3 as stats
    eff=pd.read_csv(ROOT/'reports/final_v1_data_efficiency.csv',float_precision='round_trip')
    events,summary=stats.build_failures(eff)
    saved=pd.read_csv(ROOT/'reports/final_v1_failure_summary.csv',float_precision='round_trip')
    cols=[x for x in saved.columns if x.endswith('_count')]
    check('all F1-F4 saved counts reproduced',np.array_equal(summary[cols].to_numpy(),saved[cols].to_numpy()))
    curve,_=stats.build_curves(eff)
    savedcurve=pd.read_csv(ROOT/'reports/final_v1/final_v1_site_equal_adaptation_curve.csv',float_precision='round_trip')
    nums=[x for x in curve.columns if x in savedcurve and pd.api.types.is_numeric_dtype(curve[x])]
    delta=np.max(np.abs(curve[nums].to_numpy()-savedcurve[nums].to_numpy()))
    check('all six site-equal adaptation results',delta<1e-10,f'max difference {delta:.3g}')
    h,*_=stats.build_hypotheses(eff,events)
    saved_h=pd.read_csv(ROOT/'supplementary_final/tables/Table_S4_A_hypothesis_effects.csv')
    hypothesis_ok=True
    for key in ['h1','h2','h3']:
        row=saved_h[saved_h.hypothesis==key.upper()].iloc[0]
        hypothesis_ok &= abs(h[key+'_effect']-row.estimate)<1e-9 and abs(h[key+'_ci_low']-row.ci95_lower)<1e-9 and abs(h[key+'_ci_high']-row.ci95_upper)<1e-9
    check('H1/H2/H3 saved effects and CIs',hypothesis_ok)
    cohort=pd.read_csv(ROOT/'reports/final_v1_cohort_manifest.csv')
    retained=cohort[cohort.selection_status=='selected_for_final_v1']
    check('453 buildings / 7 sites',len(retained)==453 and retained.site_id.nunique()==7)
    check('documented final missingness discrepancy',int((retained.missing_rate_all_records>.2).sum())==31)
    char=pd.read_csv(ROOT/'reports/final_v1/final_v1_building_characteristics.csv')
    assoc,groups=stats.characteristic_analysis(char,events)
    oldassoc=pd.read_csv(ROOT/'reports/final_v1/final_v1_characteristic_associations.csv')
    check('350 exploratory association records match',len(assoc)==350 and np.allclose(assoc.spearman_rho,oldassoc.spearman_rho,atol=1e-12,equal_nan=True) and np.array_equal(assoc.n_complete,oldassoc.n_complete))
    edge=module(ROOT/'edge_deployment_final/e4_cross_device_validation/analyze_e4_outputs.py','release_e4')
    b=ROOT/'edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32/validation'
    e4=ROOT/'edge_deployment_final/e4_cross_device_validation'
    for scale,max_tol,mean_tol in [('normalized',1e-4,2e-5),('physical',.05,.01)]:
        result=edge.metrics(np.load(b/f'validation_reference_output_{scale}.npy'),np.load(e4/f'jetson_validation_output_{scale}.npy'))
        check('saved E4 '+scale,result['shape']==[64,24] and result['max_absolute_difference']<=max_tol and result['mean_absolute_difference']<=mean_tol)
    e5=module(ROOT/'edge_deployment_final/e5_performance_benchmark/analyze_e5_results.py','release_e5')
    for device in ['jetson','rtx3080']:
        folder=ROOT/'edge_deployment_final/e5_performance_benchmark'/device
        oldlat=json.loads((folder/'latency_summary.json').read_text(encoding='utf-8'))
        for mode in ['model_only','end_to_end']:
            recalc=e5.recalc_latency(folder/f'latency_{mode}_raw.csv')
            result=e5.compare_recorded(oldlat[mode],recalc)
            check('saved 5000 latency values '+device+'/'+mode,result['pass'],str(result))
    print('NO TRAINING, ADAPTATION, CHECKPOINT LOADING OR MODEL INFERENCE EXECUTED')

def figures():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    # Read constants through AST: importing the historical plotting module has output side effects.
    p=ROOT/'publication/code/make_publication_figures.py'
    constants={}
    for node in ast.parse(p.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign):
            for name in node.targets:
                if isinstance(name,ast.Name) and name.id in {'MAE','RMSE','DMAE','DRMSE','COUNTS','LATENCY','DAYS','A','B'}:
                    constants[name.id]=ast.literal_eval(node.value)
    curve=pd.read_csv(ROOT/'reports/final_v1/final_v1_site_equal_adaptation_curve.csv')
    for const,col in [('MAE','mae_site_equal'),('RMSE','rmse_site_equal'),('DMAE','delta_mae_source_site_equal'),('DRMSE','delta_rmse_source_site_equal')]:
        check('publication rounded '+const,np.allclose(constants[const],curve[col],atol=5.1e-5,rtol=0))
    summary=pd.read_csv(ROOT/'reports/final_v1_failure_summary.csv').iloc[:5]
    for a,b in [('F1','f1_count'),('F2','f2_count'),('F3-PD','f3_previous_day_count'),('F3-W','f3_weekly_same_hour_count'),('F3-any','f3_any_count'),('F3-both','f3_both_count'),('F4','f4_count')]:
        check('publication counts '+a,list(summary[b])==constants['COUNTS'][a])
    with tempfile.TemporaryDirectory(prefix='paper_release_plot_') as tmp:
        oldargv=sys.argv
        sys.argv=[str(ROOT/'publication/code/make_figure8_from_predictions.py'),str(ROOT/'figure8_extraction/case_predictions.csv'),'--out',tmp]
        try:runpy.run_path(sys.argv[0],run_name='__main__')
        finally:sys.argv=oldargv
        check('Figure 8 renders from existing predictions',all((Path(tmp)/('Figure_8.'+x)).stat().st_size>0 for x in ['png','svg','pdf']))
    print('Only saved-result rendering was executed; temporary smoke graphics were not added as scientific results.')

def ancillary():
    import pandas as pd
    # This optional path reuses an already installed pure-Python python-docx;
    # it is unnecessary in a normal environment installed from requirements.txt.
    extra=os.environ.get('DOCX_SITE_PACKAGES')
    if extra:sys.path.append(extra)
    with tempfile.TemporaryDirectory(prefix='paper_release_tables_') as tmp:
        dest=Path(tmp)/'replay'
        from prepare_workspace import ROOT as release_root
        check('workspace helper resolves release root',release_root==ROOT)
        args=sys.argv
        sys.argv=['prepare_workspace','--dest',str(dest),'--mode','snapshot']
        try:runpy.run_path(str(ROOT/'tools/prepare_workspace.py'),run_name='__main__')
        finally:sys.argv=args
        sys.path.insert(0,str(dest/'publication/code'))
        table=module(dest/'publication/code/make_tables.py','release_main_tables')
        check('main table builder imports without missing local module',len(table.TABLES)>7)
        os.environ['PAPER_WORKSPACE']=str(dest)
        supp=module(dest/'supplementary_final/code/build_supplementary_data_figures.py','release_supp')
        for folder in [supp.DATA,supp.TABLES,supp.FIGURES,supp.QA]:folder.mkdir(parents=True,exist_ok=True)
        data=supp.prepare_full_data();supp.build_tables(data);qa=supp.validate_outputs(data)
        check('Supplementary data/table reconstruction without checkpoints',qa['status']=='PASS')
        check('Data S17 original gap retained',len(pd.read_csv(supp.DATA/'Data_S17_source_checkpoint_hashes.csv'))==1)
        os.environ.pop('PAPER_WORKSPACE',None)
    print('Table import and supplementary saved-result reconstruction only; no DOCX authored or model executed.')

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--core',action='store_true');g.add_argument('--figures',action='store_true');g.add_argument('--ancillary',action='store_true');args=ap.parse_args()
    started=time.monotonic();status='FAIL';out=ROOT/'verification';out.mkdir(exist_ok=True);capture=io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            (core if args.core else figures if args.figures else ancillary)()
        status='PASS'
    finally:
        mode='core' if args.core else 'figures' if args.figures else 'ancillary'
        text=capture.getvalue();print(text)
        (out/f'{mode}_smoke.log').write_text(text,encoding='utf-8')
        result={'status':status,'mode':mode,'elapsed_seconds':time.monotonic()-started,'python':sys.version.split()[0],'checks':CHECKS,'new_model_experiment':False}
        (out/f'{mode}_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
