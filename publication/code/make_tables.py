from pathlib import Path
import csv,json,html
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from make_publication_figures import ROOT,DAYS,WINDOWS,MAE,RMSE,DMAE,DRMSE,A,B,COUNTS,LATENCY

TABLES=[]
def add(n,title,headers,rows,note,widths,part=''):
    TABLES.append(dict(n=n,title=title,headers=headers,rows=rows,note=note,widths=widths,part=part))
add(1,'Study cohorts, site composition, and roles in the cross-building transfer-learning experiment.',
 ['Cohort','Sites, n','Buildings, n','Site IDs','Role in the study'],[
 ['Source','4','20','Crow, Lamb, Rat, Robin','Source-model pretraining and source-side model selection'],
 ['Development','6','12','Bear, Cockatoo, Fox, Panther, Peacock, Wolf','Protocol development, adaptation analysis, and formulation of final-evaluation hypotheses'],
 ['Final-evaluation proposal','9','510','Bobcat, Bull, Eagle, Gator, Hog, Moose, Mouse, Shrew, Swan','Prespecified candidate pool for final evaluation'],
 ['Final evaluation','7','453','Bobcat, Bull, Eagle, Gator, Hog, Moose, Swan','Frozen final evaluation of the predefined protocol']],
 'The source, development, and final-evaluation cohorts used non-overlapping site sets. Fifty-seven buildings were excluded from the final proposal for continuous zero runs of at least 30 days. No eligible buildings remained from Mouse or Shrew. The development cohort was used to finalize the experimental and analysis protocol before final evaluation and was excluded from the final confirmatory estimates.',[1.4,.65,.85,2.9,3.7])
add(2,'Forecasting-model architecture, source-training configuration, and target-adaptation protocol.',
 ['Configuration','Source pretraining','Target adaptation'],[
 ['Historical electricity input / forecast horizon','24 h / 24 h','24 h / 24 h'],
 ['Input tensor per sample','24 × 6','24 × 6'],
 ['Input representation','Historical electricity plus five deterministic target-horizon calendar variables','Same representation'],
 ['Calendar variables','Hour sine/cosine; weekday sine/cosine; weekend indicator','Same representation'],
 ['Architecture','Single-layer LSTM, hidden size 32; linear 24-output head','Same architecture'],
 ['Updated parameters','Full network','Linear forecasting head only; LSTM frozen'],
 ['Parameters at inference','5,912','5,912'],
 ['Loss','MSE on source-normalized targets; mean reduction','Same loss'],
 ['Optimizer / learning rate','Adam / 0.001','Adam / 0.0005'],
 ['Epoch limit / early stopping','100 / patience 10','50 / none'],
 ['Model selection / validation','Chronologically separated source-validation data','No target-specific validation split'],
 ['Training / validation windows','218,206 / 12,980','Budget-dependent / not applicable'],
 ['Batch size','256 in the completed run','Nominal maximum 64; effective size min(64, N)'],
 ['DataLoader settings','shuffle=True; drop_last=False','shuffle=True; drop_last=False'],
 ['Random seeds','20260924, 20260925, 20260926','Same three seeds']],
 'MSE denotes mean squared error. N is the number of available adaptation samples. Incomplete final batches were retained. Calendar inputs are deterministic; no future measured electricity or other future-observed covariates were used.',[2.5,3.5,3.5],part='A  Model and optimization')
add(2,'Forecasting-model architecture, source-training configuration, and target-adaptation protocol.',
 ['Setting','Value'],[
 ['Normalization mean','167.065198'],['Normalization population standard deviation','354.328670'],
 ['Normalization data','Source-training inputs and targets only'],['Use during adaptation and evaluation','Frozen source statistics retained'],
 ['Target adaptation budget (days)','0 / 1 / 3 / 7 / 14 / 28'],['Available adaptation windows','0 / 1 / 49 / 145 / 313 / 649'],
 ['Effective DataLoader batch size','Not applicable / 1 / 49 / 64 / 64 / 64'],
 ['0-day condition','Source-only model without target-label-based updates'],
 ['Evaluation period','Common fixed 28-day interval for every budget within each building'],
 ['Evaluation origins','28 non-overlapping daily origins'],['Evaluation targets','672 hourly values per building, seed, and budget'],
 ['Adaptation/evaluation target overlap','0 h']],
 'Budget, sample-count, and batch-size sequences use the same order. The 1-day condition provides one complete supervised input–target window. No target-test-based early stopping, normalization, or checkpoint selection was used.',[3.2,6.3],part='B  Normalization and target-data budgets')
perf=[]
for i,d in enumerate(DAYS):perf.append(['Source-only (0 d)' if i==0 else f'{d} d adaptation',str(WINDOWS[i]),f'{MAE[i]:.4f}',f'{RMSE[i]:.4f}','—' if i==0 else f'{DMAE[i]:.4f}','—' if i==0 else f'{DRMSE[i]:.4f}','—' if i==0 else f'{A[i]:.2f}',f'{B[i]:.2f}'])
perf += [['Previous-day','—','20.8740','39.5247','—','—','—','—'],['Weekly-same-hour','—','18.1359','30.0582','—','—','—','—']]
add(3,'Final-evaluation performance across target-adaptation budgets and temporal baselines.',
 ['Model / budget','Adaptation windows','MAE','RMSE','ΔMAE','ΔRMSE','A (%)','B (%)'],perf,
 'All estimates use equal weighting of the seven sites after seed averaging within buildings and building averaging within sites. Errors are in original electricity-meter units. Δ denotes adapted minus source-only error. A denotes lower MAE than source-only; B denotes lower MAE than both temporal baselines. A is not applicable to source-only. Previous-day repeats the preceding 24 h; weekly-same-hour uses the observation 168 h before each target. The latter is a strong operational comparator with a longer history than the LSTM’s 24-h electricity input.',[1.75,1.05,1.10,1.10,1.20,1.20,1.05,1.05])
add(4,'Primary and secondary final-evaluation hypotheses and site-equal effect estimates.',
 ['Hypothesis','Prespecified estimand','Estimate','95% CI','Interpretation'],[
 ['H1 Primary','ΔMAE at 28 d relative to source-only','−16.4829','[−30.9076, −6.5630]','Supported under the prespecified criterion'],
 ['H2 Secondary','F1 rate(28 d) − F1 rate(1 d)','−55.15 pp','[−67.96, −45.52] pp','Lower frequency of adaptation-induced MAE degradation'],
 ['H3 Secondary','Rate(A) − rate(B) at 28 d','+67.99 pp','[+58.36, +78.35] pp','Source-only improvement more frequent than beating both baselines']],
 'Confidence intervals are 95% percentile intervals from 10,000 site-cluster bootstrap resamples (seed 20260925; seven sites). ΔMAE is adapted minus source-only MAE, in original meter units. F1 denotes adaptation-induced MAE degradation. A denotes source-only improvement; B denotes beating both temporal baselines. pp denotes percentage points. H2 and H3 are secondary effect-size analyses; no p-values or Holm-adjusted tests were calculated.',[1.15,2.65,1.15,1.85,2.7])
failure_rows=[]
for k,values in COUNTS.items():
    den=453 if k=='F4' else 1359;failure_rows.append([k]+[f'{v:,} ({100*v/den:.2f}%)' for v in values])
add(5,'Predefined adaptation-outcome and failure-event frequencies across target-data budgets.',
 ['Outcome','1 d','3 d','7 d','14 d','28 d'],failure_rows,
 'F1: MAE degradation relative to source-only. F2: RMSE degradation relative to source-only. F3-PD and F3-W: MAE worse than the previous-day and weekly-same-hour baseline, respectively. F3-any: worse than at least one baseline. F3-both: worse than both. F4: both improving and degrading seeds within the same building and budget. F1–F3 use 1,359 building–seed events per budget; F4 uses 453 buildings per budget. EPS = 10⁻¹² resolves numerical ties only. Across all nonzero budgets, 128/6,795 events (1.88%) improved MAE but worsened RMSE. Rates were recomputed directly from the reported integer counts.',[1.2,1.66,1.66,1.66,1.66,1.66])
add(6,'Cross-device FP32 numerical consistency and edge inference benchmark results.',
 ['Consistency check','Observed value','Prespecified limit','Result'],[
 ['64-vector output shape','64 × 24','Exact match','PASS'],['Output elements / NaN / Inf','1,536 / 0 / 0','No NaN or Inf','PASS'],
 ['Normalized maximum absolute difference','1.4782 × 10⁻⁵','≤1 × 10⁻⁴','PASS'],['Normalized mean absolute difference','1.4947 × 10⁻⁶','≤2 × 10⁻⁵','PASS'],
 ['Original-scale maximum absolute difference','0.005238','≤0.05','PASS'],['Original-scale mean absolute difference','0.000530','≤0.01','PASS'],
 ['Pre-/post-benchmark gates on each device','Identical acceptance outcomes','Frozen criteria','PASS']],
 'The 64-vector checks compare Jetson CUDA FP32 outputs with the frozen RTX reference. The pre-/post-benchmark gates use the fixed benchmark vector on each device and are distinct from the 64-vector validation. The canonical source-only checkpoint uses seed 20260924. These checks concern numerical consistency, not forecasting superiority.',[3.8,2.5,2,1.2],part='A  Numerical consistency')
latrows=[['Model','Parameters / precision / batch','5,912 / FP32 / 1','5,912 / FP32 / 1'],['Runtime','PyTorch / PyTorch CUDA','2.11.0 / 12.6','2.11.0+cu128 / 12.8']]
for mode,j,r in [('Model-only',LATENCY[0],LATENCY[1]),('End-to-end',LATENCY[2],LATENCY[3])]:
    for label,k in [('Mean (ms)',2),('Median (ms)',3),('P95 (ms)',4),('P99 (ms)',5)]:latrows.append([mode,label,f'{j[k]:.4f}',f'{r[k]:.4f}'])
latrows += [['CUDA memory','Peak allocated (MiB)','17.24','17.24'],['CUDA memory','Peak reserved (MiB)','22.00','22.00'],['Process memory','RSS (MiB)','558.80','Unavailable']]
add(6,'Cross-device FP32 numerical consistency and edge inference benchmark results.',
 ['Category','Metric','Jetson Orin Nano','RTX 3080'],latrows,
 'Each mode used 200 warm-up iterations followed by five runs of 1,000 measured inferences. Model-only measures the GPU forward pass with GPU-resident input/output. End-to-end adds CPU-to-GPU and GPU-to-CPU transfers; disk I/O, Python startup, and checkpoint loading are excluded. One Jetson end-to-end observation of 195.8847 ms was retained. RSS converts 585,945,088 bytes to MiB. RTX RSS was unavailable because of a Windows collector error; the benchmark was not rerun.',[1.6,3.1,2.4,2.4],part='B  Inference latency and memory')
add(6,'Cross-device FP32 numerical consistency and edge inference benchmark results.',
 ['Category','Metric','Jetson Orin Nano','RTX 3080'],[
 ['Power','Measurement boundary','tegrastats VDD_IN input rail','nvidia-smi GPU board power'],['Power','Idle reported power (W)','3.146','63.373'],['Power','Sustained reported power (W)','4.031','70.422'],['Power','Peak reported power (W)','4.032','123.660'],
 ['GPU temperature','Initial telemetry value (°C)','49.750','61.000'],['GPU temperature','Sustained mean (°C)','50.329','60.600'],['GPU temperature','Peak (°C)','50.843','63.000']],
 'Telemetry was measured during a separate sustained-inference workload. Power measurement boundaries differ across devices; no cross-device power or energy-per-inference ratio was calculated. Jetson jetson_clocks status could not be verified. RTX measurements were collected under Windows WDDM with background graphical contexts. The benchmark does not include sensor acquisition, networking, database access, or building-control actions.',[1.6,3.1,2.4,2.4],part='C  Device-specific power and thermal telemetry')
audit=[
 ['Cohort freeze','Final cohort after predefined quality screening','453 buildings / 7 sites','PASS'],
 ['Quality-gate traceability','Exclusions from the 510-building proposal','57; continuous zero runs ≥30 d','PASS'],
 ['Structural integrity','Buildings passing the structural gate','453/453; 0 integrity failures','PASS'],
 ['Substitution control','Replacement buildings or alternative test periods after the structural gate','None','PASS'],
 ['Adaptation/test separation','Overlap of adaptation and evaluation target timestamps','0 h','PASS'],
 ['Common evaluation truth','Forecast origin, target timestamp, horizon, and observed truth across models, budgets, and seeds','Identical building-level digest','PASS'],
 ['Normalization integrity','Source normalization statistics versus frozen values','Match','PASS'],
 ['Checkpoint integrity','Three source checkpoint hashes versus frozen values','Match','PASS'],
 ['Execution completeness','Stored final predictions and execution errors','6,088,320 rows; 0 errors','PASS'],
 ['Independent metric verification','Maximum stored-versus-recalculated metric difference','1.71 × 10⁻¹³; tolerance 10⁻¹⁰','PASS'],
 ['Prediction validity','Unavailable predictions / abnormal metric status','0 / 0','PASS'],
 ['Information boundary','Evaluation-period observations used for exploratory characteristics','None; all cutoffs preceded evaluation','PASS'],
 ['Protocol stability','Post-run changes to settings, buildings, periods, seeds, learning rate, or epochs','None','PASS'],
 ['Prediction preservation','Raw predictions regenerated after inspecting performance','No','PASS'],
 ['Exploratory addendum','Missing usage metadata for 18 Swan buildings','Exploratory outputs corrected; predictions, metrics, F1–F4 and H1–H3 unchanged','Documented / resolved'],
 ['Training-configuration audit','Executed loss and batching versus reported configuration','No result-affecting discrepancy identified','PASS']]
add(7,'Final-evaluation reproducibility, integrity, and protocol-freeze checks.',
 ['Category','Verification','Result','Status'],audit[:8],
 'PASS indicates successful completion of the specified integrity check. It is not a general claim of scientific validity. Full artifact hashes and software identifiers belong in the Supplementary Materials.',[1.9,3.5,2.9,1.2],part='A  Cohort and input integrity')
add(7,'Final-evaluation reproducibility, integrity, and protocol-freeze checks.',
 ['Category','Verification','Result','Status'],audit[8:],
 'The exploratory software correction retained 18 Swan buildings with missing usage metadata and changed only the descriptive characteristic outputs. The training audit distinguished the nominal adaptation batch maximum of 64 from the effective DataLoader size min(64, N); no result-affecting discrepancy was identified. Raw predictions and frozen final-evaluation results were preserved.',[1.9,3.5,2.9,1.2],part='B  Execution and analysis integrity')

def create():
    doc=Document();sec=doc.sections[0];sec.orientation=WD_ORIENT.LANDSCAPE;sec.page_width=Inches(11);sec.page_height=Inches(8.5);sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        s=doc.styles[name];s.font.name='Arial';s.font.color.rgb=RGBColor(0,0,0);s.font.size=Pt(10 if name=='Normal' else 14);s.paragraph_format.space_after=Pt(7)
    htmls=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Manuscript tables 1–7</title><style>body{font:14px Arial;color:#111;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #d9d9d9;padding:9px;vertical-align:middle}th{background:#edf4f7;text-align:left}h2{font-size:20px;margin-top:35px}p{line-height:1.5}button{padding:8px;cursor:pointer}.note{font-size:12px}.copy{color:#236a91}</style><body><h1>Manuscript tables 1–7</h1><p>Copy a table with its button, then paste into Google Docs. If clipboard access is blocked, select the rendered table and copy it normally.</p>']
    md=['# Manuscript tables 1–7\n'];last=0
    for ix,t in enumerate(TABLES):
        if ix:doc.add_page_break()
        title=f'Table {t["n"]}. {t["title"]}'
        p=doc.add_paragraph();p.paragraph_format.keep_with_next=True;r=p.add_run(title);r.bold=True;r.font.size=Pt(12)
        if t['part']:p=doc.add_paragraph(t['part']);p.paragraph_format.keep_with_next=True;p.runs[0].bold=True
        table=doc.add_table(rows=1,cols=len(t['headers']));table.autofit=False
        for c,w in zip(table.columns,t['widths']):c.width=Inches(w)
        for cell,w,h in zip(table.rows[0].cells,t['widths'],t['headers']):cell.width=Inches(w);cell.text=h
        trPr=table.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
        for row in t['rows']:
            cells=table.add_row().cells
            for cell,w,val in zip(cells,t['widths'],row):cell.width=Inches(w);cell.text=str(val)
        for ri,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                tcpr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    x=OxmlElement('w:'+side);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
                tcpr.append(borders);m=OxmlElement('w:tcMar')
                for side,val in [('top',65),('bottom',65),('left',85),('right',85)]:x=OxmlElement('w:'+side);x.set(qn('w:w'),str(val));x.set(qn('w:type'),'dxa');m.append(x)
                tcpr.append(m)
                if ri==0:s=OxmlElement('w:shd');s.set(qn('w:fill'),'EDF4F7');tcpr.append(s)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
                    for r in p.runs:r.font.size=Pt(9.5);r.bold=ri==0
        p=doc.add_paragraph('Note: '+t['note']);p.paragraph_format.space_before=Pt(8)
        for r in p.runs:r.font.size=Pt(9)
        suffix=('_'+t['part'][0]) if t['part'] else '';file=ROOT/'tables'/f'Table_{t["n"]}{suffix}.csv'
        with file.open('w',newline='',encoding='utf-8-sig') as f:w=csv.writer(f);w.writerow(t['headers']);w.writerows(t['rows'])
        ht='<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in t['headers'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in t['rows'])+'</tbody></table>'
        htmls += [f'<h2>{html.escape(title)}</h2>',f'<p><b>{html.escape(t["part"])}</b></p>' if t['part'] else '',f'<button onclick="copyTable(\'t{ix}\',this)">Copy table</button>',f'<div id="t{ix}">'+ht+'</div>',f'<p class="note">Note: {html.escape(t["note"])}</p>']
        md += ['\n## '+title+'\n',t['part']+'\n','| '+' | '.join(t['headers'])+' |','| '+' | '.join(['---']*len(t['headers']))+' |']+['| '+' | '.join(map(str,r))+' |' for r in t['rows']]+['\nNote: '+t['note']+'\n']
    doc.core_properties.title='Manuscript Tables 1 to 7';doc.core_properties.subject='Editable academic tables for the integrated manuscript v2';doc.save(ROOT/'Manuscript_Tables_1_to_7.docx')
    htmls.append('''<script>async function copyTable(id,btn){const el=document.getElementById(id);try{await navigator.clipboard.write([new ClipboardItem({'text/html':new Blob([el.innerHTML],{type:'text/html'}),'text/plain':new Blob([el.innerText],{type:'text/plain'})})]);btn.textContent='Copied';}catch(e){const s=window.getSelection(),r=document.createRange();r.selectNodeContents(el);s.removeAllRanges();s.addRange(r);btn.textContent='Selected — press Ctrl/Cmd+C';}}</script></body></html>''')
    (ROOT/'Manuscript_Tables_1_to_7.html').write_text('\n'.join(htmls),encoding='utf-8');(ROOT/'tables'/'Manuscript_Tables_1_to_7.md').write_text('\n'.join(md),encoding='utf-8');(ROOT/'data'/'tables.json').write_text(json.dumps(TABLES,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Created',len(TABLES),'table panels in one DOCX')
if __name__=='__main__':create()
