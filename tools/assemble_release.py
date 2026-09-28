"""Maintainer-only release assembly from a sibling research_stage1 checkout.

Copies existing evidence; never executes research scripts or creates predictions.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
SOURCE = RELEASE.parent / 'research_stage1'
ROWS = []
TEXT_SUFFIX = {'.py', '.csv', '.json', '.md', '.txt', '.sh', '.ps1', '.svg', '.drawio'}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sanitize(text):
    # Replace a project-root prefix first, including JSON-escaped Windows paths.
    variants = [str(SOURCE), str(SOURCE).replace('\\', '/'), str(SOURCE).replace('\\', '\\\\')]
    for value in sorted(variants, key=len, reverse=True):
        text = text.replace(value, '.')
    text = re.sub(r'C:[\\/]+Users[\\/]+[^\s\"\'<>`|]+', '<LOCAL_PATH>', text)
    text = re.sub(r'/home/[^/\s\"\']+', '/home/USER', text)
    text = text.replace('han@jetson-iot', 'USER@JETSON_HOST').replace('jetson-iot', 'JETSON_HOST')
    # No personal absolute interpreter path in examples.
    text = re.sub(r'& \"<LOCAL_PATH>\"', 'python', text)
    return text

def transform_code(text, relative):
    # Keep the historical file layout, with an optional isolated run workspace.
    if re.search(r'^ROOT = ', text, re.M):
        text = re.sub(r'^ROOT = (.+)$', r'ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(\1))).resolve()', text, flags=re.M)
    if 'MODEL_DIR = ROOT_MODEL_DIR =' in text:
        text = text.replace('Path(__file__).resolve().parent.parent / "models"', 'Path(os.environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parent.parent))) / "models"')
    # Imports must use the release code directory, not the external data workspace.
    text = text.replace('SCRIPTS = ROOT / "scripts"', 'SCRIPTS = Path(__file__).resolve().parents[1]')
    if relative.startswith('scripts/'):
        text = text.replace('os.path.join("research_stage1", ', 'os.path.join(os.environ.get("PAPER_WORKSPACE", "."), ')
    if relative == 'figure8_extraction/extract_figure8_cases.py':
        text = text.replace('OUT_DIR = Path(__file__).resolve().parent', 'OUT_DIR = ROOT / "figure8_extraction"')
    return sanitize(text)

def copy_bytes(data, origin, relative):
    dest = RELEASE / relative
    original = data
    if dest.suffix in TEXT_SUFFIX:
        text = data.decode('utf-8-sig')
        new = transform_code(text, relative) if dest.suffix == '.py' else sanitize(text)
        if new != text:
            data = new.encode('utf-8-sig' if original.startswith(b'\xef\xbb\xbf') else 'utf-8')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite release artifact: {relative}')
    dest.write_bytes(data)
    ROWS.append({'source': origin, 'release_path': relative, 'source_size': len(original),
                 'source_sha256': digest(original), 'release_sha256_at_copy': digest(data),
                 'copy_change': 'path/privacy portability edits' if data != original else 'byte-identical'})

def copy_file(relative, target=None):
    path = SOURCE / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    copy_bytes(path.read_bytes(), relative, target or relative)

def main():
    entries = '''audit_bdg2_electricity audit_bdg2_zero_values join_bdg2_metadata_quality
review_bdg2_experiment_candidates prepare_bdg2_cohort_review review_bdg2_cohort_split
build_bdg2_windows build_bdg2_temporal_splits prepare_bdg2_multibuilding_v2
prepare_bdg2_stage6_c0_protocol_v3 run_bdg2_stage6_c0_lstm_v3
analyze_bdg2_v5_building_characteristics'''.split()
    queue = [SOURCE/'scripts'/f'{x}.py' for x in entries]
    queue += list((SOURCE/'scripts/final_v1').glob('*.py'))
    visited = set()
    while queue:
        p = queue.pop()
        if p in visited: continue
        visited.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            mods = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else []
            for mod in mods:
                for parent in [p.parent, SOURCE/'scripts']:
                    candidate = parent/f'{mod}.py'
                    if candidate.exists(): queue.append(candidate)
    for p in sorted(visited): copy_file(p.relative_to(SOURCE).as_posix())
    copy_file('figure8_extraction/extract_figure8_cases.py')
    for name in ['case_predictions.csv', 'case_metrics.csv', 'VALIDATION_REPORT.md', 'SEARCH_REPORT.md']:
        copy_file('figure8_extraction/'+name)

    # Frozen quality/selection evidence and small final results only.
    names = '''bdg2_electricity_quality.csv bdg2_zero_value_audit.csv bdg2_metadata_quality_join.csv
bdg2_experiment_candidates.csv bdg2_cohort_candidates.csv bdg2_cohort_split_alternative.csv
bdg2_split_manifest.csv bdg2_temporal_split_manifest.csv
bdg2_multibuilding_cohort_manifest_v2.csv bdg2_multibuilding_temporal_manifest_v2.csv
bdg2_stage6_c0_protocol_manifest_v3.csv bdg2_v5_source_scale_reference.csv
bdg2_v5_final_analysis_protocol_draft.md bdg2_multibuilding_selection_review_v2.md
bdg2_stage6_c0_lstm_results_v3.md bdg2_stage6_c0_training_history_v3.csv
bdg2_stage6_c0_source_only_metrics_v3.csv bdg2_stage6_c0_adaptation_metrics_v3.csv
bdg2_source_review.md bdg2_meter_scope_review.md'''.split()
    names += [p.name for p in (SOURCE/'reports').glob('final_v1*') if p.is_file() and p.suffix in {'.csv','.md'} and p.stat().st_size < 12_000_000]
    for name in sorted(set(names)): copy_file('reports/'+name)
    for p in sorted((SOURCE/'reports/final_v1').iterdir()):
        if p.is_file() and p.suffix in {'.csv','.json','.md'} and p.stat().st_size < 5_000_000:
            copy_file(p.relative_to(SOURCE).as_posix())
    copy_file('models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json')

    for p in (SOURCE/'supplementary_final').iterdir():
        if p.suffix == '.md': copy_file(p.relative_to(SOURCE).as_posix())
    for folder in ['data','tables','figures','code']:
        for p in sorted((SOURCE/'supplementary_final'/folder).iterdir()):
            if p.is_file() and p.suffix in {'.csv','.py','.pdf','.svg','.png'}:
                copy_file(p.relative_to(SOURCE).as_posix())
    manuscript = SOURCE/'supplementary_final/reference/handoff_source/handoff/English_Manuscript'
    for p in manuscript.glob('*.md'):
        copy_file(p.relative_to(SOURCE).as_posix(), 'docs/manuscript/'+p.name)

    archive = next((SOURCE/'supplementary_final/reference').rglob('Figure_Table_Package_v3_publication.zip'))
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or len(parts)<2: continue
            rel = Path(*parts[1:])
            if rel.parts[0] not in {'code','data','tables','figures','drawio'}: continue
            if 'legacy' in rel.name or 'LAYOUT_ONLY' in rel.name: continue
            if rel.suffix not in {'.py','.csv','.json','.md','.png','.pdf','.svg','.drawio'}: continue
            copy_bytes(z.read(info), archive.relative_to(SOURCE).as_posix()+'!'+info.filename,
                       'publication/'+rel.as_posix())

    bundle = 'edge_deployment_final/e2_transfer_freeze/edge_deployment_jetson_fp32'
    for folder in ['scripts','config','validation']:
        for p in sorted((SOURCE/bundle/folder).glob('*')):
            if p.is_file() and p.suffix in {'.py','.json','.csv','.npy'} and p.stat().st_size < 2_000_000:
                copy_file(p.relative_to(SOURCE).as_posix())
    for stage in ['e4_cross_device_validation','e5_performance_benchmark']:
        base = SOURCE/'edge_deployment_final'/stage
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.stat().st_size > 3_000_000: continue
            if p.suffix not in {'.py','.csv','.json','.npy','.md'}: continue
            if any(v in p.name for v in ['run_e4_transfer','run_e5_benchmark','command_log','artifact_manifest','build_artifact']): continue
            if p.suffix=='.py' and p.name not in {'analyze_e4_outputs.py','analyze_e5_results.py','benchmark_fp32.py','benchmark_fp32_executed_v1.py'}: continue
            copy_file(p.relative_to(SOURCE).as_posix())

    # Real hashes of excluded inputs: recorded hashes are never substituted.
    excluded=[]
    paths=[SOURCE/'data/raw/bdg2/electricity.csv', SOURCE/'data/raw/bdg2/metadata.csv', SOURCE/'reports/final_v1_raw_predictions.csv']
    paths += sorted((SOURCE/'models/bdg2_stage6_v3').glob('*.pt'))
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
        excluded.append({'path':p.relative_to(SOURCE).as_posix(),'size_bytes':p.stat().st_size,'sha256':h.hexdigest(),'included':False,
                         'reason':'external raw dataset / checkpoint / full prediction archive; deliberately not redistributed'})
    (RELEASE/'configs').mkdir(exist_ok=True)
    (RELEASE/'configs/external_artifacts.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    (RELEASE/'provenance').mkdir(exist_ok=True)
    with (RELEASE/'provenance/source_map.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
    print(f'Copied {len(ROWS)} existing artifacts. Recorded {len(excluded)} excluded-input hashes.')

if __name__=='__main__': main()
