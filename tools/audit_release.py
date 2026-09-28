"""Read-only release/privacy/provenance audit; writes only its audit artifacts.

--check-originals also verifies the sibling private source files have not changed.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEXT={'.py','.json','.md','.csv','.txt','.svg','.drawio'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def tracked():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-originals',action='store_true');args=ap.parse_args()
    issues=[];syntax=[];local_missing=[];privacy=[]
    py=list(ROOT.rglob('*.py'))
    mods={p.stem for p in py}
    packages={'numpy','pandas','torch','PIL','matplotlib','docx'}
    for p in py:
        tree=ast.parse(p.read_text(encoding='utf-8-sig'));syntax.append(p.relative_to(ROOT).as_posix())
        for n in ast.walk(tree):
            imports=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module else []
            for name in imports:
                if name not in mods|packages|sys.stdlib_module_names:local_missing.append({'file':p.relative_to(ROOT).as_posix(),'module':name})
    patterns={
        'private_windows_user_path':re.compile(r'[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_-]+',re.I),
        'private_unix_home':re.compile(r'/home/(?!USER\b)[A-Za-z0-9_-]+'),
        'ssh_private_key':re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
        'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
        'openai_token':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
        'aws_key':re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
        'local_login':re.compile(r'\b(?:han@jetson-iot|korea@|192\.168\.\d+\.\d+)\b'),
    }
    for p in tracked():
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix in {'.pt','.pth','.ckpt','.safetensors','.pkl','.zip','.pem','.key'}:issues.append('forbidden binary '+rel)
        if p.stat().st_size>50_000_000:issues.append('oversized '+rel)
        if rel.startswith('data/raw/') or rel.endswith('final_v1_raw_predictions.csv'):issues.append('excluded raw '+rel)
        if p.suffix in TEXT:
            text=p.read_text(encoding='utf-8-sig')
            if p.suffix=='.json':json.loads(text)
            for name,pattern in patterns.items():
                for match in pattern.finditer(text):
                    # Pattern definitions are not a secret or private address.
                    if rel in ['tools/audit_release.py','tools/assemble_release.py']:continue
                    privacy.append({'file':rel,'pattern':name,'line':text.count('\n',0,match.start())+1})
    issues += ['privacy scan: '+str(x) for x in privacy]
    issues += ['unresolved import: '+str(x) for x in local_missing]
    source_rows=list(csv.DictReader((ROOT/'provenance/source_map.csv').open(encoding='utf-8')))
    copies=[]
    for row in source_rows:
        p=ROOT/row['release_path']
        copies.append({**row,'release_sha256_final':sha(p) if p.exists() else '', 'release_status':'included' if p.exists() else 'excluded_document_authoring_helper'})
    write_csv(ROOT/'provenance/source_map_final.csv',copies,list(copies[0]))

    if args.check_originals:
        source=ROOT.parent/'research_stage1'; preserved=[];archives={}
        for row in source_rows:
            name=row['source']
            if '!' in name:
                archive,member=name.split('!',1)
                if archive not in archives:archives[archive]=zipfile.ZipFile(source/archive)
                actual=hashlib.sha256(archives[archive].read(member)).hexdigest()
            else:actual=sha(source/name)
            good=actual==row['source_sha256']
            preserved.append({'source':name,'before_sha256':row['source_sha256'],'after_sha256':actual,'unchanged':good})
            if not good:issues.append('source modified: '+name)
        for z in archives.values():z.close()
        for row in json.loads((ROOT/'configs/external_artifacts.json').read_text()):
            actual=sha(source/row['path']); good=actual==row['sha256']
            preserved.append({'source':row['path'],'before_sha256':row['sha256'],'after_sha256':actual,'unchanged':good})
            if not good:issues.append('external source changed: '+row['path'])
        write_csv(ROOT/'provenance/original_preservation.csv',preserved,list(preserved[0]))
    results={'status':'PASS' if not issues else 'FAIL','python_files_parsed':len(syntax),'local_imports_unresolved':local_missing,'privacy_findings':privacy,'issues':issues,'scan_scope':'all released text files, standard token/private path patterns; no claim of exhaustive privacy proof','source_preservation_checked':args.check_originals}
    (ROOT/'verification/release_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    # Self-reference excluded; all other checked output hashes come from current bytes.
    manifest=ROOT/'provenance/release_manifest.csv'
    records=[{'path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in tracked() if p!=manifest]
    write_csv(manifest,records,list(records[0]))
    print(json.dumps(results,indent=2));print('Manifest entries:',len(records))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
