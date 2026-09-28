"""Verify supplied external inputs against hashes measured during release assembly."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=ROOT)
    ap.add_argument('--kind',choices=['data','checkpoints','predictions','all'],default='all')
    args=ap.parse_args(); missing=[]
    for item in json.loads((ROOT/'configs/external_artifacts.json').read_text()):
        name=item['path'];kind='data' if name.startswith('data/') else 'checkpoints' if name.endswith('.pt') else 'predictions'
        if args.kind not in ['all',kind]:continue
        path=args.workspace/name
        if not path.is_file():print('MISSING',name);missing.append(name);continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if h.hexdigest()!=item['sha256']:raise ValueError('Hash mismatch: '+name)
        print('PASS',name)
    if missing:raise SystemExit(2)

if __name__=='__main__':main()
