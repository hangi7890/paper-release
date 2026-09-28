"""Apply documented release-only packaging repairs, with no numerical changes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGES = []

def edit(name, old, new, reason):
    p=ROOT/name
    text=p.read_text(encoding='utf-8-sig')
    if old not in text: raise AssertionError(f'Expected patch not found: {name}: {old[:60]}')
    p.write_text(text.replace(old,new),encoding='utf-8')
    CHANGES.append({'file':name,'change':reason})

def main():
    edit('publication/code/make_tables.py','from make_figure_table_package import','from make_publication_figures import',
         'Fix missing import: only make_publication_figures.py exists in supplied v3 ZIP; same imported constants.')
    p=ROOT/'publication/code/make_publication_figures.py'
    lines=p.read_text(encoding='utf-8').splitlines()
    lines=['if __name__ == "__main__":\n    '+line if line.startswith('csvout(') else line for line in lines]
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    CHANGES.append({'file':str(p.relative_to(ROOT)),'change':'Do not rewrite presentation CSVs when imported by table builder; direct script behavior retained.'})
    edit('publication/code/make_figure8_from_predictions.py',"pd.to_datetime(df[c],utc=True,errors='raise')", "pd.to_datetime(df[c],errors='raise')",
         'Preserve original naive timestamps; remove unsupported UTC labeling in rendering only. Values and relative-hour plots unchanged.')
    # Historical hash checks protect exact inputs. Sanitized path-bearing metadata
    # has different bytes: pin the actually shipped metadata, keep raw/weights hashes.
    gate=ROOT/'scripts/final_v1/run_final_v1_gate2.py'
    text=gate.read_text(encoding='utf-8')
    rows=list(csv.DictReader((ROOT/'provenance/source_map.csv').open(encoding='utf-8')))
    for row in rows:
        rel=row['release_path']
        if rel in ['reports/final_v1_protocol_manifest.md','reports/bdg2_v5_final_analysis_protocol_draft.md','models/bdg2_stage6_v3/bdg2_stage6_c0_lstm_config_v3.json']:
            actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
            text=text.replace(row['source_sha256'],actual)
            CHANGES.append({'file':str(gate.relative_to(ROOT)),'change':f'Pin sanitized {rel}: {actual}; historical digest remains in provenance.'})
    gate.write_text(text,encoding='utf-8')
    # Table S9 can describe excluded weights from the measured release inventory.
    edit('supplementary_final/code/build_supplementary_data_figures.py',
         'path = ROOT / row.path\n        reproducibility_rows.append',
         'path = ROOT / row.path\n        external = json.loads((Path(__file__).resolve().parents[2] / "configs" / "external_artifacts.json").read_text(encoding="utf-8"))\n        recorded_size = next(item["size_bytes"] for item in external if item["path"] == row.path)\n        reproducibility_rows.append',
         'Use locally measured excluded-checkpoint size inventory; do not require private checkpoint to render tables.')
    edit('supplementary_final/code/build_supplementary_data_figures.py',
         '"size_bytes": path.stat().st_size, "sha256": row.sha256',
         '"size_bytes": path.stat().st_size if path.exists() else recorded_size, "sha256": row.sha256',
         'Report measured original checkpoint size when checkpoint deliberately omitted.')
    # These are packaging/Word-authoring helpers, not the paper experiment pipeline;
    # omit to avoid dependencies on the private handoff archive and installed Office.
    omitted=[]
    for name in ['compose_supplementary_markdown.py','build_supplementary_docx.py','build_supplementary_manifest.py','verify_supplementary_package.py']:
        p=ROOT/'supplementary_final/code'/name
        if not p.is_file(): raise FileNotFoundError(p)
        omitted.append({'path':str(p.relative_to(ROOT)),'reason':'document/package authoring helper, not scientific pipeline; private handoff/Office dependencies omitted'})
        p.unlink()
    (ROOT/'provenance/omitted_packaging_helpers.json').write_text(json.dumps(omitted,indent=2),encoding='utf-8')
    # Resolve stale archived command examples in public documentation.
    for p in [ROOT/'supplementary_final/SOURCE_TRACE.md']:
        text=p.read_text(encoding='utf-8').replace("& '<LOCAL_PATH>'",'python')
        p.write_text(text,encoding='utf-8')
    (ROOT/'provenance/portability_changes.json').write_text(json.dumps(CHANGES,indent=2),encoding='utf-8')
    print(f'{len(CHANGES)} portability changes; {len(omitted)} ancillary helpers omitted.')

if __name__=='__main__':main()
