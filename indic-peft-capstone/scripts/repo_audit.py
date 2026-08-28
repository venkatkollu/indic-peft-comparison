import json, csv, os, re, subprocess, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path('/home/ubuntu/indic-peft-comparison')
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)
lines = []

def add(s=''):
    lines.append(s)

add('# Repository Audit')
add('')
add(f'- Repository: {ROOT}')
add(f"- Commit: {subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()}")
add(f"- Audited files: {sum(1 for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts)}")
add('')
add('## File inventory')
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and '.git' not in p.parts and 'audit' not in p.parts:
        add(f'- `{p.relative_to(ROOT)}` ({p.stat().st_size:,} bytes)')

add('')
add('## Markdown documentation')
for p in sorted(ROOT.rglob('*.md')):
    if 'audit' in p.parts: continue
    add(f'\n### `{p.relative_to(ROOT)}`\n')
    add(p.read_text(errors='replace'))

add('')
add('## Structured result files')
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or 'audit' in p.parts: continue
    if p.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(p)
            add(f'\n### `{p.relative_to(ROOT)}`')
            add(f'- Shape: {df.shape[0]} rows x {df.shape[1]} columns')
            add(f'- Columns: {", ".join(map(str, df.columns))}')
            add('```text')
            add(df.to_string(index=False, max_rows=40))
            add('```')
        except Exception as e: add(f'- ERROR reading CSV: {e}')
    elif p.suffix.lower() == '.json':
        try:
            obj = json.loads(p.read_text(errors='replace'))
            add(f'\n### `{p.relative_to(ROOT)}`')
            add('```json')
            add(json.dumps(obj, indent=2)[:12000])
            add('```')
        except Exception as e: add(f'- ERROR reading JSON: {e}')

add('')
add('## Notebook audit')
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    try:
        nb=json.loads(p.read_text(errors='replace'))
        cells=nb.get('cells',[])
        md='\n'.join(''.join(c.get('source',[])) for c in cells if c.get('cell_type')=='markdown')
        code='\n'.join(''.join(c.get('source',[])) for c in cells if c.get('cell_type')=='code')
        outputs=sum(len(c.get('outputs',[])) for c in cells if c.get('cell_type')=='code')
        add(f'\n### `{p.relative_to(ROOT)}`')
        add(f'- Cells: {len(cells)}; code cells: {sum(c.get("cell_type")=="code" for c in cells)}; outputs: {outputs}')
        add('#### Markdown')
        add(md[:10000])
        add('#### Code excerpts / configuration tokens')
        for pat in ['model_name','learning_rate','lr','seed','budget','max_steps','num_train_epochs','modules_to_save','target_modules','lora','dora','ia3','macro','f1','memory','time','parquet','csv']:
            hits=[ln.strip() for ln in code.splitlines() if re.search(pat,ln,re.I)]
            if hits:
                add(f'**{pat}:**')
                add('\n'.join(hits[:30]))
    except Exception as e: add(f'\n### `{p.relative_to(ROOT)}` ERROR: {e}')

add('')
add('## DOCX text extraction')
for p in sorted((ROOT/'reports').glob('*.docx')):
    add(f'\n### `{p.relative_to(ROOT)}`')
    try:
        with zipfile.ZipFile(p) as z:
            xml=z.read('word/document.xml').decode('utf-8','ignore')
        text=re.sub(r'<w:tab[^>]*/>', '\\t', xml)
        text=re.sub(r'</w:p>', '\\n', text)
        text=re.sub(r'<[^>]+>', '', text)
        text=re.sub(r'&amp;','&',text)
        add(text[:50000])
    except Exception as e: add(f'ERROR: {e}')

(OUT/'audit_report.md').write_text('\n'.join(lines))
print(OUT/'audit_report.md')
print('lines', len(lines))
