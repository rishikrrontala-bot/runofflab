"""Prepare a private Adaption upload from fully generated TRAIN families only."""
import hashlib,json
from pathlib import Path
rows=[]
for path in sorted(Path('models/data/cases').glob('*.json')):
    value=json.loads(path.read_text())
    if value['split']=='train':rows.append(value)
if len(rows)<1000:raise RuntimeError(f'Only {len(rows)} qualified training rows; minimum 1000')
families={r['family'] for r in rows}
assert all(f'synthetic-{i:02}' in families for i in range(1,11)), 'Missing training family'
assert not any(int(r['family'].split('-')[1])>10 for r in rows), 'Split leakage'
assert len({r['prompt'] for r in rows})==len(rows), 'Duplicate physical input'
assert all(r['searchCoverage']['evaluated']==r['searchCoverage']['feasible'] for r in rows)
path=Path('models/data/runofflab-sft-train.jsonl')
path.write_text('\n'.join(json.dumps({'prompt':r['prompt'],'completion':r['completion']},separators=(',',':')) for r in rows)+'\n')
manifest={'dataset':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),'families':sorted(families),'realRows':0,'syntheticRows':len(rows),'showcaseInTraining':False,'columnMapping':{'prompt':'prompt','completion':'completion'},'solver':rows[0]['solver'],'label':'best evaluated feasible design, exhaustive feasible subsets','visibility':'private','resolvedTrainingMethodRequired':'instruction/SFT','paidJobStatus':'not launched'}
Path('models/upload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
