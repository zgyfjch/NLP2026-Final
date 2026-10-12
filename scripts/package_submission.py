"""Create reviewable local hand-in archives; never publishes or submits."""
from pathlib import Path
import json,zipfile,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'submission';DEST.mkdir(exist_ok=True)
verification=json.loads((ROOT/'artifacts/verification.json').read_text())
if not verification['all_complete']: raise SystemExit('Refusing final archive: at least one required experiment is incomplete.')
for p in (ROOT/'cache/bert-base-uncased').glob('provenance.json'):
    shutil.copy2(p,ROOT/'artifacts/model_provenance.json')
manifest={}
for rel in ['HW-1/nyt.csv','HW-1/ag.csv','cache/glove.6B.100d.txt','cache/bert-base-uncased/model.safetensors','reports/acl.sty']:
    p=ROOT/rel;h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    manifest[rel]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
(ROOT/'artifacts/resource_manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(ROOT/'reports/TeamXX_Final_Report.pdf',ROOT/'NLP2026_TeamXX_Demo/Final/TeamXX_Final_Report.pdf')
include=[]
for folder in ['src','templates','tests','scripts','HW-1','artifacts/results','artifacts/figures','artifacts/screenshots','reports','NLP2026_TeamXX_Demo']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.aux','.out','.zip'] and p.name!='.DS_Store':include.append(p)
for name in ['README.md','DEFENSE_GUIDE.md','PROVENANCE.md','requirements.txt','requirements-lock.txt','Launch_Demo.ps1','Run_Experiments.ps1','SUBMISSION_GUIDE.md']:
    p=ROOT/name
    if p.exists():include.append(p)
include.extend((ROOT/'artifacts').glob('*.json'))
# Small lexical bundles make the demonstration runnable without retraining.
for name in ['binary.joblib','frequency.joblib','tfidf.joblib','calibration.json']:
    include.append(ROOT/'artifacts/models'/name)
include=sorted(set(include))
archive=DEST/'NLP2026_2312260_TanYi.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in include:z.write(p,p.relative_to(ROOT))
code=ROOT/'NLP2026_TeamXX_Demo/Final/TeamXX_Code.zip'
with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in include:
        if 'NLP2026_TeamXX_Demo' not in p.parts and p.suffix!='.pdf':z.write(p,p.relative_to(ROOT))
print(archive,archive.stat().st_size)
print(code,code.stat().st_size)
