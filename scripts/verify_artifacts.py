"""Check metric consistency and assignment invariants before packaging."""
import sys,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import load_data,write_json
df,s=load_data(); expected=['binary','frequency','tfidf','glove','w2v_ag','w2v_nyt','bert'];checks=[]
for name in expected:
    p=ROOT/'artifacts/results'/f'{name}.json'
    if not p.exists(): checks.append(dict(method=name,status='pending'));continue
    result=json.loads(p.read_text()); pred=pd.read_csv(p.with_name(name+'_predictions.csv'))
    assert pred.row_id.tolist()==s['test'],name
    assert pred.truth.tolist()==df.iloc[s['test']].label.tolist(),name
    assert abs(accuracy_score(pred.truth,pred.prediction)-result['test']['accuracy'])<1e-12,name
    assert abs(f1_score(pred.truth,pred.prediction,average='macro')-result['test']['macro_f1'])<1e-12,name
    if name=='bert':
        assert result['epochs']==3 and result['max_length']==64
        assert len(result['history'])==3
    elif name in ['glove','w2v_ag','w2v_nyt']: assert result['embedding_dimension']==100
    if name=='w2v_nyt': assert result['corpus_documents']==len(s['train'])
    checks.append(dict(method=name,status='verified',test_samples=len(pred)))
cal=json.loads((ROOT/'artifacts/models/calibration.json').read_text())
assert not set(cal['calibration_rows'])&set(cal['selection_rows'])
assert set(cal['calibration_rows']+cal['selection_rows'])==set(s['val'])
source=ROOT/'third_party/acl-style-files/acl.sty'
if source.exists():assert source.read_bytes()==(ROOT/'reports/acl.sty').read_bytes()
summary=dict(checks=checks,all_complete=all(x['status']=='verified' for x in checks),calibration_disjoint=True,acl_style_unmodified=True)
write_json(ROOT/'artifacts/verification.json',summary)
print(json.dumps(summary,indent=2))
