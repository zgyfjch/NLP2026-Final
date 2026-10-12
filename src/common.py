"""Shared data contract. Test rows never fit vocabulary, embeddings or hyperparameters."""
from pathlib import Path
import hashlib, json, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
SEED = 42

def tokenize(text):
    return TOKEN.findall(str(text).lower())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_data():
    path = ROOT / 'HW-1/nyt.csv'
    df = pd.read_csv(path)
    if df[['text', 'label']].isna().any().any():
        raise ValueError('Missing text or labels; resolve explicitly before splitting.')
    df['label'] = df.label.astype(str)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    splitpath = OUT / 'split.json'
    if splitpath.exists():
        split = json.loads(splitpath.read_text(encoding='utf-8'))
        if split['sha256'] != digest: raise ValueError('Dataset changed.')
    else:
        ids = np.random.RandomState(SEED).permutation(len(df))
        a,b = int(.8*len(df)),int(.9*len(df))
        split = dict(seed=SEED, sha256=digest, train=ids[:a].tolist(), val=ids[a:b].tolist(), test=ids[b:].tolist())
        write_json(splitpath, split)
    groups=[set(split[k]) for k in ['train','val','test']]
    assert sum(map(len,groups))==len(df) and not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    return df, split

def scores(y, p, labels):
    return dict(accuracy=float(accuracy_score(y,p)), macro_f1=float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
                per_class=classification_report(y,p,labels=labels,output_dict=True,zero_division=0),
                confusion_matrix=confusion_matrix(y,p,labels=labels).tolist())

def record(name, df, split, valpred, testpred, **metadata):
    labels=sorted(df.label.unique())
    result=dict(method=name, labels=labels, completed_at_utc=datetime.now(timezone.utc).isoformat(), validation=scores(df.iloc[split['val']].label,valpred,labels),
                test=scores(df.iloc[split['test']].label,testpred,labels), **metadata)
    write_json(OUT/'results'/f'{name}.json',result)
    pd.DataFrame({'row_id':split['test'],'truth':df.iloc[split['test']].label.to_numpy(),'prediction':testpred}).to_csv(OUT/'results'/f'{name}_predictions.csv',index=False)
    print(name, result['test']['accuracy'], result['test']['macro_f1'], flush=True)
    return result
