"""Descriptive input-length audit; no model selection or fitting."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import load_data,tokenize,write_json
df,s=load_data();tok=Tokenizer.from_file(str(ROOT/'cache/bert-base-uncased/tokenizer.json'));tok.no_truncation();tok.no_padding()
lengths=[]
for start in range(0,len(df),128):
    lengths.extend(len(e.ids) for e in tok.encode_batch(df.text.iloc[start:start+128].tolist()))
lengths=np.asarray(lengths)
def stats(x):return dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),fraction_over_64=float(np.mean(np.asarray(x)>64)))
ag=pd.read_csv(ROOT/'HW-1/ag.csv')
audit=dict(wordpiece_includes_special_tokens=True,nyt_wordpiece_by_split={k:stats(lengths[s[k]]) for k in ['train','val','test']},nyt_regex=stats([len(tokenize(t)) for t in df.text]),ag_regex=stats([len(tokenize(t)) for t in ag.text]),conflicting_exact_text_labels=int((df.groupby('text').label.nunique()>1).sum()))
write_json(ROOT/'artifacts/length_audit.json',audit);print(audit)
