"""Homework tasks 1 and 2: fit representations on allowed corpora only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, time, warnings
import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from common import ROOT, OUT, SEED, tokenize, load_data, record, write_json

def select_lr(X, y, V, vy, scale=False):
    best=None; trials=[]
    for c in [.1,1.,10.]:
        clf=LogisticRegression(C=c,max_iter=2000,solver='lbfgs',random_state=SEED,tol=1e-4)
        model=make_pipeline(StandardScaler(),clf) if scale else clf
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X,y)
        f=float(f1_score(vy,model.predict(V),average='macro'))
        trials.append(dict(C=c,validation_macro_f1=f,convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in ws)))
        if best is None or f>best[0]: best=(f,c,model)
    return best[2],best[1],trials

def mean_embeddings(tokens, vectors):
    X=np.zeros((len(tokens),100),dtype=np.float32); hits=total=zero=0
    for i,doc in enumerate(tokens):
        known=[vectors[w] for w in doc if w in vectors]
        hits+=len(known); total+=len(doc)
        if known: X[i]=np.mean(known,axis=0)
        else: zero+=1
    return X,dict(token_coverage=hits/max(total,1),zero_documents=zero)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--methods',nargs='+',default=['binary','frequency','tfidf','w2v_ag','w2v_nyt']); ap.add_argument('--glove',type=str,default=str(ROOT/'cache/glove.6B.100d.txt')); args=ap.parse_args()
    df,s=load_data(); tr,va,te=[s[k] for k in ['train','val','test']]
    texts=df.text.tolist(); ys=df.label.to_numpy(); tokens=None
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    for name in args.methods:
        started=time.perf_counter(); extra={}
        if name in ['binary','frequency','tfidf']:
            cls=TfidfVectorizer if name=='tfidf' else CountVectorizer
            kw=dict(tokenizer=tokenize,token_pattern=None,lowercase=False,min_df=1)
            if name=='binary': kw['binary']=True
            vec=cls(**kw); Xtr=vec.fit_transform(df.iloc[tr].text); Xva=vec.transform(df.iloc[va].text); Xte=vec.transform(df.iloc[te].text)
            model,c,trials=select_lr(Xtr,ys[tr],Xva,ys[va])
            joblib.dump(dict(vectorizer=vec,classifier=model),OUT/'models'/f'{name}.joblib',compress=3)
            extra=dict(vocabulary_size=len(vec.vocabulary_))
        else:
            if tokens is None: tokens=[tokenize(t) for t in texts]
            if name=='glove':
                needed=set(w for doc in tokens for w in doc); vectors={}
                with open(args.glove,encoding='utf-8') as f:
                    for line in f:
                        word,_,rest=line.partition(' ')
                        if word in needed:
                            v=np.fromstring(rest,sep=' ',dtype=np.float32)
                            if v.shape==(100,): vectors[word]=v
                extra['source']='Stanford glove.6B.100d'
            else:
                corpus=[tokenize(t) for t in pd.read_csv(ROOT/'HW-1/ag.csv').text] if name=='w2v_ag' else [tokens[i] for i in tr]
                emb=Word2Vec(sentences=corpus,vector_size=100,window=5,min_count=2,sg=1,negative=5,epochs=5,workers=1,seed=SEED)
                emb.save(str(OUT/'models'/f'{name}.model')); vectors=emb.wv
                extra.update(corpus_documents=len(corpus),embedding_epochs=5,sg=1,window=5,min_count=2,workers=1)
            X,coverage=mean_embeddings(tokens,vectors); Xtr,Xva,Xte=X[tr],X[va],X[te]
            model,c,trials=select_lr(Xtr,ys[tr],Xva,ys[va],scale=True)
            joblib.dump(model,OUT/'models'/f'{name}_classifier.joblib',compress=3)
            extra.update(embedding_dimension=100,coverage_all=coverage,coverage_by_split={k:mean_embeddings([tokens[i] for i in s[k]],vectors)[1] for k in ['train','val','test']})
        record(name,df,s,model.predict(Xva),model.predict(Xte),C=c,validation_trials=trials,seconds=time.perf_counter()-started,**extra)

if __name__=='__main__': main()
