"""Independent project contribution: calibration, selective prediction and explanations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from time import perf_counter
import json, joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, accuracy_score, f1_score
from common import ROOT, OUT, SEED, load_data, scores, tokenize, write_json

def ece(y,p,labels,bins=10):
    correct=np.asarray(labels)[p.argmax(1)]==np.asarray(y); conf=p.max(1); value=0.
    for lo in np.linspace(0,1,bins,endpoint=False):
        mask=(conf>=lo)&(conf<lo+1/bins if lo+1/bins<1 else conf<=1)
        if mask.any(): value+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return float(value)

def main():
    df,s=load_data(); bundle=joblib.load(OUT/'models/tfidf.joblib'); vec,clf=bundle['vectorizer'],bundle['classifier']; labels=clf.classes_
    Xv=vec.transform(df.iloc[s['val']].text); Xt=vec.transform(df.iloc[s['test']].text)
    yv=df.iloc[s['val']].label.to_numpy(); yt=df.iloc[s['test']].label.to_numpy(); zv=clf.decision_function(Xv); zt=clf.decision_function(Xt)
    # Split the validation set into calibration and threshold-selection subsets.
    ids=np.random.RandomState(SEED).permutation(len(yv)); cal,sel=ids[:len(ids)//2],ids[len(ids)//2:]
    opt=minimize_scalar(lambda t:log_loss(yv[cal],softmax(zv[cal]/t,axis=1),labels=labels),bounds=(.1,5.),method='bounded'); temperature=float(opt.x)
    pv=softmax(zv/temperature,axis=1); pt=softmax(zt/temperature,axis=1)
    grid=np.linspace(.34,.999,200); choices=[]
    for threshold in grid:
        keep=pv[sel].max(1)>=threshold
        if keep.mean()>=.2 and accuracy_score(yv[sel][keep],labels[pv[sel][keep].argmax(1)])>=.99: choices.append((float(keep.mean()),float(threshold)))
    threshold=max(choices,key=lambda x:(x[0],-x[1]))[1] if choices else 1.01
    write_json(OUT/'models/calibration.json',dict(temperature=temperature,threshold=threshold,target_validation_accuracy=.99,calibration_rows=[s['val'][i] for i in cal],selection_rows=[s['val'][i] for i in sel],selected_on='validation only'))
    curves=[]
    for t in [.0,.5,.7,.8,.9,.95,.99,threshold]:
        keep=pt.max(1)>=t
        curves.append(dict(threshold=t,coverage=float(keep.mean()),accepted=int(keep.sum()),accuracy=float(accuracy_score(yt[keep],labels[pt[keep].argmax(1)])) if keep.any() else None))
    raw=softmax(zt,axis=1)
    audit=dict(total=len(df),counts=df.label.value_counts().to_dict(),duplicates=int(df.text.duplicated().sum()),split_counts={k:df.iloc[s[k]].label.value_counts().to_dict() for k in ['train','val','test']})
    traintexts=set(df.iloc[s['train']].text); unseen=~df.iloc[s['test']].text.isin(traintexts).to_numpy()
    audit['test_rows_seen_in_train']=int((~unseen).sum()); audit['val_rows_seen_in_train']=int(df.iloc[s['val']].text.isin(traintexts).sum())
    ag=pd.read_csv(ROOT/'HW-1/ag.csv')
    agtexts=set(ag.text)
    audit['ag_documents']=len(ag)
    audit['ag_exact_overlap_by_split']={k:int(df.iloc[s[k]].text.isin(agtexts).sum()) for k in ['train','val','test']}
    write_json(OUT/'data_audit.json',audit)
    # Explanation is an exact additive decomposition of winner-minus-runner-up logit.
    names=vec.get_feature_names_out(); pred=clf.predict(Xt); rng=np.random.RandomState(SEED)
    erased_top=[]; erased_random=[]; margins=[]; chosen=[]
    for i in range(min(200,Xt.shape[0])):
        row=Xt[i]; rank=np.argsort(zt[i]); winner,runner=rank[-1],rank[-2]
        contributions=row.data*(clf.coef_[winner,row.indices]-clf.coef_[runner,row.indices]); order=np.argsort(contributions)[::-1]
        positive=order[contributions[order]>0][:5]; k=len(positive)
        top=row.copy(); randomrow=row.copy()
        top.data[positive]=0
        if k: randomrow.data[rng.choice(len(row.data),k,replace=False)]=0
        erased_top.append(clf.predict(top)[0]); erased_random.append(clf.predict(randomrow)[0])
        reconstructed=float(contributions.sum()+clf.intercept_[winner]-clf.intercept_[runner]); margins.append(abs(reconstructed-float(zt[i,winner]-zt[i,runner])))
        if i<5: chosen.append(dict(row_id=s['test'][i],truth=str(yt[i]),prediction=str(pred[i]),runner_up=str(labels[runner]),terms=[dict(term=str(names[row.indices[j]]),contribution=float(contributions[j])) for j in positive]))
    n=len(erased_top)
    times=[]
    for text in df.iloc[s['test'][:100]].text:
        t=perf_counter(); clf.predict_proba(vec.transform([text])); times.append((perf_counter()-t)*1000)
    # Fixed diagnostic, not used to choose any model setting.
    short=vec.transform([' '.join(tokenize(t)[:64]) for t in df.iloc[s['test']].text])
    majority=df.iloc[s['train']].label.value_counts().idxmax()
    result=dict(majority_baseline=dict(label=majority,**scores(yt,np.repeat(majority,len(yt)),labels)),temperature=temperature,selected_threshold=threshold,calibration_n=len(cal),threshold_selection_n=len(sel),
        calibration=dict(raw_nll=float(log_loss(yt,raw,labels=labels)),scaled_nll=float(log_loss(yt,pt,labels=labels)),raw_ece=ece(yt,raw,labels),scaled_ece=ece(yt,pt,labels)),
        selective_curve=curves,unseen_test=scores(yt[unseen],pred[unseen],labels),head64=scores(yt,clf.predict(short),labels),
        explanation=dict(n=n,top5_flip_rate=float(np.mean(np.array(erased_top)!=pred[:n])),random5_flip_rate=float(np.mean(np.array(erased_random)!=pred[:n])),max_reconstruction_error=max(margins),examples=chosen,protocol='Zero up to five positive TF-IDF feature contributions without renormalization; fixed 200 test rows.'),
        latency_ms=dict(n=len(times),median=float(np.median(times)),p95=float(np.percentile(times,95)),scope='TF-IDF transform + predict_proba, one document, CPU; HTTP excluded'))
    write_json(OUT/'project_results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
