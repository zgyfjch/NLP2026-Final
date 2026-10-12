"""Local demo. No external inference API and no upload of submitted text."""
import json, joblib
import numpy as np
from scipy.special import softmax
from flask import Flask, request, jsonify, render_template
from common import ROOT,OUT,tokenize

app=Flask(__name__,template_folder=str(ROOT/'templates'))
app.config['MAX_CONTENT_LENGTH']=256*1024
MODELS={name:joblib.load(OUT/'models'/f'{name}.joblib') for name in ['binary','frequency','tfidf']}
CAL=json.loads((OUT/'models/calibration.json').read_text())

def infer(text,method='tfidf'):
    if not isinstance(text,str) or not text.strip(): raise ValueError('请输入英文新闻文本。')
    if len(text)>30000: raise ValueError('单篇最多 30000 个字符。')
    if not isinstance(method,str) or method not in MODELS: raise ValueError('未知模型。')
    bundle=MODELS[method]; vec,clf=bundle['vectorizer'],bundle['classifier']; x=vec.transform([text]); tokens=tokenize(text)
    coverage=sum(t in vec.vocabulary_ for t in tokens)/max(len(tokens),1)
    if x.nnz==0: return dict(abstain=True,reason='没有可识别的英文词汇，请输入英文新闻。',probabilities={},terms=[],coverage=coverage)
    z=clf.decision_function(x)[0]; temperature=CAL['temperature'] if method=='tfidf' else 1.; p=softmax(z/temperature)
    order=np.argsort(z); winner,runner=order[-1],order[-2]; diff=clf.coef_[winner]-clf.coef_[runner]; vals=x.data*diff[x.indices]; top=np.argsort(vals)[::-1][:10]; names=vec.get_feature_names_out()
    threshold=CAL['threshold'] if method=='tfidf' else .9
    reject=bool(p[winner]<threshold or coverage<.05)
    return dict(prediction=str(clf.classes_[winner]),runner_up=str(clf.classes_[runner]),confidence=float(p[winner]),abstain=reject,reason='置信度或词汇覆盖不足，建议人工复核。' if reject else '通过当前复核阈值。',
        probabilities={str(k):float(v) for k,v in zip(clf.classes_,p)},coverage=coverage,temperature=temperature,threshold=threshold,
        terms=[dict(term=str(names[x.indices[j]]),contribution=float(vals[j])) for j in top if vals[j]>0],
        margin=float(z[winner]-z[runner]),reconstructed_margin=float(vals.sum()+clf.intercept_[winner]-clf.intercept_[runner]))

@app.get('/')
def home(): return render_template('index.html')

@app.get('/api/health')
def health(): return jsonify(status='ok',models=list(MODELS))

@app.post('/api/predict')
def predict():
    data=request.get_json(silent=True)
    if not isinstance(data,dict): return jsonify(error='请求必须是 JSON 对象。'),400
    try: return jsonify(infer(data.get('text'),data.get('model','tfidf')))
    except ValueError as e: return jsonify(error=str(e)),400

@app.get('/api/results')
def results():
    return jsonify(project=json.loads((OUT/'project_results.json').read_text(encoding='utf-8')),models=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'results').glob('*.json')])

if __name__=='__main__': app.run(host='127.0.0.1',port=8765,debug=False)
