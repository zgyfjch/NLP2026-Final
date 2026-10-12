"""Scientific figures use recorded results, never hand-entered scores."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts'; F=A/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
names=[];acc=[];f1=[]
for name in ['binary','frequency','tfidf','glove','w2v_ag','w2v_nyt','bert']:
    p=A/'results'/f'{name}.json'
    if p.exists():
        r=json.loads(p.read_text());names.append(name);acc.append(r['test']['accuracy']);f1.append(r['test']['macro_f1'])
x=np.arange(len(names));fig,ax=plt.subplots(figsize=(8,2.7));ax.bar(x-.18,acc,.36,label='Accuracy',color='#245f88');ax.bar(x+.18,f1,.36,label='Macro-F1',color='#67a6a1');ax.set_xticks(x,names);ax.set_ylim(0,1.07);ax.set_ylabel('Score');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(F/'classification.png');plt.close(fig)
pr=json.loads((A/'project_results.json').read_text()); curve=pr['selective_curve'][1:7]
fig,axs=plt.subplots(1,2,figsize=(8,2.7));axs[0].plot([r['coverage']*100 for r in curve],[r['accuracy']*100 for r in curve],'o-',color='#245f88');axs[0].set_xlabel('Coverage (%)');axs[0].set_ylabel('Accepted accuracy (%)');axs[0].grid(alpha=.2)
axs[1].bar(['Raw','Temperature scaled'],[pr['calibration']['raw_ece'],pr['calibration']['scaled_ece']],color=['#b68058','#245f88']);axs[1].set_ylabel('ECE (10 bins)');fig.tight_layout();fig.savefig(F/'calibration.png');plt.close(fig)
r=json.loads((A/'results/tfidf.json').read_text());m=np.array(r['test']['confusion_matrix']);fig,ax=plt.subplots(figsize=(4,3));ax.imshow(m,cmap='Blues');ax.set_xticks(range(3),r['labels']);ax.set_yticks(range(3),r['labels']);ax.set_xlabel('Predicted');ax.set_ylabel('True')
for i in range(3):
 for j in range(3): ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if m[i,j]>m.max()/2 else 'black')
fig.tight_layout();fig.savefig(F/'confusion.png');plt.close(fig)
