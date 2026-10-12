"""Full BERT-base fine tuning, max_length=64, exactly three epochs."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, random, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import ROOT, OUT, SEED, load_data, record, write_json

class Encoded(Dataset):
    def __init__(self,texts,y,tokenizer):
        self.x=tokenizer(texts,truncation=True,padding='max_length',max_length=64,return_tensors='pt'); self.y=torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return dict(**{k:v[i] for k,v in self.x.items()},labels=self.y[i])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='google-bert/bert-base-uncased'); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--threads',type=int,default=6); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(args.threads)
    df,s=load_data(); labels=sorted(df.label.unique()); label2id={v:i for i,v in enumerate(labels)}
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    loaders={k:DataLoader(Encoded(df.iloc[s[k]].text.tolist(),[label2id[v] for v in df.iloc[s[k]].label],tokenizer),batch_size=args.batch_size,shuffle=k=='train',num_workers=0) for k in ['train','val','test']}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=len(labels),id2label=dict(enumerate(labels)),label2id=label2id).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    total_steps=3*len(loaders['train']); scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    folder=OUT/'models/bert'; folder.mkdir(parents=True,exist_ok=True)
    ckpt=folder/'resume.pt'; start=0; history=[]; best=-1.; elapsed=0.
    if args.resume and ckpt.exists():
        c=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(c['model']); optimizer.load_state_dict(c['optimizer']); scheduler.load_state_dict(c['scheduler']); start=c['epoch']; history=c['history']; best=c['best']; elapsed=c['elapsed']; torch.set_rng_state(c['rng'].cpu())
    def predict(loader):
        model.eval(); preds=[]
        with torch.inference_mode():
            for batch in loader:
                batch.pop('labels'); preds.extend(model(**{k:v.to(device) for k,v in batch.items()}).logits.argmax(-1).cpu().tolist())
        return np.array(labels)[preds]
    from sklearn.metrics import f1_score
    begin=time.perf_counter()
    for epoch in range(start,3):
        model.train(); losses=0.
        for step,batch in enumerate(loaders['train']):
            optimizer.zero_grad(set_to_none=True)
            loss=model(**{k:v.to(device) for k,v in batch.items()}).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step(); losses+=loss.item()
            if step%25==0:
                status=dict(epoch=epoch+1,step=step+1,steps_per_epoch=len(loaders['train']),loss=loss.item(),elapsed_seconds=elapsed+time.perf_counter()-begin)
                write_json(OUT/'bert_progress.json',status); print(status,flush=True)
        vp=predict(loaders['val']); f=float(f1_score(df.iloc[s['val']].label,vp,average='macro'))
        history.append(dict(epoch=epoch+1,loss=losses/len(loaders['train']),validation_macro_f1=f))
        if f>best:
            best=f; model.save_pretrained(folder/'best'); tokenizer.save_pretrained(folder/'best')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,history=history,best=best,elapsed=elapsed+time.perf_counter()-begin,rng=torch.get_rng_state()),ckpt)
        write_json(OUT/'bert_history.json',history)
    model=AutoModelForSequenceClassification.from_pretrained(folder/'best').to(device)
    record('bert',df,s,predict(loaders['val']),predict(loaders['test']),epochs=3,max_length=64,history=history,device=str(device),batch_size=args.batch_size,learning_rate=2e-5,seconds=elapsed+time.perf_counter()-begin,model='google-bert/bert-base-uncased',selection='best validation macro-F1 across 3 epochs')

if __name__=='__main__': main()
