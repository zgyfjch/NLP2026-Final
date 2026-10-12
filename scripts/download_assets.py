"""Download official assets; honors HTTPS_PROXY and resumes partial files."""
from pathlib import Path
import requests, zipfile, hashlib, json, concurrent.futures, time
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cache'; CACHE.mkdir(exist_ok=True)

def download(url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return path
    part=path.with_suffix(path.suffix+'.part'); size=part.stat().st_size if part.exists() else 0
    for attempt in range(8):
        size=part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url,headers={'Range':f'bytes={size}-'} if size else {},stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if size and r.status_code!=206: size=0
                with part.open('ab' if size else 'wb') as f:
                    for b in r.iter_content(1024*1024): f.write(b)
            break
        except (requests.RequestException,OSError) as e:
            if attempt==7: raise
            print('Retrying interrupted download',path.name,'attempt',attempt+1,flush=True)
            time.sleep(min(2**attempt,15))
    part.replace(path); print('Downloaded',path.name,path.stat().st_size,flush=True); return path

def bert():
    api=requests.get('https://huggingface.co/api/models/google-bert/bert-base-uncased',timeout=30); api.raise_for_status(); rev=api.json()['sha']
    dest=CACHE/'bert-base-uncased'; dest.mkdir(exist_ok=True)
    for name in ['config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']:
        download(f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{rev}/{name}',dest/name)
    (dest/'provenance.json').write_text(json.dumps(dict(repository='google-bert/bert-base-uncased',revision=rev),indent=2))

def glove():
    z=download('https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip',CACHE/'glove.6B.zip')
    with zipfile.ZipFile(z) as f: f.extract('glove.6B.100d.txt',CACHE)
    print('Extracted GloVe 100d',flush=True)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        tasks=[pool.submit(bert),pool.submit(glove)]
        for task in tasks: task.result()
