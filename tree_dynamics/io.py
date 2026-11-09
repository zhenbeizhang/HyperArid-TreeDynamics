import csv
import json
import math
import numpy as np

def value(s):
    if s in ['True','False']:return s=='True'
    if s=='':return None
    try:return int(s)
    except ValueError:pass
    try:return float(s)
    except ValueError:return s

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:value(v) for k,v in r.items()} for r in csv.DictReader(f)]

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def write_json(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
