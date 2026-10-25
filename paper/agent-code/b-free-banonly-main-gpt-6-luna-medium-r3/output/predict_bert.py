import os,sys,json,collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATA=sys.argv[1] if len(sys.argv)>1 else '/home/agent'
MODEL=sys.argv[2] if len(sys.argv)>2 else '/home/agent/bert_best'
OUT=sys.argv[3] if len(sys.argv)>3 else '/home/agent/predictions.tsv'
MAXLEN=512;STRIDE=128;BATCH=16
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(8,os.cpu_count() or 1))

def read_articles(path):
 d={}
 with open(path,encoding='utf8') as f:
  for line in f:
   p,t,a=line.rstrip('\n').split('\t',2);d[p]=(t,a)
 return d

def make_chunks(tok,text):
 enc=tok(text,truncation=True,max_length=MAXLEN,stride=STRIDE,return_overflowing_tokens=True,return_offsets_mapping=True)
 out=[]
 for i in range(len(enc['input_ids'])):
  x={k:enc[k][i] for k in ('input_ids','attention_mask')}
  if 'token_type_ids' in enc:x['token_type_ids']=enc['token_type_ids'][i]
  x['offset_mapping']=enc['offset_mapping'][i]
  out.append(x)
 return out

class Collate:
 def __init__(self,tok):self.tok=tok
 def __call__(self,items):
  offsets=[x['offset_mapping'] for x in items]
  feats=[{k:v for k,v in x.items() if k!='offset_mapping'} for x in items]
  return self.tok.pad(feats,padding=True,return_tensors='pt'),offsets

def main():
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
 model=AutoModelForTokenClassification.from_pretrained(MODEL).to(DEVICE).eval()
 bias=json.load(open(os.path.join(MODEL,'best.json')))['bias'] if os.path.exists(os.path.join(MODEL,'best.json')) else 0
 articles=read_articles(os.path.join(DATA,'test.abstracts.txt'))
 items=[];chunk_to_doc=[];docs=[]
 for p,(title,abstract) in articles.items():
  for sec,text in [('T',title),('A',abstract)]:
   docs.append((p,sec,text));cs=make_chunks(tok,text);items.extend(cs);chunk_to_doc.extend([len(docs)-1]*len(cs))
 loader=DataLoader(items,batch_size=BATCH,shuffle=False,collate_fn=Collate(tok))
 sums=[{} for _ in docs];counts=[collections.Counter() for _ in docs]
 with torch.no_grad():
  cursor=0
  for batch,offsets in loader:
   batch={k:v.to(DEVICE) for k,v in batch.items()}
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if DEVICE.type=='cuda' else torch.autocast('cpu',enabled=False)
   with ctx:logits=model(**batch).logits.float().cpu().numpy()
   for b,row in enumerate(offsets):
    docidx=chunk_to_doc[cursor];cursor+=1
    for (s,e),lg in zip(row,logits[b]):
     if e<=s:continue
     key=(s,e); sums[docidx][key]=sums[docidx].get(key,np.zeros(3,dtype=np.float64))+lg;counts[docidx][key]+=1
 preds=[]
 for (p,sec,_),dct,cnt in zip(docs,sums,counts):
  toks=[(s,e,dct[(s,e)]/cnt[(s,e)]) for s,e in sorted(dct)]
  labels=[]
  for s,e,lg in toks:
   lg=lg.copy();lg[1:]+=bias;labels.append(int(lg.argmax()))
  active=None;end=None
  for (s,e,_),lab in zip(toks,labels):
   if lab==1:
    if active is not None:preds.append((p,sec,active,end))
    active=s;end=e
   elif lab==2:
    if active is None:active=s
    end=e
   elif active is not None:
    preds.append((p,sec,active,end));active=None;end=None
  if active is not None:preds.append((p,sec,active,end))
 with open(OUT,'w') as f:
  for p,sec,s,e in sorted(set(preds)):f.write(f'{p}\t{sec}:{s}:{e}\n')
 print('wrote',len(set(preds)),'mentions to',OUT,flush=True)
 bucket=os.getenv('GCS_BUCKET','')
 if bucket:
  from google.cloud import storage
  client=storage.Client();client.bucket(bucket).blob('predictions/predictions.tsv').upload_from_filename(OUT)
  print('uploaded predictions to gs://'+bucket+'/predictions/predictions.tsv',flush=True)
if __name__=='__main__':main()
