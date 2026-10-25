import os, json, random, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

random.seed(41); np.random.seed(41); torch.manual_seed(41)
BASE='/workspace/data'; OUT='/workspace/output'; MODEL='distilbert/distilbert-base-uncased'

def read_abstracts(fn):
 d={}
 with open(fn,encoding='utf8') as f:
  for line in f:
   pmid,title,abstract=line.rstrip('\n').split('\t',2); d[pmid]=(title,abstract)
 return d

def read_anns(fn):
 d={}
 with open(fn,encoding='utf8') as f:
  for line in f:
   p,s,a,b,t,c=line.rstrip('\n').split('\t'); d.setdefault((p,s),[]).append((int(a),int(b)))
 return d

train=read_abstracts(BASE+'/training.abstracts.txt'); dev=read_abstracts(BASE+'/development.abstracts.txt'); test=read_abstracts(BASE+'/test.abstracts.txt')
ann=read_anns(BASE+'/training.annotations.txt'); dann=read_anns(BASE+'/development.annotations.txt')
tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
class Chunks(Dataset):
 def __init__(self, articles, annotations, train=True):
  self.items=[]
  for p,(title,abstract) in articles.items():
   for s,text in [('T',title),('A',abstract)]:
    anns=annotations.get((p,s),[])
    enc=tok(text,return_offsets_mapping=True,truncation=False,add_special_tokens=False)
    ids=enc['input_ids']; offs=enc['offset_mapping']
    # break to chunks, aligned with words by max length
    for start in range(0,len(ids),190):
     end=min(len(ids),start+190)
     chunkids=ids[start:end]; choffs=offs[start:end]
     labels=[]
     for a,b in choffs:
      if a==b: labels.append(-100); continue
      labels.append(1 if any(a>=x and b<=y for x,y in anns) else 0)
     # Identify first token of each mention for conventional BIO labels
     for k,(a,b) in enumerate(choffs):
      if labels[k]==1 and (k==0 or labels[k-1]!=1): labels[k]=2
     if not train and not chunkids: continue
     self.items.append((chunkids,labels,choffs,p,s))
 def __len__(self):return len(self.items)
 def __getitem__(self,i): return self.items[i]

def collate(batch):
 mx=max(len(x[0]) for x in batch); ids=[]; masks=[]; labs=[]
 for x in batch:
  n=len(x[0]); ids.append(x[0]+[tok.pad_token_id]*(mx-n)); masks.append([1]*n+[0]*(mx-n)); labs.append(x[1]+[-100]*(mx-n))
 return torch.tensor(ids),torch.tensor(masks),torch.tensor(labs)

tr=Chunks(train,ann); dv=Chunks(dev,dann)
print('chunks',len(tr),len(dv),flush=True)
model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3,id2label={0:'O',1:'I-CHEM',2:'B-CHEM'},label2id={'O':0,'I-CHEM':1,'B-CHEM':2})
# CPU training, short epochs; CUDA if present
if torch.cuda.is_available(): model.cuda()
bs=24 if torch.cuda.is_available() else 12
loader=DataLoader(tr,batch_size=bs,shuffle=True,collate_fn=collate)
steps=len(loader)*3
opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
sched=get_linear_schedule_with_warmup(opt,int(steps*.06),steps)
for ep in range(3):
 model.train(); losses=[]
 for bi,(ids,mask,labs) in enumerate(loader):
  if torch.cuda.is_available():ids,mask,labs=ids.cuda(),mask.cuda(),labs.cuda()
  loss=model(input_ids=ids,attention_mask=mask,labels=labs).loss
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  opt.step();sched.step();opt.zero_grad();losses.append(loss.item())
  if bi%500==0: print('ep',ep,'batch',bi,'loss',round(loss.item(),4),flush=True)
 print('epoch',ep,'avg',np.mean(losses),flush=True)
 torch.save(model.state_dict(),OUT+f'/model_ep{ep}.pt')
