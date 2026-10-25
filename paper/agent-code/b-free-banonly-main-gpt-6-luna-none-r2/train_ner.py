import os, json, random, math, sys, time
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from collections import defaultdict

ROOT='/workspace'; MODEL=ROOT+'/model'; random.seed(42); torch.manual_seed(42); torch.set_num_threads(6)

def read_data(name):
 texts={}
 with open(f'{ROOT}/data/{name}.abstracts.txt',encoding='utf8') as f:
  for line in f:
   p,t,a=line.rstrip('\n').split('\t');texts[p]={'T':t,'A':a}
 anns=defaultdict(list)
 fn=f'{ROOT}/data/{name}.annotations.txt'
 if os.path.exists(fn):
  with open(fn,encoding='utf8') as f:
   for l in f:
    p,s,b,e,txt,cl=l.rstrip('\n').split('\t');anns[(p,s)].append((int(b),int(e)))
 samples=[]
 for p,ss in texts.items():
  for s,text in ss.items():
   samples.append((p,s,text,anns[(p,s)]))
 return samples

tok=BertTokenizer.from_pretrained(MODEL,local_files_only=True,do_lower_case=False)
# tokenize with character offsets via basic tokenizer and wordpiece per basic token
# BertTokenizer has tokenize but recover offsets by incremental basic tokenization

def encode(text, ents, max_len=254):
 # use basic tokenizer with offsets
 words=[]
 for token in []:
  # locate casefold/normalized token sequentially; unicode punctuation split etc
  pass
 # use regex word tokenizer aligned chars. WordPiece per regex lexical tokens, fallback punctuation/whitespace.
 import re
 spans=[m.span() for m in re.finditer(r"\w+|[^\w\s]",text,flags=re.UNICODE)]
 pieces=[]; labels=[]; offsets=[]
 ent_i=0; ents=sorted(ents); # labels B/I based on offsets
 for st,en in spans:
  word=text[st:en]
  wp=tok.tokenize(word)
  if not wp: continue
  for k,x in enumerate(wp):
   pieces.append(x); offsets.append((st,en))
 # convert token ids; truncate strategy centered? standard first 510; long abstract split needed
 ids=tok.convert_tokens_to_ids(pieces)
 y=[]
 for st,en in offsets:
  e=next((x for x in ents if x[0]<=st and en<=x[1]),None)
  # subword start should B if starts at entity start
  if e is None:y.append(0)
  elif st==e[0]:y.append(1)
  else:y.append(2)
 return ids,y,offsets

class NERDataset(Dataset):
 def __init__(self,samples):
  self.rows=[]
  for p,s,text,ents in samples:
   ids,labels,offs=encode(text,ents)
   # overlapping windows, entity mentions are usually short
   step=200
   starts=list(range(0,max(1,len(ids)),step))
   if len(ids)<=510: starts=[0]
   for st in starts:
    en=min(st+254,len(ids))
    if st>=len(ids):continue
    self.rows.append((ids[st:en],labels[st:en]))
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.rows[i]

def collate(batch):
 L=max(len(x[0]) for x in batch); L=((L+7)//8)*8
 ids=torch.zeros((len(batch),L),dtype=torch.long); mask=torch.zeros_like(ids); y=torch.full_like(ids,-100)
 for i,(a,b) in enumerate(batch):
  n=len(a);ids[i,:n]=torch.tensor(a);mask[i,:n]=1;y[i,:n]=torch.tensor(b)
 return ids,mask,y

train=NERDataset(read_data('training')); dev_samples=read_data('development')
print('rows',len(train),flush=True)
model=BertForTokenClassification.from_pretrained(MODEL,num_labels=3,local_files_only=True)
model.config.id2label={0:'O',1:'B-CHEM',2:'I-CHEM'};model.config.label2id={'O':0,'B-CHEM':1,'I-CHEM':2}
opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
loader=DataLoader(train,batch_size=2,shuffle=True,collate_fn=collate,num_workers=0)
steps=len(loader)
sched=get_linear_schedule_with_warmup(opt,int(.06*steps),steps)
model.train(); start=time.time()
for ep in range(1):
 total=0
 for it,(ids,mask,y) in enumerate(loader):
  opt.zero_grad(set_to_none=True)
  out=model(input_ids=ids,attention_mask=mask,labels=y)
  loss=out.loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();total+=loss.item()
  if (it+1)%200==0:print('epoch',ep+1,'iter',it+1,'loss',total/(it+1),'minutes',(time.time()-start)/60,flush=True)
 print('epoch',ep+1,'loss',total/len(loader),'time',(time.time()-start)/60,flush=True)
 torch.save(model.state_dict(),ROOT+f'/model/finetuned.pt')
