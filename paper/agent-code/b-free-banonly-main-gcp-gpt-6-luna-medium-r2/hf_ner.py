import argparse, json, os, random, re, math, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LABELS=['O','B','I']

def read_docs(abstract_file, ann_file=None):
 docs=[]
 with open(abstract_file,encoding='utf8') as f:
  for line in f:
   pmid,title,abstract=line.rstrip('\n').split('\t',2)
   docs.append({'pmid':pmid,'T':title,'A':abstract,'anns':[]})
 if ann_file:
  by={d['pmid']:d for d in docs}
  with open(ann_file,encoding='utf8') as f:
   for line in f:
    p=line.rstrip('\n').split('\t'); by[p[0]]['anns'].append((p[1],int(p[2]),int(p[3])))
 return docs

class Encoded(Dataset):
 def __init__(self, docs, tok, max_length=512, stride=96, labeled=True):
  self.examples=[]; self.meta=[]; self.labeled=labeled
  for d in docs:
   for sec in 'TA':
    text=d[sec]
    anns=sorted((a,b) for s,a,b in d['anns'] if s==sec)
    enc=tok(text, max_length=max_length, stride=stride, truncation=True,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    for ci, offsets in enumerate(enc['offset_mapping']):
     item={k:enc[k][ci] for k in tok.model_input_names if k in enc}
     self.meta.append((d['pmid'],sec,offsets,text))
     if labeled:
      labels=[]; last_ann=-1; active=-1
      for a,b in offsets:
       if a==b:
        labels.append(-100); continue
       while last_ann+1<len(anns) and anns[last_ann+1][1] <= a: last_ann+=1
       hit=-1
       for ai in range(max(0,last_ann),min(len(anns),last_ann+3)):
        x,y=anns[ai]
        if x < b and y > a:
         hit=ai; break
        if x>=b: break
       if hit<0:
        labels.append(0); active=-1
       elif hit==active:
        labels.append(2)
       else:
        labels.append(1); active=hit
      item['labels']=labels
     self.examples.append(item)
 def __len__(self): return len(self.examples)
 def __getitem__(self,i): return self.examples[i]

def make_collate(tok, labeled=True):
 def collate(batch):
  metas=[x.pop('_idx',None) for x in batch]
  return tok.pad(batch,padding=True,return_tensors='pt')
 return collate

# custom collator leaves batch items immutable, metadata indices travel separately
def collate_items(batch, tok, labeled=True):
 idx=[x['idx'] for x in batch]
 labels=[x.get('labels') for x in batch]
 features=[]
 for x in batch:
  features.append({k:v for k,v in x.items() if k not in ('idx','labels')})
 padded=tok.pad(features,padding=True,return_tensors='pt')
 if labels[0] is not None:
  length=padded['input_ids'].shape[1]
  padded_labels=[]
  for lab in labels:
   pad=length-len(lab)
   padded_labels.append(lab+([-100]*pad if tok.padding_side=='right' else [-100]*pad))
  padded['labels']=torch.tensor(padded_labels,dtype=torch.long)
 return padded,idx

def batches(ds,tok,batch_size,shuffle=False):
 order=list(range(len(ds)))
 if shuffle: random.shuffle(order)
 for st in range(0,len(order),batch_size):
  ids=order[st:st+batch_size]
  items=[]
  for ix in ids:
   ex=dict(ds[ix]); ex['idx']=ix; items.append(ex)
  yield collate_items(items,tok)

def encode_dataset(docs,tok,max_length=512,stride=96,labeled=True):
 ds=Encoded(docs,tok,max_length,stride,labeled)
 # indices included by iterator, keeping separate metadata
 return ds

def get_spans_for_batch(model,ds,tok,device,batch_size=8):
 model.eval(); found={}
 with torch.no_grad():
  for inp,ids in batches(ds,tok,batch_size,False):
   inp={k:v.to(device) for k,v in inp.items() if k in ('input_ids','attention_mask','token_type_ids')}
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(device.type=='cuda')):
    logits=model(**inp).logits
   probs=torch.softmax(logits.float(),-1).cpu().numpy(); pred=probs.argmax(-1)
   for row,ix in enumerate(ids):
    pm,sec,offsets,text=ds.meta[ix]
    labels=pred[row,:len(offsets)]
    pp=probs[row,:len(offsets)]
    i=0
    while i<len(offsets):
     if offsets[i][0]==offsets[i][1] or labels[i]==0:
      i+=1; continue
     if labels[i]==2 and i>0 and labels[i-1] in (1,2):
      i+=1; continue
     start_i=i; i+=1
     while i<len(offsets) and offsets[i][0]!=offsets[i][1] and labels[i]==2:
      i+=1
     st=offsets[start_i][0]; en=offsets[i-1][1]
     if en<=st: continue
     # confidence averages predicted entity-class probability across subwords
     conf=float(np.mean([max(pp[k,1],pp[k,2]) for k in range(start_i,i)]))
     key=(pm,sec,st,en)
     if key not in found or conf>found[key]: found[key]=conf
  return found

def score(pred,gold):
 tp=len(pred&gold); p=tp/max(1,len(pred)); r=tp/len(gold); f=2*p*r/(p+r) if p+r else 0
 return p,r,f

def best_threshold(found,gold):
 best=(0,0,None)
 for th in [i/100 for i in range(5,96)]:
  pred={k for k,v in found.items() if v>=th}
  p,r,f=score(pred,gold)
  if f>best[0]: best=(f,th,(len(pred),p,r))
 return best

def train_loop(train_docs, dev_docs, args, final=False):
 random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
 if args.resume and os.path.isdir(args.resume): model=AutoModelForTokenClassification.from_pretrained(args.resume,num_labels=3,id2label={i:x for i,x in enumerate(LABELS)},label2id={x:i for i,x in enumerate(LABELS)})
 else: model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3,id2label={i:x for i,x in enumerate(LABELS)},label2id={x:i for i,x in enumerate(LABELS)})
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
 tr=encode_dataset(train_docs,tok,args.max_length,args.stride,True)
 dv=encode_dataset(dev_docs,tok,args.max_length,args.stride,False) if dev_docs else None
 print('samples',len(tr),'dev',len(dv) if dv else 0,'device',device,flush=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
 steps=max(1,math.ceil(len(tr)/args.batch_size)*args.epochs); warmup=int(steps*.06)
 def lr_lambda(step):
  if step<warmup: return step/max(1,warmup)
  return max(0.,(steps-step)/max(1,steps-warmup))
 sched=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
 scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda'))
 bestf=0; bestth=.5
 for epoch in range(args.epochs):
  model.train(); random.shuffle(tr.examples); losses=[]; optimizer.zero_grad(set_to_none=True)
  start=time.time()
  for bi,(inp,ids) in enumerate(batches(tr,tok,args.batch_size,False)):
   inp={k:v.to(device) for k,v in inp.items()}
   labels=inp.pop('labels')
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(device.type=='cuda')):
    logits=model(**inp).logits
    loss=torch.nn.functional.cross_entropy(logits.view(-1,3),labels.view(-1),ignore_index=-100,weight=torch.tensor([1.0,args.b_weight,1.0],device=device))
   scaler.scale(loss).backward()
   scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
   scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); sched.step()
   losses.append(float(loss.detach().cpu()))
   if bi and bi%200==0: print('epoch',epoch+1,'step',bi,'/',math.ceil(len(tr)/args.batch_size),'loss',round(np.mean(losses[-50:]),4),flush=True)
  print('epoch',epoch+1,'train loss',round(np.mean(losses),4),'seconds',round(time.time()-start,1),flush=True)
  if dv:
   found=get_spans_for_batch(model,dv,tok,device,args.eval_batch_size)
   gold={(d['pmid'],s,a,b) for d in dev_docs for s,a,b in d['anns']}
   best=best_threshold(found,gold); print('DEV',epoch+1,'spans',len(found),'best',best,'th .5',score({k for k,v in found.items() if v>=.5},gold),flush=True)
   if best[0]>bestf:
    bestf=best[0]; bestth=best[1]
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    with open(args.out+'/dev_predictions.json','w') as f: json.dump({'threshold':bestth,'preds':[[*k,v] for k,v in found.items()]},f)
  else:
   model.save_pretrained(args.out); tok.save_pretrained(args.out)
 if dv: print('best dev f1/th',bestf,bestth,flush=True)
 return bestth if dv else args.threshold

def predict(test_file,out_dir,out_file,threshold,batch_size=8,max_length=512,stride=96):
 docs=read_docs(test_file); tok=AutoTokenizer.from_pretrained(out_dir,use_fast=True)
 model=AutoModelForTokenClassification.from_pretrained(out_dir); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
 ds=encode_dataset(docs,tok,max_length,stride,False); print('test chunks',len(ds),flush=True)
 found=get_spans_for_batch(model,ds,tok,device,batch_size)
 os.makedirs(os.path.dirname(out_file),exist_ok=True)
 with open(out_file,'w') as f:
  for pm,sec,st,en in sorted(k for k,v in found.items() if v>=threshold): f.write(f'{pm}\t{sec}:{st}:{en}\n')
 print('written',sum(v>=threshold for v in found.values()),'threshold',threshold,flush=True)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['dev','final','predict'],required=True)
 p.add_argument('--train',default='/workspace/data/training.abstracts.txt'); p.add_argument('--train_ann',default='/workspace/data/training.annotations.txt')
 p.add_argument('--dev',default='/workspace/data/development.abstracts.txt'); p.add_argument('--dev_ann',default='/workspace/data/development.annotations.txt')
 p.add_argument('--test',default='/workspace/data/test.abstracts.txt'); p.add_argument('--out',default='/tmp/chembert'); p.add_argument('--pred',default='/tmp/bertpred.tsv')
 p.add_argument('--epochs',type=int,default=4); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--eval_batch_size',type=int,default=8)
 p.add_argument('--max_length',type=int,default=512); p.add_argument('--stride',type=int,default=96); p.add_argument('--lr',type=float,default=3e-5); p.add_argument('--b_weight',type=float,default=2.0); p.add_argument('--seed',type=int,default=42); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--resume',default='')
 args=p.parse_args()
 if args.mode=='predict': predict(args.test,args.out,args.pred,args.threshold,args.eval_batch_size,args.max_length,args.stride); return
 train=read_docs(args.train,args.train_ann)
 if args.mode=='dev': dev=read_docs(args.dev,args.dev_ann)
 else: train+=read_docs(args.dev,args.dev_ann); dev=None
 os.makedirs(args.out,exist_ok=True)
 th=train_loop(train,dev,args,args.mode=='final')
 if args.mode=='final': predict(args.test,args.out,args.pred,args.threshold,args.eval_batch_size,args.max_length,args.stride)
if __name__=='__main__': main()
