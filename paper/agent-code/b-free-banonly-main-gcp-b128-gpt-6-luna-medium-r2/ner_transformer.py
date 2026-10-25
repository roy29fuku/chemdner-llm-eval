import os, re, collections, argparse, json, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
BASE='/home/agent/'
MODEL='allenai/scibert_scivocab_uncased'
LABELS=['O','B','I']; L2I={x:i for i,x in enumerate(LABELS)}
MAXLEN=384; STRIDE=96

def read_split(name):
 docs=[]
 with open(BASE+name+'.abstracts.txt',encoding='utf8') as f:
  for line in f:
   p,t,a=line.rstrip('\n').split('\t',2);docs.append((p,{'T':t,'A':a}))
 anns=collections.defaultdict(list)
 with open(BASE+name+'.annotations.txt',encoding='utf8') as f:
  for line in f:
   p,s,b,e,txt,c=line.rstrip('\n').split('\t');anns[p,s].append((int(b),int(e),txt))
 return docs,anns

def make_items(split,tokenizer):
 docs,anns=read_split(split); items=[]
 for pmid,sections in docs:
  for sec,text in sections.items():
   gold=anns.get((pmid,sec),[])
   enc=tokenizer(text,return_offsets_mapping=True,truncation=True,max_length=MAXLEN,stride=STRIDE,
                 return_overflowing_tokens=True,padding=False)
   for win in range(len(enc['input_ids'])):
    offsets=enc['offset_mapping'][win]
    lbl=[]; prev_mid=None
    for a,b in offsets:
     if a==b: lbl.append(-100);prev_mid=None;continue
     mid=None
     for j,(x,y,_) in enumerate(gold):
      if a<y and b>x:
       mid=j;break
     if mid is None: lbl.append(L2I['O']);prev_mid=None
     else:
      lbl.append(L2I['I'] if prev_mid==mid else L2I['B']);prev_mid=mid
    items.append({'input_ids':enc['input_ids'][win], 'attention_mask':enc['attention_mask'][win], 'labels':lbl,
                  'offsets':offsets,'pmid':pmid,'section':sec,'text':text})
 return items

class Collate:
 def __init__(self,tokenizer):self.pad=tokenizer.pad_token_id
 def __call__(self,batch):
  ml=max(len(x['input_ids']) for x in batch)
  out={k:[] for k in ('input_ids','attention_mask','labels')}
  for x in batch:
   n=ml-len(x['input_ids']);out['input_ids'].append(x['input_ids']+[self.pad]*n)
   out['attention_mask'].append(x['attention_mask']+[0]*n);out['labels'].append(x['labels']+[-100]*n)
  return {k:torch.tensor(v,dtype=torch.long) for k,v in out.items()}

def infer_items(model,items,tokenizer,batch_size=16):
 # Average logits for duplicated subword offsets across overlapping windows.
 model.eval(); collate=Collate(tokenizer); logits_by=collections.defaultdict(lambda:[np.zeros(3,dtype=np.float64),0])
 loader=DataLoader(items,batch_size=batch_size,shuffle=False,collate_fn=collate)
 with torch.no_grad():
  for idx,b in enumerate(loader):
   ids=b['input_ids'].to('cuda');mask=b['attention_mask'].to('cuda')
   logits=model(input_ids=ids,attention_mask=mask).logits.float().cpu().numpy()
   for j in range(len(ids)):
    item=items[idx*batch_size+j]
    for pos,(a,z) in enumerate(item['offsets']):
     if z<=a:continue
     q=logits[j,pos];v=logits_by[item['pmid'],item['section'],a,z][0];v+=q;logits_by[item['pmid'],item['section'],a,z][1]+=1
 return logits_by

def spans_from_logits(text,entries,bias=(0,0,0)):
 # entries are (start,end,logits), sorted. Smooth I after O into a new mention.
 tags=[]
 for a,b,sc in entries:
  ss=sc+np.asarray(bias);tag=int(np.argmax(ss))
  if tag==2 and (not tags or tags[-1][2]==0):tag=1
  tags.append((a,b,tag))
 out=[];st=en=None;prev=None
 for a,b,t in tags:
  if t==1:
   if st is not None:out.append((st,en))
   st,en=a,b
  elif t==2:
   if st is None:st=a
   en=b
  else:
   if st is not None:out.append((st,en));st=en=None
 if st is not None:out.append((st,en))
 return out

def dictionary(splitnames):
 out=collections.Counter()
 for s in splitnames:
  with open(BASE+s+'.annotations.txt',encoding='utf8') as f:
   for line in f:
    p,sec,b,e,t,c=line.rstrip('\n').split('\t');out[t]+=1
 return out

ELEMENTS={'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'}
_DICT_PATTERNS={}
def dict_matches(text,dic):
 key=tuple(sorted(dic))
 pats=_DICT_PATTERNS.get(key)
 if pats is None:
  long_terms=[t for t in key if len(t.strip())>3]
  short_terms=[t for t in key if len(t.strip())>=2 and (t.isupper() or any(c.isdigit() for c in t) or t in ELEMENTS)]
  pats=[]
  if long_terms:pats.append(re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(t) for t in sorted(long_terms,key=len,reverse=True))+r')(?!\w)',re.I))
  if short_terms:pats.append(re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(t) for t in sorted(short_terms,key=len,reverse=True))+r')(?!\w)'))
  _DICT_PATTERNS[key]=pats
 return [(m.start(),m.end()) for pat in pats for m in pat.finditer(text)]

def combine_spans(text,entries,dic,bias=(0,1,2)):
 learned=set(spans_from_logits(text,entries,bias));known=set(dict_matches(text,dic))
 learned={x for x in learned if not any(x[0]<y[1] and y[0]<x[1] for y in known)}
 return learned|known

def eval_preds(items,logits_by,dic=None,bias=(0,0,0),verbose=False):
 docs,anns=read_split('development');by=collections.defaultdict(list)
 seen=set()
 for item in items:
  key=(item['pmid'],item['section'])
  if key not in seen:seen.add(key);by[key]=[]
 # logits dictionary provides unique subword offsets
 for (p,s,a,b), (sc,n) in logits_by.items():by[p,s].append((a,b,sc/n))
 tp=fp=fn=0;details=[]
 for p,secs in docs:
  for s,text in secs.items():
   gs={(x,y) for x,y,_ in anns.get((p,s),[])}
   ps=set(spans_from_logits(text,sorted(by[p,s]),bias))
   if dic:
    known=set(dict_matches(text,dic));ps={x for x in ps if not any(x[0]<y[1] and y[0]<x[1] for y in known)}|known
   tp+=len(gs&ps);fp+=len(ps-gs);fn+=len(gs-ps)
   if verbose:details.extend([(text[a:b],'FP') for a,b in ps-gs]+[(text[a:b],'FN') for a,b in gs-ps])
 f=2*tp/(2*tp+fp+fn or 1);print('dev',tp,fp,fn,'P/R/F1',round(tp/(tp+fp or 1),4),round(tp/(tp+fn or 1),4),round(f,4),flush=True)
 if verbose:print(details[:150])
 return f

def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',default='eval');ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--train',nargs='+',default=['training']);ap.add_argument('--save',default='/home/agent/scibert-chem');ap.add_argument('--checkpoint',default=None)
 args=ap.parse_args();seed();torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
 if args.checkpoint:
  model=AutoModelForTokenClassification.from_pretrained(args.checkpoint).cuda()
 else:
  model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3,id2label={i:x for i,x in enumerate(LABELS)},label2id=L2I).cuda()
 if args.mode in ('eval','train'):
  train=[]
  for s in args.train:train += make_items(s,tok)
  dev=make_items('development',tok) if args.mode=='eval' else []
  collate=Collate(tok);loader=DataLoader(train,batch_size=8,shuffle=True,collate_fn=collate)
  optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
  steps=len(loader)*args.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*0.08),steps)
  best=-1
  for ep in range(args.epochs):
   model.train();losses=[]
   for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    ids=batch['input_ids'].cuda();mask=batch['attention_mask'].cuda();labels=batch['labels'].cuda()
    with torch.autocast(device_type='cuda',dtype=torch.float16):loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();losses.append(float(loss.detach().cpu()))
   print('epoch',ep+1,'loss',round(sum(losses)/len(losses),4),flush=True)
   if args.mode=='eval':
    lb=infer_items(model,dev,tok);f=eval_preds(dev,lb,dictionary(args.train))
    if f>best:best=f;model.save_pretrained(args.save);tok.save_pretrained(args.save)
   else:
    model.save_pretrained(args.save);tok.save_pretrained(args.save)
 if args.mode=='predict':
  test_docs=[]
  with open(BASE+'test.abstracts.txt',encoding='utf8') as f:
   for line in f:
    p,t,a=line.rstrip('\n').split('\t',2);test_docs.extend([(p,'T',t),(p,'A',a)])
  items=[]
  # Directly tokenize without annotations.
  for p,s,text in test_docs:
   enc=tok(text,return_offsets_mapping=True,truncation=True,max_length=MAXLEN,stride=STRIDE,return_overflowing_tokens=True,padding=False)
   for wi in range(len(enc['input_ids'])):
    items.append({'input_ids':enc['input_ids'][wi],'attention_mask':enc['attention_mask'][wi], 'labels':[-100]*len(enc['input_ids'][wi]),'offsets':enc['offset_mapping'][wi],'pmid':p,'section':s,'text':text})
  lb=infer_items(model,items,tok,batch_size=24);dic=dictionary(['training','development'])
  docs=[];grouped=collections.defaultdict(list)
  for (pp,ss,a,b),(sc,n) in lb.items():grouped[pp,ss].append((a,b,sc/n))
  for p,s,text in test_docs:
   ps=combine_spans(text,sorted(grouped[p,s]),dic,(0,1,2))
   docs.extend((p,s,a,b) for a,b in ps)
  with open('/home/agent/predictions.tsv','w',encoding='utf8') as f:
   for p,s,a,b in sorted(set(docs)):f.write(f'{p}\t{s}:{a}:{b}\n')
  print('wrote',len(set(docs)),'mentions',flush=True)

if __name__=='__main__':main()
