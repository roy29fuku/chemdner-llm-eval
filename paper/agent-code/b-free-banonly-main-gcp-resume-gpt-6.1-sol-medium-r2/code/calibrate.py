import json,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from evaluate import load_corpus,decode,write,scores_from_files
from ensemble import load_records,merge
from lexicon import metric

def category(text,a,b):
 term=text[a:b]
 if (a and text[a-1].isalnum()) or (b<len(text) and text[b].isalnum()):return 0
 if not term.isascii() or not term.isalpha() or len(term)>3:return 0
 n=len(term)
 if n==1:return 1 if term.islower() else 2
 if term.islower():return 3 if n==2 else 6
 if term.isupper():return 4 if n==2 else 7
 if term.istitle():return 5 if n==2 else 8
 return 9

def prepare(scores,docs,offs,gold):
 bysec=defaultdict(list)
 for p,s,a,b in gold:bysec[p,s].append((a,b))
 cats=[];labels=[];margins=[]
 for v,(p,s,text),offsets in zip(scores,docs,offs):
  c=np.array([category(text,a,b) for a,b in offsets],np.int8);y=np.zeros(len(v),np.int8);spans=sorted(bysec[p,s]);j=0
  for k,(a,b) in enumerate(offsets):
   while j<len(spans) and spans[j][1]<=a:j+=1
   if j<len(spans) and a<spans[j][1] and b>spans[j][0]:y[k]=1
  cats.append(c);labels.append(y);margins.append(np.logaddexp(v[:,1],v[:,2])-v[:,0])
 return cats,labels,margins

def fit(cats,labels,margins,indices,regularization=50):
 c=np.concatenate([cats[i] for i in indices]);y=np.concatenate([labels[i] for i in indices]);m=np.concatenate([margins[i] for i in indices]);deltas=np.zeros(10,np.float32)
 for k in range(1,10):
  sel=c==k;xx=m[sel];yy=y[sel]
  if len(xx)<100:continue
  delta=0.
  for _ in range(25):
   z=np.clip(xx+delta,-25,25);prob=1/(1+np.exp(-z));grad=np.sum(prob-yy,dtype=np.float64)+regularization*delta;hess=np.sum(prob*(1-prob),dtype=np.float64)+regularization
   nxt=np.clip(delta-grad/hess,-2,2)
   if abs(nxt-delta)<1e-5:break
   delta=nxt
  deltas[k]=delta
 return deltas

def adjust(scores,cats,deltas,strength=1):
 out=[]
 for i,(v,c) in enumerate(zip(scores,cats)):
  w=v.copy();delta=(deltas[i] if isinstance(deltas,list) else deltas)[c]*strength;w[:,1:]+=delta[:,None];out.append(w)
 return out

def oof_fit(scores,docs,offs,gold):
 cats,labels,margins=prepare(scores,docs,offs,gold)
 ids=sorted({p for p,s,t in docs});rng=np.random.default_rng(2026);rng.shuffle(ids);fold={p:j%5 for j,p in enumerate(ids)};deltas=[None]*len(docs)
 for f in range(5):
  train=[i for i,(p,s,t) in enumerate(docs) if fold[p]!=f];weights=fit(cats,labels,margins,train)
  for i,(p,s,t) in enumerate(docs):
   if fold[p]==f:deltas[i]=weights
 allweights=fit(cats,labels,margins,list(range(len(docs))))
 return cats,deltas,allweights

def main():
 ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);args=ap.parse_args();docs,gold,offs=load_corpus(args.split);records=[load_records(f,offs) for f in args.files];scores,offs=merge(records,docs,[1]*len(records));cats,labels,margins=prepare(scores,docs,offs,gold)
 # Article-level out-of-fold corrections: each article's labels are unseen by its calibrator.
 ids=sorted({p for p,s,t in docs});rng=np.random.default_rng(2026);rng.shuffle(ids);fold={p:j%5 for j,p in enumerate(ids)};deltas=[None]*len(docs)
 for f in range(5):
  train=[i for i,(p,s,t) in enumerate(docs) if fold[p]!=f];weights=fit(cats,labels,margins,train);print('fold',f,weights.tolist(),flush=True)
  for i,(p,s,t) in enumerate(docs):
   if fold[p]==f:deltas[i]=weights
 results=[]
 for strength in [0,.5,1]:
  corrected=adjust(scores,cats,deltas,strength)
  for bias in [-.3,-.15,0,.15,.3]:
   pred=decode(corrected,docs,offs,bias);m=metric(pred,gold);row={'strength':strength,'bias':bias,'metrics':m};results.append(row);print(row,flush=True)
 allweights=fit(cats,labels,margins,list(range(len(docs))));best=max(results,key=lambda x:x['metrics'][0]);Path(args.out).write_text(json.dumps({'best':best,'weights':allweights.tolist(),'results':results},indent=2))
 print('BEST',best,'weights',allweights.tolist(),flush=True)

if __name__=='__main__':main()
