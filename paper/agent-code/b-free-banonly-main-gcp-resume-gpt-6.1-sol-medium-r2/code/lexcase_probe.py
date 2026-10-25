import numpy as np,json
from collections import defaultdict,Counter
from evaluate import load_corpus,decode
from ensemble import load_records,merge
from calibrate import prepare,oof_fit,adjust
from lexicon import metric

docs,gold,offs=load_corpus('development')
files=['/workspace/results/validation/average/development_scores.npz','/workspace/results/validation2/dev_epoch6.npz']
scores,offs=merge([load_records(f,offs) for f in files],docs,[1,1]);_,_,rawmargins=prepare(scores,docs,offs,gold);cats,deltas,allweights=oof_fit(scores,docs,offs,gold);scores=adjust(scores,cats,deltas)
_,labels,margins=prepare(scores,docs,offs,gold)
words=[]
for (_,_,text),offsets in zip(docs,offs):
 row=[]
 for a,b in offsets:
  t=text[a:b]
  row.append(t if 1<=len(t)<=16 and t.isascii() and t.isalnum() and not(a and text[a-1].isalnum()) and not(b<len(text) and text[b].isalnum()) else '')
 words.append(row)
ids=sorted({p for p,s,t in docs});rng=np.random.default_rng(2026);rng.shuffle(ids);fold={p:i%5 for i,p in enumerate(ids)};corrections=[np.zeros(len(v),np.float32) for v in scores]
for f in range(5):
 shapeweights=next(deltas[i] for i,(p,s,t) in enumerate(docs) if fold[p]==f)
 groups=defaultdict(list);variants=defaultdict(set);pos=Counter();neg=Counter()
 for i,(p,s,t) in enumerate(docs):
  if fold[p]==f:continue
  foldmargins=rawmargins[i]+shapeweights[cats[i]]
  for w,m,y in zip(words[i],foldmargins,labels[i]):
   if not w:continue
   groups[w].append((m,y));variants[w.lower()].add(w);pos[w.lower()]+=int(y);neg[w.lower()]+=1-int(y)
 weights={}
 for w,rows in groups.items():
  if len(rows)<10 or len(variants[w.lower()])<2 or pos[w.lower()]<5 or neg[w.lower()]<5:continue
  xx=np.array([m for m,y in rows]);yy=np.array([y for m,y in rows]);delta=0
  for _ in range(20):
   prob=1/(1+np.exp(-np.clip(xx+delta,-25,25)));g=np.sum(prob-yy)+20*delta;h=np.sum(prob*(1-prob))+20;nxt=np.clip(delta-g/h,-1.5,1.5)
   if abs(nxt-delta)<1e-5:break
   delta=nxt
  weights[w]=float(delta)
 for i,(p,s,t) in enumerate(docs):
  if fold[p]==f:corrections[i]=np.array([weights.get(w,0) for w in words[i]],np.float32)
 print('fold',f,'word parameters',len(weights),flush=True)
for strength in [0,.5,1]:
 ss=[]
 for v,d in zip(scores,corrections):
  w=v.copy();w[:,1:]+=strength*d[:,None];ss.append(w)
 for bias in [-.3,-.15,0,.15]:
  pred=decode(ss,docs,offs,bias);print(strength,bias,metric(pred,gold),flush=True)
