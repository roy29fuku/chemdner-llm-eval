import re,math,collections,sys,os,functools,bisect
import pycrfsuite
from model import load,TOKEN_RE,evaluate
from pathlib import Path
# punctuation separated; words alphabetic or numeric runs, so Arg176Met becomes three tokens
TOK=re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]",re.UNICODE)

@functools.lru_cache(maxsize=50000)
def shape(w):
 s=''.join('d' if c.isdigit() else 'X' if c.isupper() else 'x' if c.islower() else c for c in w)
 out=[]
 for c in s:
  if not out or out[-1]!=c: out.append(c)
 return ''.join(out)[:12]
def token_feats(tok, i, ts, sec):
 w=tok.group(); lo=w.lower(); fs=['bias','w='+lo,'sh='+shape(w), 'sec='+sec]
 if len(w)>=2: fs+=['pre2='+lo[:2],'suf2='+lo[-2:]]
 if len(w)>=3: fs+=['pre3='+lo[:3],'suf3='+lo[-3:]]
 if len(w)>=4: fs+=['pre4='+lo[:4],'suf4='+lo[-4:]]
 if w.isupper(): fs+=['upper']
 if w.istitle(): fs+=['title']
 if any(c.isdigit() for c in w): fs+=['hasdigit']
 if w.isdigit(): fs+=['numeric']
 if any(ord(c)>127 for c in w): fs+=['unicode']
 if w in ['-','/','+',',','.','(',')','[',']',':',';','%']: fs+=['punct='+w]
 for off in (-2,-1,1,2):
  j=i+off
  if 0<=j<len(ts):
   x=ts[j].group().lower()
   fs+=['w%d=%s'%(off,x)]
   if off in (-1,1): fs+=['sh%d=%s'%(off,shape(ts[j].group()))]
  else: fs+=['BOS' if off<0 else 'EOS']
 # include neighboring bigrams
 if i>0: fs+=['bi-1='+ts[i-1].group().lower()+'_'+lo]
 if i+1<len(ts): fs+=['bi+1='+lo+'_'+ts[i+1].group().lower()]
 return fs

def prepare(docs):
 X=[]; Y=[]; META=[]
 for pm,d in docs.items():
  for sec in 'TA':
   text=d[sec]; ts=list(TOK.finditer(text)); gold=[]
   for s,a,b,txt,cls in d['anns']:
    if s==sec: gold.append((a,b))
   labels=['O']*len(ts); starts=[t.start() for t in ts]; ends=[t.end() for t in ts]
   for a,b in gold:
    lo=bisect.bisect_left(starts,a); hi=bisect.bisect_right(ends,b)
    inds=[i for i in range(lo,hi) if starts[i]>=a and ends[i]<=b]
    if not inds:
     lo=bisect.bisect_left(ends,a+1); hi=bisect.bisect_left(starts,b)
     inds=list(range(lo,hi))
    for k,i in enumerate(inds): labels[i]='B' if k==0 else 'I'
   feats=[token_feats(t,i,ts,sec) for i,t in enumerate(ts)]
   X.append(feats); Y.append(labels); META.append((pm,sec,[(t.start(),t.end()) for t in ts],text))
 return X,Y,META

def spans_from_tagger(tagger, feats, toks):
 labels=tagger.tag(feats); out=[]; i=0
 while i<len(labels):
  if labels[i]!='B': i+=1; continue
  j=i+1
  while j<len(labels) and labels[j]=='I': j+=1
  start=toks[i].start(); end=toks[j-1].end()
  # span confidence = min of all token label marginals
  marg=[]
  for k in range(i,j):
   lab='B' if k==i else 'I'
   marg.append(tagger.marginal(lab,k))
  conf=min(marg) if marg else 0
  out.append((start,end,conf,labels[i:j]))
  i=j
 return out

def run():
 tr=load('training.abstracts.txt','training.annotations.txt'); dv=load('development.abstracts.txt','development.annotations.txt')
 X,Y,meta=prepare(tr); print('prepared train',len(X),sum(map(len,X)),flush=True)
 trainer=pycrfsuite.Trainer(verbose=False)
 for x,y in zip(X,Y): trainer.append(x,y)
 trainer.set_params({'c1':0.05,'c2':0.1,'max_iterations':100,'feature.possible_transitions':True,'feature.possible_states':True})
 trainer.train('/tmp/chem.crfsuite'); print('trained',flush=True)
 tagger=pycrfsuite.Tagger(); tagger.open('/tmp/chem.crfsuite')
 Xv,Yv,metav=prepare(dv); print('prepared dev',len(Xv),sum(map(len,Xv)),flush=True)
 gold={(pm,a[0],a[1],a[2]) for pm,d in dv.items() for a in d['anns']}
 allp=[]
 for (pm,sec,offs,text),feats in zip(metav,Xv):
  ts=list(TOK.finditer(text))
  for a,b,c,_ in spans_from_tagger(tagger,feats,ts): allp.append((pm,sec,a,b,c))
 for th in [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8]:
  pred={(pm,s,a,b) for pm,s,a,b,p in allp if p>=th}
  print(th,len(pred),evaluate(pred,gold),flush=True)
 # exact full best threshold
 best=(0,None)
 for i in range(0,100):
  th=i/100; pred={(pm,s,a,b) for pm,s,a,b,p in allp if p>=th}; f=evaluate(pred,gold)[2]
  if f>best[0]: best=f,th
 print('best',best)
 # baseline exact decoded spans regardless confidence
if __name__=='__main__': run()
