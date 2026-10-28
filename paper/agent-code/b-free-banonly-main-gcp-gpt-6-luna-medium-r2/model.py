import re,sys,math,collections,ahocorasick
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

BASE=Path('/workspace/data')

def load(name, ann=None):
 d={}
 with (BASE/name).open(encoding='utf8') as f:
  for line in f:
   p=line.rstrip('\n').split('\t',2); d[p[0]]={'T':p[1], 'A':p[2], 'anns':[]}
 if ann:
  with (BASE/ann).open(encoding='utf8') as f:
   for line in f:
    p=line.rstrip('\n').split('\t'); d[p[0]]['anns'].append((p[1],int(p[2]),int(p[3]),p[4],p[5]))
 return d

TOKEN_RE=re.compile(r"[\w]+|[^\w\s]",re.UNICODE)

def make_automaton(docs, minfreq=1):
 freq=collections.Counter(); docfreq=collections.Counter()
 for d in docs.values():
  seen=set()
  for sec,a,b,s,c in d['anns']:
   v=s.lower()
   freq[v]+=1; seen.add(v)
  docfreq.update(seen)
 A=ahocorasick.Automaton()
 for w,c in freq.items():
  if w: A.add_word(w,(w,c,docfreq[w]))
 A.make_automaton()
 return A,freq,docfreq

def get_cands(docs,A):
 allc=[]
 for pm,d in docs.items():
  gold={(sec,a,b) for sec,a,b,s,c in d['anns']}
  for sec in 'TA':
   text=d[sec]; low=text.lower()
   toks=list(TOKEN_RE.finditer(text))
   starts=[m.start() for m in toks]; ends=[m.end() for m in toks]
   # each match gets contextual tokens via bisect
   import bisect
   for e,(surface,freq,df) in A.iter(low):
    st=e-len(surface)+1; en=e+1
    if st>0 and (low[st-1].isalnum() or low[st-1]=='_'): continue
    if en<len(low) and (low[en].isalnum() or low[en]=='_'): continue
    # exact duplicate annotations are impossible; normalize candidate uniqueness later
    i=bisect.bisect_right(ends,st)
    j=bisect.bisect_left(starts,en)
    # find immediately preceding/following tokens
    prev=toks[i-1].group().lower() if i else '<BOS>'
    prev2=toks[i-2].group().lower() if i>1 else '<BOS2>'
    nxt=toks[j].group().lower() if j<len(toks) else '<EOS>'
    nxt2=toks[j+1].group().lower() if j+1<len(toks) else '<EOS2>'
    f=features(text[st:en],prev,prev2,nxt,nxt2,sec,freq,df,st,en)
    allc.append((pm,sec,st,en,f,(sec,st,en) in gold))
 return allc

def features(s,prev,prev2,nxt,nxt2,sec,freq,df,st,en):
 lo=s.lower(); f={}
 f['surface='+lo]=1
 f['prev='+prev]=1; f['prev2='+prev2]=1; f['next='+nxt]=1; f['next2='+nxt2]=1
 f['pbig='+prev+' '+lo[:40]]=1; f['nbig='+lo[-40:]+' '+nxt]=1
 f['context='+prev+'|'+nxt]=1
 f['section='+sec]=1
 f['logfreq']=math.log1p(freq); f['logdf']=math.log1p(df)
 f['len']=len(s); f['ntokens']=len(TOKEN_RE.findall(s)); f['isdigit']=int(any(ch.isdigit() for ch in s)); f['has_upper']=int(any(ch.isupper() for ch in s)); f['all_upper']=int(s.isupper()); f['has_hyphen']=int('-' in s); f['has_slash']=int('/' in s); f['has_digit']=int(any(ch.isdigit() for ch in s)); f['has_paren']=int(any(ch in '()[]' for ch in s))
 # char class/affix
 f['first='+lo[0:1]]=1; f['last='+lo[-1:]]=1
 for n in [2,3,4]:
  if len(lo)>=n:
   f[f'prefix{n}='+lo[:n]]=1; f[f'suffix{n}='+lo[-n:]]=1
 # word features and number of constituent tokens
 words=[x.lower() for x in TOKEN_RE.findall(s)]
 for w in words:
  f['part='+w]=1
 # broad suffix/patterns
 if lo.endswith('s'): f['plural']=1
 if re.search(r'\d',lo): f['digitpattern']=re.sub(r'\d','0',lo)
 if re.search(r'\b(?:mg|ml|mmol|μm|um|nm|mM|nM)\b',s,re.I): f['units-nearby']=1
 return f

def evaluate(pred,gold):
 tp=len(pred&gold); p=tp/len(pred) if pred else 0; r=tp/len(gold)
 return p,r,2*p*r/(p+r) if p+r else 0

def run():
 tr=load('training.abstracts.txt','training.annotations.txt'); dv=load('development.abstracts.txt','development.annotations.txt')
 A,freq,df=make_automaton(tr)
 print('automaton entries',len(freq),flush=True)
 trc=get_cands(tr,A); dvc=get_cands(dv,A)
 print('candidates',len(trc),len(dvc),'train positives',sum(x[-1] for x in trc),'dev positives',sum(x[-1] for x in dvc),flush=True)
 vec=DictVectorizer(sparse=True)
 X=vec.fit_transform([x[4] for x in trc]); X.indices=X.indices.astype('int32'); X.indptr=X.indptr.astype('int32'); y=[int(x[-1]) for x in trc]
 print('fit',X.shape,flush=True)
 clf=LogisticRegression(max_iter=200,C=0.5,class_weight=None,solver='liblinear',verbose=0)
 clf.fit(X,y)
 Xv=vec.transform([x[4] for x in dvc]); Xv.indices=Xv.indices.astype('int32'); Xv.indptr=Xv.indptr.astype('int32'); probs=clf.predict_proba(Xv)[:,1]
 gold={(pm,a[0],a[1],a[2]) for pm,d in dv.items() for a in d['anns']}
 for t in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8]:
  pred={(x[0],x[1],x[2],x[3]) for x,p in zip(dvc,probs) if p>=t}
  print(t,len(pred),evaluate(pred,gold))
 # best threshold exact
 best=(0,None)
 for t in [i/100 for i in range(1,100)]:
  pred={(x[0],x[1],x[2],x[3]) for x,p in zip(dvc,probs) if p>=t}
  m=evaluate(pred,gold)[2]
  if m>best[0]: best=(m,t)
 print('best',best)

if __name__=='__main__':run()
