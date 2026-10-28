import re,collections,pickle,sys,os,math,time
from dict_tag import load,build
from sklearn_crfsuite import CRF
from sklearn.metrics import f1_score

TOK=re.compile(r"[A-Za-z]+(?:[0-9]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",re.UNICODE)
def toks(text): return [(m.group(),m.start(),m.end()) for m in TOK.finditer(text)]
def shape(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 return re.sub(r'(A+)','A',re.sub(r'(a+)','a',re.sub(r'(0+)','0',s)))[:12]

def seq_features(text,forms):
 ts=toks(text); out=[]
 for i,(w,b,e) in enumerate(ts):
  wl=w.lower(); f={'bias':1.0,'w':wl,'shape':shape(w),'isupper':w.isupper(),'istitle':w.istitle(),'isdigit':w.isdigit(),'hasdigit':any(c.isdigit() for c in w),'prefix1':wl[:1],'prefix2':wl[:2],'prefix3':wl[:3],'suffix2':wl[-2:],'suffix3':wl[-3:], 'len':min(len(w),20),'hasGreek':any(ord(c)>127 for c in w),'hasHyphen':'-' in w,'chem_suffix':any(wl.endswith(x) for x in ['ol','one','ate','ide','ine','ium','yl','ose','ase','ene','ane','azole','amine','acid','oxide','oate','thiol','phenyl']),'chem_prefix':any(wl.startswith(x) for x in ['hydroxy','amino','methyl','ethyl','chloro','bromo','fluoro','nitro','oxo','acetyl','benzyl'])}
  for j in [-2,-1,1,2]:
   if 0<=i+j<len(ts):
    z=ts[i+j][0];f[f'{j}:w']=z.lower();f[f'{j}:shape']=shape(z)
  # name-dictionary lexical feature only on single-token mentions
  f['dict_'+wl]=forms.get(wl,0)>0
  if i>0:f['BOS']=False
  else:f['BOS']=True
  if i==len(ts)-1:f['EOS']=True
  out.append(f)
 return ts,out

def labels_for(ts,aa):
 y=[];j=0;aa=sorted(aa)
 for w,b,e in ts:
  while j<len(aa) and aa[j][1]<=b:j+=1
  if j<len(aa) and aa[j][0]<=b and e<=aa[j][1]:
   y.append('B' if b==aa[j][0] else 'I')
  else:y.append('O')
 return y

trtexts,tranns=load('training'); devtexts,devanns=load('development'); forms=build((trtexts,tranns))
X=[];Y=[]
for (p,s),aa in tranns.items():
 text=trtexts[p][s];ts,ff=seq_features(text,forms);yy=labels_for(ts,aa)
 # partition into chunks to prevent long CRF sequence
 for st in range(0,len(ts),180):X.append(ff[st:st+180]);Y.append(yy[st:st+180])
print('training sequences',len(X),sum(map(len,X)),flush=True)
model=CRF(algorithm='lbfgs',c1=0.05,c2=0.1,max_iterations=120,all_possible_transitions=True,verbose=True)
model.fit(X,Y)
with open('/workspace/model/crf.pkl','wb') as f:pickle.dump(model,f)

def spans_for(text,forms,model):
 ts,ff=seq_features(text,forms); labs=[]
 for st in range(0,len(ts),180):labs.extend(model.predict_single(ff[st:st+180]))
 spans=[];start=None;last=None
 for i,((w,b,e),lab) in enumerate(zip(ts,labs)):
  if lab=='B':
   if start is not None:spans.append((start,last))
   start=b;last=e
  elif lab=='I':
   if start is None:start=b
   last=e
  else:
   if start is not None:spans.append((start,last));start=None
 if start is not None:spans.append((start,last))
 return spans

TP=FP=FN=0
for (p,s),aa in devanns.items():
 pr=set(spans_for(devtexts[p][s],forms,model));g={(b,e) for b,e,_ in aa}
 TP+=len(pr&g);FP+=len(pr-g);FN+=len(g-pr)
print('DEV',TP,FP,FN,2*TP/(2*TP+FP+FN),flush=True)
