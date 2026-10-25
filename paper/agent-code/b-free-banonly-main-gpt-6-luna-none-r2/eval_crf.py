import re,collections,pickle,sys
from dict_tag import load,build
from sklearn_crfsuite import CRF
TOK=re.compile(r"[A-Za-z]+(?:[0-9]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",re.UNICODE)
def toks(text):return [(m.group(),m.start(),m.end()) for m in TOK.finditer(text)]
def shape(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 return re.sub(r'(A+)','A',re.sub(r'(a+)','a',re.sub(r'(0+)','0',s)))[:12]
def seq_features(text,forms):
 ts=toks(text);out=[]
 for i,(w,b,e) in enumerate(ts):
  wl=w.lower();f={'bias':1.0,'w':wl,'shape':shape(w),'isupper':w.isupper(),'istitle':w.istitle(),'isdigit':w.isdigit(),'hasdigit':any(c.isdigit() for c in w),'prefix1':wl[:1],'prefix2':wl[:2],'prefix3':wl[:3],'suffix2':wl[-2:],'suffix3':wl[-3:],'len':min(len(w),20),'dict_'+wl:forms.get(wl,0)>0,'BOS':i==0,'EOS':i==len(ts)-1}
  for j in [-2,-1,1,2]:
   if 0<=i+j<len(ts):z=ts[i+j][0];f[f'{j}:w']=z.lower();f[f'{j}:shape']=shape(z)
  out.append(f)
 return ts,out

def spans_probs(text,forms,model):
 ts,ff=seq_features(text,forms);probs=[]
 for st in range(0,len(ts),180):probs+=model.predict_marginals_single(ff[st:st+180])
 # use max P(B/I) to form entities then confidence average
 out=[];start=last=None;cs=[]
 for i,((w,b,e),d) in enumerate(zip(ts,probs)):
  lab=max(d,key=d.get)
  if lab=='B':
   if start is not None:out.append((start,last,sum(cs)/len(cs)))
   start=b;last=e;cs=[d['B']]
  elif lab=='I':
   if start is None:start=b;cs=[]
   last=e;cs.append(d['I'])
  else:
   if start is not None:out.append((start,last,sum(cs)/len(cs)));start=None;cs=[]
 if start is not None:out.append((start,last,sum(cs)/len(cs)))
 return out

tr=load('training');dev=load('development');forms=build(tr);m=pickle.load(open('/workspace/model/crf.pkl','rb'))
allpred=[];tp=fp=fn=0
for (p,s),aa in dev[1].items():
 pred=spans_probs(dev[0][p][s],forms,m);gold={(a,b) for a,b,_ in aa}; allpred.append((pred,gold))
 tp+=len({(a,b) for a,b,c in pred}&gold);fp+=len(pred)-len({(a,b) for a,b,c in pred}&gold);fn+=len(gold-{(a,b) for a,b,c in pred})
print('base',tp,fp,fn,2*tp/(2*tp+fp+fn),flush=True)
for th in [0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
 T=P=F=0
 for pred,gold in allpred:
  pr={(a,b) for a,b,c in pred if c>=th};T+=len(pr&gold);P+=len(pr);F+=len(gold-pr)
 print('th',th,'p/r/f',P,T,F,2*T/(P+T+F),flush=True)
