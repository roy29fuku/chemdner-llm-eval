import pickle, numpy as np
from sklearn.metrics import precision_recall_curve
from lex_eval import load,make_automaton
from lex_clf import surfaces,augment_aliases,candidate_docs,build_features

td,tg=load('training'); vd,vg=load('development')
stats=augment_aliases(surfaces(td,tg)); pats=list(stats); go,fail,out=make_automaton(pats)
vc=candidate_docs(vd,pats,go,fail,out); vx,vm=build_features(vd,vc,stats)
vec,clf,_,global_th=pickle.load(open('/workspace/lexclf_alias.pkl','rb'))
scores=clf.predict_proba(vec.transform(vx))[:,1]; gold=set(vg); y=np.asarray([int(x in gold) for x in vm])
freq=np.asarray([stats[vd[p][s][a:b].lower()][0] for p,s,a,b in vm])
def bucket(n):
  if n<=1:return 0
  if n<=2:return 1
  if n<=4:return 2
  if n<=9:return 3
  return 4
groups=np.asarray([bucket(n) for n in freq]); cuts=[global_th]*5
def evaluate(ths):
  pred=np.asarray([scores[i]>=ths[groups[i]] for i in range(len(scores))])
  tp=int(np.sum(y*pred)); n=int(np.sum(pred)); return 2*tp/(n+len(gold)),tp,n
print('global',evaluate([global_th]*5),global_th)
for rep in range(3):
 for g in range(5):
  idx=np.where(groups==g)[0]
  cand=np.quantile(scores[idx],np.linspace(.05,.95,41)) if len(idx) else np.array([global_th])
  best=(-1,None)
  for th in cand:
   t=cuts.copy(); t[g]=float(th); val=evaluate(t)[0]
   if val>best[0]: best=(val,float(th))
  cuts[g]=best[1]
  print('round',rep,'group',g,'n',len(idx),'th',cuts[g],'f1',evaluate(cuts),flush=True)
print('thresholds',cuts)
