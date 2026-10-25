import json,pickle,re
import numpy as np
from lex_eval import load,make_automaton
from lex_clf import surfaces,augment_aliases,candidate_docs,build_features,WORD_RE,featurize

td,tg=load('training'); vd,vg=load('development')
stats=augment_aliases(surfaces(td,tg)); pats=list(stats); go,fail,out=make_automaton(pats)
vc=candidate_docs(vd,pats,go,fail,out); vx,vm=build_features(vd,vc,stats)
vec,clf,_,th=pickle.load(open('/workspace/lexclf_alias.pkl','rb'))
base_scores=clf.predict_proba(vec.transform(vx))[:,1]
base={x for x,s in zip(vm,base_scores) if s>=th}; gold=set(vg)
bert={tuple(x) for x in json.load(open('/workspace/devpred_best.json'))}
known=set(vc)
extra=[x for x in bert if x not in known]
feature_cache={}
for p,sec,a,b in extra:
    text=vd[p][sec]; toks=[(m.start(),m.end(),m.group().lower()) for m in WORD_RE.finditer(text)]
    starts=[x[0] for x in toks]; ends=[x[1] for x in toks]; q=text[a:b].lower()
    feature_cache[(p,sec,a,b)]=featurize(q,text[a:b],a,b,starts,ends,toks,(0,0))
scores=clf.predict_proba(vec.transform([feature_cache[x] for x in extra]))[:,1] if extra else []
print('base',len(base),len(base&gold),2*len(base&gold)/(len(base)+len(gold)),'bert novel',len(extra),flush=True)
for t in [0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.8,.9,1.0]:
    pred=base|{x for x,s in zip(extra,scores) if s>=t}
    tp=len(pred&gold); print(t,len(pred),tp,2*tp/(len(pred)+len(gold)),tp/len(pred),tp/len(gold))
