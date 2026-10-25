import collections, json, pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from lex_eval import load,make_automaton
from lex_clf import surfaces,augment_aliases,candidate_docs,build_features

trdocs,tgold=load('training'); dvdocs,dgold=load('development')
stats=augment_aliases(surfaces(trdocs,tgold)); pats=list(stats); go,fail,out=make_automaton(pats)
tc=candidate_docs(trdocs,pats,go,fail,out); vc=candidate_docs(dvdocs,pats,go,fail,out)
print('terms/candidates',len(pats),len(tc),len(vc),flush=True)
tx,tm=build_features(trdocs,tc,stats); vx,vm=build_features(dvdocs,vc,stats)
tg=set(tgold); vg=set(dgold); y=np.asarray([int(x in tg) for x in tm]); vy=np.asarray([int(x in vg) for x in vm])
vec=DictVectorizer(dtype=np.float32); X=vec.fit_transform(tx); V=vec.transform(vx)
best=None
for alpha in [1e-6,3e-6,1e-5,3e-5,1e-4]:
 clf=SGDClassifier(loss='log_loss',alpha=alpha,max_iter=40,tol=1e-4,random_state=42)
 clf.fit(X,y); probs=clf.predict_proba(V)[:,1]
 p,r,ts=precision_recall_curve(vy,probs); ra=r[:-1]*vy.sum()/len(vg)
 fs=2*p[:-1]*ra/(p[:-1]+ra+1e-12); i=int(fs.argmax()); th=float(ts[i])
 pred={x for x,z in zip(vm,probs) if z>=th}; tp=len(pred&vg); f=2*tp/(len(pred)+len(vg))
 print('alpha',alpha,'th',th,'f1',f,'n',len(pred),'tp',tp,'p',tp/len(pred),'r',tp/len(vg),flush=True)
 if best is None or f>best[0]: best=(f,alpha,th,clf)
f,alpha,threshold,clf=best
print('BEST',f,alpha,threshold,flush=True)
serial_stats={q:(v[0],len(v[1])) for q,v in stats.items()}
with open('/workspace/lexclf_alias.pkl','wb') as h: pickle.dump((vec,clf,serial_stats,threshold),h)
json.dump(sorted({x for x,z in zip(vm,clf.predict_proba(V)[:,1]) if z>=threshold}),open('/workspace/lexclf_alias_dev.json','w'))
