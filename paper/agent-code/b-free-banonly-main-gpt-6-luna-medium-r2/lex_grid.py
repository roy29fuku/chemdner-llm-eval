import numpy as np, pickle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction import DictVectorizer
from lex_eval import load,make_automaton
from lex_clf import surfaces,candidate_docs,build_features
import collections

trdocs,tgold=load('training'); dvdocs,dgold=load('development')
stats=surfaces(trdocs,tgold); pats=list(stats); go,fail,out=make_automaton(pats)
tc=candidate_docs(trdocs,pats,go,fail,out); vc=candidate_docs(dvdocs,pats,go,fail,out)
tx,tm=build_features(trdocs,tc,stats); vx,vm=build_features(dvdocs,vc,stats)
tg=set(tgold); vg=set(dgold); y=np.asarray([int(x in tg) for x in tm]); vy=np.asarray([int(x in vg) for x in vm])
vec=DictVectorizer(dtype=np.float32); X=vec.fit_transform(tx); V=vec.transform(vx)
best=None
for alpha in [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4]:
  for weight in [None,'balanced']:
    clf=SGDClassifier(loss='log_loss',alpha=alpha,max_iter=50,tol=1e-4,random_state=42,class_weight=weight)
    clf.fit(X,y); probs=clf.predict_proba(V)[:,1]
    p,r,ts=precision_recall_curve(vy,probs)
    rall=r[:-1]*vy.sum()/len(vg); fs=2*p[:-1]*rall/(p[:-1]+rall+1e-12)
    i=int(fs.argmax()); threshold=float(ts[i]); pred={x for x,z in zip(vm,probs) if z>=threshold}; tp=len(pred&vg)
    f1=2*tp/(len(pred)+len(vg)); row=(f1,alpha,weight,threshold,len(pred),tp,tp/len(pred),tp/len(vg),clf.n_iter_)
    print(row,flush=True)
    if best is None or f1>best[0]: best=(f1,alpha,weight,threshold,clf)
f1,alpha,weight,threshold,clf=best
print('BEST',f1,alpha,weight,threshold,flush=True)
serial_stats={q:(v[0],len(v[1])) for q,v in stats.items()}
with open('/workspace/lexclf.pkl','wb') as f: pickle.dump((vec,clf,serial_stats,threshold),f)
