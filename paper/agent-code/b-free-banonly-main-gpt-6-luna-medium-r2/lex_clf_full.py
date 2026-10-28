import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from lex_eval import load,make_automaton
from lex_clf import surfaces,augment_aliases,candidate_docs,build_features

td,ta=load('training'); dd,da=load('development')
docs={**td,**dd}; gold=ta|da
stats=augment_aliases(surfaces(docs,gold)); pats=list(stats); go,fail,out=make_automaton(pats)
candidates=candidate_docs(docs,pats,go,fail,out)
print('terms/candidates',len(pats),len(candidates),flush=True)
features,meta=build_features(docs,candidates,stats)
gold=set(gold); y=np.asarray([int(x in gold) for x in meta])
print('examples',len(y),'positives',int(y.sum()),flush=True)
vec=DictVectorizer(dtype=np.float32); X=vec.fit_transform(features)
clf=SGDClassifier(loss='log_loss',alpha=3e-5,max_iter=45,tol=1e-4,random_state=42)
clf.fit(X,y)
print('fit',clf.n_iter_,'feature_count',len(vec.vocabulary_),flush=True)
serial_stats={q:(v[0],len(v[1])) for q,v in stats.items()}
threshold=0.36320773639469456
with open('/workspace/lexclf_full.pkl','wb') as f: pickle.dump((vec,clf,serial_stats,threshold),f)
