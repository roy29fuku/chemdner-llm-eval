import re, collections, numpy as np, pickle, os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from model import *
P='/workspace/data/'

def get_xy(docs,anns):
 X=[];Y=[]; infos=[]; poss=[]
 for pmid,(title,abstract) in docs.items():
  for sec,text in [('T',title),('A',abstract)]:
   toks=tokenize(text); spans=anns.get((pmid,sec),[])
   for i,t in enumerate(toks):
    f=token_features(t,i,toks,text,sec)
    X.append(f); poss.append((pmid,sec,text,toks,i))
    s,e=t[1:]
    overlaps=[(a,b) for a,b in spans if s<b and e>a]
    if not overlaps:Y.append('O')
    else:Y.append('B' if s==overlaps[0][0] else 'I')
   infos.extend((pmid,sec,text,toks) for _ in [0])
 return X,np.array(Y),poss,infos

def seq_predict(clf, X, poss):
 labels=clf.predict(X); groups=collections.defaultdict(list)
 for lab,pos in zip(labels,poss):groups[pos[:2]].append((pos,lab))
 out=[]
 for key,items in groups.items():
  _,sec,text,toks,_=items[0][0]; pmid=key[0];start=end=None
  for (p,lab) in items:
   w,s,e=p[3][p[4]]
   if lab=='B' or (lab=='I' and start is None):
    if start is not None:out.append((pmid,sec,start,end))
    start=s;end=e
   elif lab=='I':end=e
   else:
    if start is not None:out.append((pmid,sec,start,end));start=end=None
  if start is not None:out.append((pmid,sec,start,end))
 return out
if __name__=='__main__':
 train=load_docs(P+'training.abstracts.txt');dev=load_docs(P+'development.abstracts.txt');print('docs',flush=True)
 ann=load_ann(P+'training.annotations.txt');da=load_ann(P+'development.annotations.txt')
 Xt,Yt,pt,_=get_xy(train,ann);print('train seq',flush=True);Xd,Yd,pd,_=get_xy(dev,da);print('dev seq',flush=True)
 print('loaded',len(Xt),len(Xd),collections.Counter(Yt),flush=True)
 vec=DictVectorizer(); Xt=vec.fit_transform(Xt);Xd=vec.transform(Xd); print('vectorized',Xt.shape,flush=True)
 for alpha in [.00001,.0001,.001,.01]:
  clf=SGDClassifier(loss='modified_huber',alpha=alpha,max_iter=30,tol=1e-3,class_weight={'O':1,'B':2,'I':1.3},n_jobs=-1,random_state=42)
  clf.fit(Xt,Yt); pred=seq_predict(clf,Xd,pd)
  gold=[(pmid,sec,a,b) for (pmid,sec),spans in da.items() for a,b in spans]
  print(alpha,evaluate(gold,pred),flush=True)
  with open('/workspace/output/linear_model.pkl','wb') as f:pickle.dump((vec,clf),f)
