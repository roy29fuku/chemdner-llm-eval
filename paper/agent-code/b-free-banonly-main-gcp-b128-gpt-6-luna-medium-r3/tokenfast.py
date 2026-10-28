import ner,numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.build_lexicon(ta);X,Y,_,_=ner.examples(tr,ta,lex);Xd,Yd,loc,_=ner.examples(dv,da,lex)
vec=DictVectorizer(); x=vec.fit_transform([f for seq in X for f in seq]).tocsr(); xd=vec.transform([f for seq in Xd for f in seq]).tocsr();
for mat in (x,xd): mat.indices=mat.indices.astype(np.int32); mat.indptr=mat.indptr.astype(np.int32)
y=np.array([z for seq in Y for z in seq]);
# gold spans
gold=set()
for (pm,s,t,toks),ys in zip(loc,Yd):
 for b,e in ner.spans_from_labels(t,toks,ys):gold.add((pm,s,b,e))
print('matrices',x.shape,flush=True)
for weights in [None,'balanced',{'B':25,'I':25,'O':1}]:
 clf=SGDClassifier(loss='log_loss',alpha=1e-5,class_weight=weights,max_iter=60,tol=1e-3,random_state=1,n_jobs=1)
 clf.fit(x,y); prob=clf.predict_proba(xd); classes=list(clf.classes_)
 print('\nweights',weights,classes,flush=True)
 B=classes.index('B'); I=classes.index('I')
 best=(0,None)
 for tb in [.03,.06,.1,.15,.2,.3,.4,.5]:
  for ti in [.03,.06,.1,.15,.2,.3,.4,.5]:
   p=set(); off=0
   for (pm,s,t,toks) in loc:
    n=len(toks); sub=prob[off:off+n]; off+=n; labels=[]
    for row in sub:
     b=row[B]; ii=row[I]; labels.append('B' if b>=tb and b>=ii else 'I' if ii>=ti else 'O')
    for b,e in ner.spans_from_labels(t,toks,labels):p.add((pm,s,b,e))
   sc=ner.score(gold,p)
   if sc[2]>best[0]:best=(sc[2],(tb,ti,sc,len(p)))
 print('BEST',best,flush=True)
