import ner,numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.build_lexicon(ta);X,Y,_,_=ner.examples(tr,ta,lex);Xd,Yd,loc,_=ner.examples(dv,da,lex)
vec=DictVectorizer(); x=vec.fit_transform([f for seq in X for f in seq]); xd=vec.transform([f for seq in Xd for f in seq]); y=np.array([z for seq in Y for z in seq]);
gold=set()
for (pm,s,t,toks),ys in zip(loc,Yd):
 for b,e in ner.spans_from_labels(t,toks,ys):gold.add((pm,s,b,e))
print(x.shape,dict(zip(*np.unique(y,return_counts=True))))
for weights in [None,{'B':10,'I':10,'O':1},{'B':25,'I':25,'O':1},{'B':50,'I':50,'O':1}]:
 for C in [.1,1,10]:
  clf=LogisticRegression(C=C,class_weight=weights,max_iter=200,solver='lbfgs')
  clf.fit(x,y); prob=clf.predict_proba(xd); classes=list(clf.classes_)
  print('\n',weights,C,classes)
  best=(0,None)
  for tb in [.02,.05,.1,.15,.2,.3,.4,.5,.6]:
   for ti in [.02,.05,.1,.15,.2,.3,.4,.5,.6]:
    p=set(); off=0
    for (pm,s,t,toks) in loc:
     n=len(toks); sub=prob[off:off+n]; off+=n
     labels=[]
     for row in sub:
      b=row[classes.index('B')] if 'B' in classes else 0; ii=row[classes.index('I')] if 'I' in classes else 0
      labels.append('B' if b>=tb else 'I' if ii>=ti else 'O')
     for b,e in ner.spans_from_labels(t,toks,labels): p.add((pm,s,b,e))
    sc=ner.score(gold,p)
    if sc[2]>best[0]:best=(sc[2],(tb,ti,sc,len(p)))
  print('BEST',best)
