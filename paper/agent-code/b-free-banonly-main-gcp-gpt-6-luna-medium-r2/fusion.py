from model import *
from crfmodel import prepare,TOK,spans_from_tagger
import pycrfsuite
from itertools import product
tr=load('training.abstracts.txt','training.annotations.txt'); dv=load('development.abstracts.txt','development.annotations.txt')
A,freq,df=make_automaton(tr); trc=get_cands(tr,A); dvc=get_cands(dv,A)
vec=DictVectorizer(sparse=True); X=vec.fit_transform([x[4] for x in trc]); X.indices=X.indices.astype('int32'); X.indptr=X.indptr.astype('int32')
y=[int(x[-1]) for x in trc]
clf=LogisticRegression(max_iter=200,C=.5,solver='liblinear'); clf.fit(X,y)
Xv=vec.transform([x[4] for x in dvc]); Xv.indices=Xv.indices.astype('int32'); Xv.indptr=Xv.indptr.astype('int32'); probs=clf.predict_proba(Xv)[:,1]
gold={(pm,a[0],a[1],a[2]) for pm,d in dv.items() for a in d['anns']}
# candidate predictions at varied threshold
CP={t:{(x[0],x[1],x[2],x[3]) for x,p in zip(dvc,probs) if p>=t} for t in [.2,.3,.39,.45,.5,.6]}
Xvseq,Yv,metav=prepare(dv); tagger=pycrfsuite.Tagger(); tagger.open('/tmp/chem.crfsuite')
crf=[]
for (pm,sec,offs,text),feats in zip(metav,Xvseq):
 ts=list(TOK.finditer(text))
 crf.extend((pm,sec,a,b,c) for a,b,c,_ in spans_from_tagger(tagger,feats,ts))
CR={t:{(pm,s,a,b) for pm,s,a,b,p in crf if p>=t} for t in [.0,.2,.3,.4,.45,.5,.6,.7]}
for t,ct in CP.items():
 print('C',t,len(ct),evaluate(ct,gold))
for t,cr in CR.items():
 print('R',t,len(cr),evaluate(cr,gold))
print('intersect crf .4, cand .39',len(CR[.4]&CP[.39]),'cand correct not crf',len((CP[.39]-CR[.4])&gold),'crf missing cand correct',len((CR[.4]-CP[.39])&gold))
for tc,trc_ in product([.2,.3,.39,.45,.5,.6],[.2,.3,.4,.45,.5,.6,.7]):
 pred=CP[tc]|CR[trc_]; p,r,f=evaluate(pred,gold)
 if f>.800: print('U',tc,trc_,len(pred),round(p,4),round(r,4),round(f,4))
