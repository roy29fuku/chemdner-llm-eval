import sys,random
sys.path.insert(0,'/workspace');import ner
from candidate_model import cands_for,featurize
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def combine(files):
 docs={}; anns=defaultdict(lambda:defaultdict(list))
 for fn in files:
  docs.update(ner.load_articles(fn));aa=ner.load_annotations(fn)
  for p,ss in aa.items():
   for s,ls in ss.items():anns[p][s].extend(ls)
 return docs,anns

def make(docs,anns,knownanns):
 known={a[2].lower() for ss in knownanns.values() for ls in ss.values() for a in ls}
 X=[]; y=[]; meta=[]
 for p,secs in docs.items():
  for sec,text in secs.items():
   toks,cs=cands_for(text)
   gold={(a,b) for a,b,_,_ in anns.get(p,{}).get(sec,[])}
   for c in cs:
    X.append(featurize(text,toks,c,sec=='T',known));y.append(int((c[2],c[3]) in gold));meta.append((p,sec,text,toks,c))
 return X,np.array(y),meta

files=['training','development'];docs,anns=combine(files);ids=list(docs);random.Random(10).shuffle(ids); folds=[set(ids[i::5]) for i in range(5)]
gold={(p,s,a,b)for p,ss in anns.items()for s,aa in ss.items()for a,b,_,_ in aa};pred_by_th={x:set() for x in [.02,.03,.04,.05,.06,.08,.1,.15,.2,.3]};union_by_th={x:set() for x in pred_by_th}
for k,validids in enumerate(folds):
 td={p:docs[p] for p in ids if p not in validids}; vd={p:docs[p] for p in validids}; ta={p:anns[p] for p in td}; va={p:anns[p] for p in vd}
 X,y,_=make(td,ta,ta); VX,valid_y,vm=make(vd,va,ta)
 v=DictVectorizer();xx=v.fit_transform(X);vv=v.transform(VX)
 for z in [xx,vv]:z.indices=z.indices.astype(np.int32);z.indptr=z.indptr.astype(np.int32)
 clf=LogisticRegression(C=.15,max_iter=250,class_weight=None,solver='lbfgs').fit(xx,y);probs=clf.predict_proba(vv)[:,list(clf.classes_).index(1)]
 # Exact matches to chemical surfaces learned from other articles.
 knownseq={tuple(t[0].lower() for t in ner.tokenize(a[2])) for ss in ta.values() for ls in ss.values() for a in ls}
 direct=set()
 for p,secs in vd.items():
  for sec,text in secs.items():
   toks=ner.tokenize(text); lo=[t[0].lower() for t in toks]
   for i in range(len(lo)):
    for n in range(1,min(12,len(lo)-i)+1):
     seq=tuple(lo[i:i+n]); base=seq
     if seq not in knownseq and len(seq[-1])>4 and seq[-1].endswith('s'): base=seq[:-1]+(seq[-1][:-1],)
     if base in knownseq and toks[i+n-1][2]-toks[i][1]>=3: direct.add((p,sec,toks[i][1],toks[i+n-1][2]))
 # CRF scores strings not covered by ontology candidates.
 known_terms=ner.terms_from_annotations(ta)
 tx,ty,_=ner.build_data(td,ta,known_terms);vx,seq_y,mv=ner.build_data(vd,va,known_terms)
 crf=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=.02,c2=.1,max_iterations=80,all_possible_transitions=True)
 crf.fit(tx,ty); yp=crf.predict(vx)
 crfpred={(p,s,a,b)for(p,s,t,toks,ann),labs in zip(mv,yp)for a,b in ner.spans_from_labels(toks,labs)}
 print('fold',k,'train/valid candidates',len(y),len(valid_y),'positive',sum(y),sum(valid_y),'crf docs',len(seq_y),flush=True)
 for th in pred_by_th:
  predicted={(p,s,c[2],c[3])for(p,s,t,toks,c),pr in zip(vm,probs)if pr>=th}
  pred_by_th[th].update(predicted)
  union_by_th[th].update(predicted|direct|crfpred)
for th,pred in pred_by_th.items():print(th,'model',ner.score(pred,gold),'model+known+crf',ner.score(union_by_th[th],gold))
