import sys,collections,random
sys.path.insert(0,'/workspace');import model
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def make_examples(splits):
 X=[];y=[]
 for sp in splits:
  for d,s,t,tk,l in model.load(sp):
   for i,tag in enumerate(l):
    f={k:1 for k in model.word_features(tk,i)}
    w=tk[i][2].lower()
    # character ngrams capture unseen systematic names
    chars='^'+w+'$'
    for n in (2,3,4,5):
     for j in range(max(0,len(chars)-n+1)):
      f['cg%d=%s'%(n,chars[j:j+n])]=1
    # chemical-like morphology/character patterns
    X.append(f); y.append(tag)
 return X,y

def fit(splits):
 X,y=make_examples(splits);v=DictVectorizer();X=v.fit_transform(X)
 # balanced weighting overcomes heavy O class without overwhelming precision
 clf=LogisticRegression(C=0.15,max_iter=250,solver='lbfgs',class_weight=None,n_jobs=-1)
 clf.fit(X,y);return v,clf

def get_tags(v,clf,toks,bias=None):
 if not toks:return []
 feats=[]
 for i in range(len(toks)):
  f={k:1 for k in model.word_features(toks,i)};w=toks[i][2].lower(); chars='^'+w+'$'
  for n in (2,3,4,5):
   for j in range(max(0,len(chars)-n+1)):f['cg%d=%s'%(n,chars[j:j+n])]=1
  feats.append(f)
 probs=clf.predict_proba(v.transform(feats)); classes=list(clf.classes_);bias=bias or {}
 out=[];prev='O'
 for row in probs:
  scores={c:row[classes.index(c)]+bias.get(c,0) for c in classes}
  tag=max(classes,key=lambda c:scores[c])
  if tag=='I' and prev=='O':tag='B'
  out.append(tag);prev=tag
 return out

def evalmodel(splits,dev='development',use_dic=True):
 v,c=fit(splits);dic=model.get_dictionary(splits);tp=fp=fn=0
 for d,s,t,tk,g in model.load(dev):
  gs=set(model.spans_from_tags(tk,g));ps=set(model.spans_from_tags(tk,get_tags(v,c,tk)))
  if use_dic:ps.update(model.dict_matches(t,dic))
  tp+=len(gs&ps);fp+=len(ps-gs);fn+=len(gs-ps)
 print('TP FP FN',tp,fp,fn,'P/R/F1',tp/(tp+fp or 1),tp/(tp+fn or 1),2*tp/(2*tp+fp+fn or 1))

if __name__=='__main__':evalmodel(['training'])
