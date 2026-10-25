import sys,collections,pickle,re
sys.path.insert(0,'/workspace')
from model import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
P='/workspace/data/'
# Fast feature extraction with cached simple dictionaries; tokenize corpus separately

def get(docs,anns):
 X=[];Y=[]; blocks=[]
 for pmid,(title,abstract) in docs.items():
  for sec,text in [('T',title),('A',abstract)]:
   toks=tokenize(text); fs=[]; ys=[]
   spans=anns.get((pmid,sec),[]); j=0
   for i,t in enumerate(toks):
    w,s,e=t; low=w.lower(); sh=shapes(w)
    f={'w='+low:1,'shape='+sh:1,'pre='+low[:3]:1,'suf='+low[-3:]:1,'len='+str(min(20,len(w))):1,'sec='+sec:1}
    if w.isupper():f['upper']=1
    if w.istitle():f['title']=1
    if any(c.isdigit() for c in w):f['digit']=1
    for off in (-1,1):
     k=i+off
     if 0<=k<len(toks):
      q=toks[k][0].lower();f[f'{off}:w={q}']=1
      if off in (-1,1):f[f'{off}:sh={shapes(toks[k][0])}']=1
    fs.append(f)
    while j<len(spans) and spans[j][1]<=s:j+=1
    ol=spans[j] if j<len(spans) and s<spans[j][1] and e>spans[j][0] else None
    ys.append('O' if ol is None else ('B' if s==ol[0] else 'I'))
   X.extend(fs);Y.extend(ys); blocks.append((pmid,sec,text,toks,len(fs)))
 return X,Y,blocks

def prediction(clf,vec,X,blocks):
 labs=iter(clf.predict(vec.transform(X))); allp=[]
 for pmid,sec,text,toks,n in blocks:
  ls=[next(labs) for _ in range(n)]; st=en=None
  for (w,s,e),lab in zip(toks,ls):
   if lab=='B' or (lab=='I' and st is None):
    if st is not None:allp.append((pmid,sec,st,en))
    st=s;en=e
   elif lab=='I':en=e
   elif st is not None:allp.append((pmid,sec,st,en));st=en=None
  if st is not None:allp.append((pmid,sec,st,en))
 return allp
tr=load_docs(P+'training.abstracts.txt');dv=load_docs(P+'development.abstracts.txt');an=load_ann(P+'training.annotations.txt');da=load_ann(P+'development.annotations.txt')
Xt,Yt,Bt=get(tr,an);print('train',len(Xt),flush=True);Xd,Yd,Bd=get(dv,da);print('dev',len(Xd),flush=True)
vec=DictVectorizer(); Xt=vec.fit_transform(Xt); Xd=vec.transform(Xd);print('mat',Xt.shape,flush=True)
gold=[(p,s,a,b) for (p,s),spans in da.items() for a,b in spans]
for alpha in [.001]:
 clf=SGDClassifier(loss='log_loss',alpha=alpha,max_iter=10,tol=1e-3,class_weight={'O':1,'B':2,'I':1.2},n_jobs=8,random_state=42)
 clf.fit(Xt,Yt);print('fit',alpha,flush=True)
 pr=prediction(clf,vec,Xd,Bd);print(alpha,evaluate(gold,pr),flush=True)
 with open('/workspace/output/lin.pkl','wb') as f:pickle.dump((vec,clf),f)
