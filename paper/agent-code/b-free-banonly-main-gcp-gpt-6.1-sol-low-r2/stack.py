import argparse,json,time
import numpy as np
from transformers import AutoTokenizer
from threadpoolctl import threadpool_limits
P=argparse.ArgumentParser();P.add_argument('--scores',nargs='+',required=True);P.add_argument('--split',default='development');P.add_argument('--out',required=True);P.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract');P.add_argument('--fit',action='store_true');P.add_argument('--params',required=True);P.add_argument('--shape',action='store_true');P.add_argument('--C',type=float,default=.005);a=P.parse_args()
tok=AutoTokenizer.from_pretrained(a.model);raws=[np.load(p) for p in a.scores];anns={}
if a.fit:
 for l in open('/workspace/data/development.annotations.txt'):
  pm,c,s,e,*_=l.rstrip('\n').split('\t');anns.setdefault((pm,c),[]).append((int(s),int(e)))
Xs=[];ys=[];lens=[];papers=[];docidx=0
for l in open('/workspace/data/'+a.split+'.abstracts.txt'):
 pm,t,ab=l.rstrip('\n').split('\t')
 for c,text in [('T',t),('A',ab)]:
  off=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];n=len(off);lens.append(n);papers.append(pm)
  scores=[r['arr_'+str(docidx)] for r in raws];docidx+=1
  assert all(s.shape==(n,3) for s in scores)
  margins=[np.clip(s[:,1:]-s[:,0:1],-15,15) for s in scores];x=np.concatenate(margins,axis=1)
  if a.shape:
   avg=sum(margins)/len(margins);ent=np.max(avg,axis=1)
   prev=np.concatenate([[-15],ent[:-1]]) if n else ent; nxt=np.concatenate([ent[1:],[-15]]) if n else ent
   shp=[]
   for s,e in off:
    v=text[s:e];shp.append([np.log1p(e-s),v.isupper(),v.istitle(),any(k.isdigit() for k in v),any(ord(k)>127 for k in v),s==0 or not text[s-1].isalnum(),e==len(text) or not text[e].isalnum(),c=='T'])
   x=np.concatenate([x,np.asarray(shp,np.float32).reshape(n,8),np.asarray(prev,np.float32)[:,None],np.asarray(nxt,np.float32)[:,None]],axis=1)
  Xs.append(x);lab=np.zeros(n,np.int32)
  if a.fit:
   spans=sorted(anns.get((pm,c),[]));k=0
   for i,(s,e) in enumerate(off):
    while k<len(spans) and spans[k][1]<=s:k+=1
    if k<len(spans) and s<spans[k][1] and e>spans[k][0]:lab[i]=1 if i==0 or off[i-1][1]<=spans[k][0] else 2
  ys.append(lab)
X=np.concatenate(Xs);Y=np.concatenate(ys);print('features',X.shape,flush=True)
def params(clf):
 return dict(mean=clf[0].mean_.tolist(),std=clf[0].scale_.tolist(),coef=clf[1].coef_.tolist(),intercept=clf[1].intercept_.tolist(),shape=a.shape,C=a.C,sources=a.scores)
def apply(p,X):
 return (((X-np.asarray(p['mean']))/np.asarray(p['std']))@np.asarray(p['coef']).T+np.asarray(p['intercept'])).astype('float32')
with threadpool_limits(limits=6):
 if a.fit:
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  unique=sorted(set(papers));np.random.default_rng(234).shuffle(unique);folds={p:i%2 for i,p in enumerate(unique)};fold=np.concatenate([np.full(n,folds[p],np.int8) for n,p in zip(lens,papers)]);out=np.zeros((len(X),3),np.float32)
  for f in range(2):
   clf=make_pipeline(StandardScaler(),LogisticRegression(C=a.C,max_iter=200,solver='lbfgs',tol=1e-5));t=time.time();clf.fit(X[fold!=f],Y[fold!=f]);assert list(clf[1].classes_)==[0,1,2];out[fold==f]=apply(params(clf),X[fold==f]);print('fold',f,'seconds',time.time()-t,flush=True)
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=a.C,max_iter=200,solver='lbfgs',tol=1e-5));clf.fit(X,Y);json.dump(params(clf),open(a.params,'w'),indent=2)
 else:
  p=json.load(open(a.params));assert a.shape==p['shape'];out=apply(p,X)
arrays=[];start=0
for n in lens:arrays.append(out[start:start+n]);start+=n
np.savez_compressed(a.out,*arrays);print('DONE',a.out,flush=True)
