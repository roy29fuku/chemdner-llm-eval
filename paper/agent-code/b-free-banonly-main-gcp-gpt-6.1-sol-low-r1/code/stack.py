import argparse,os,json,shutil,time
import numpy as np
import joblib
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from project import project
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--out',required=True);p.add_argument('--mode',choices=['prob','ratio'],default='ratio');p.add_argument('--apply',action='store_true');args=p.parse_args();threadpool_limits(limits=4)
ROOT='/workspace';tok=AutoTokenizer.from_pretrained(ROOT+'/code/tokenizer');split='test' if args.apply else 'development';short='test' if args.apply else 'dev'
ann={}
if not args.apply:
 for line in open(ROOT+'/data/development.annotations.txt'):
  pm,s,a,b,*_=line.strip().split('\t');ann.setdefault((pm,s),[]).append((int(a),int(b)))
docs=[];ys=[];groups=[]
for line in open(ROOT+'/data/'+split+'.abstracts.txt'):
 pm,title,abstract=line.rstrip('\n').split('\t')
 for s,txt in [('T',title),('A',abstract)]:
  off=tok(txt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];docs.append((pm,s,txt,off));groups.extend([int(pm)%2]*len(off))
  y=np.zeros(len(off),np.int8)
  for a,b in sorted(ann.get((pm,s),[]),key=lambda x:x[1]-x[0]):
   inds=[i for i,(x,z) in enumerate(off) if x>=a and z<=b and z>x]
   if inds:y[inds]=2;y[inds[0]]=1
  ys.append(y)
y=np.concatenate(ys);group=np.asarray(groups,np.int8);features=[]
if args.apply:args.mode=json.load(open(args.out+'/meta.json'))['mode']
for run in args.runs:
 cache=run+'/'+short+'_anchor_logprobs.npz'
 if os.path.exists(cache):
  z=np.load(cache);arr=[z['arr_'+str(i)] for i in range(len(z.files))]
 else:
  z=np.load(run+'/'+short+'_logits.npz');arr=[z['arr_'+str(i)] for i in range(len(z.files))];rt=AutoTokenizer.from_pretrained(run)
  if rt.get_vocab()!=tok.get_vocab():arr=[project(lg,rt(txt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'],off,len(txt)) for (_,_,txt,off),lg in zip(docs,arr)]
  else:arr=[lg-np.log(np.exp(lg-lg.max(-1,keepdims=True)).sum(-1,keepdims=True))-lg.max(-1,keepdims=True) for lg in arr]
  np.savez_compressed(cache,*arr)
 a=np.concatenate(arr);features.append(np.exp(a) if args.mode=='prob' else np.clip(a[:,1:]-a[:,[0]],-15,15))
X=np.concatenate(features,axis=1).astype(np.float32);print('features',X.shape,args.mode,flush=True)
os.makedirs(args.out,exist_ok=True)
for f in os.listdir(ROOT+'/code/tokenizer'):shutil.copy(ROOT+'/code/tokenizer/'+f,args.out+'/'+f)
def save(logits,name):
 arr=[];st=0
 for _,_,txt,off in docs:arr.append(logits[st:st+len(off)]);st+=len(off)
 np.savez_compressed(args.out+'/'+name+'_logits.npz',*arr)
if args.apply:
 model=joblib.load(args.out+'/model.joblib');logits=model.decision_function(X).astype(np.float32);save(logits,'test')
else:
 oof=np.zeros((len(y),3),np.float32)
 for fold in [0,1]:
  t=time.time();train=group!=fold;valid=~train;model=LogisticRegression(C=.1,max_iter=200,tol=1e-5,solver='lbfgs');model.fit(X[train],y[train]);oof[valid]=model.decision_function(X[valid]);print('fold',fold,'iterations',model.n_iter_,'seconds',round(time.time()-t),flush=True)
 save(oof,'dev');model=LogisticRegression(C=.1,max_iter=200,tol=1e-5,solver='lbfgs');model.fit(X,y);joblib.dump(model,args.out+'/model.joblib');json.dump({'mode':args.mode,'runs':args.runs,'C':.1,'validation':'two folds grouped by PMID parity'},open(args.out+'/meta.json','w'),indent=2)
 print('weights',model.coef_.tolist(),'intercepts',model.intercept_.tolist(),flush=True)
print('FINISHED',args.out,flush=True)
