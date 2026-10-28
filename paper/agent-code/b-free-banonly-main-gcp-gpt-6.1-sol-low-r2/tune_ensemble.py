import argparse,json
import numpy as np
from transformers import AutoTokenizer
from viterbi import tags
p=argparse.ArgumentParser();p.add_argument('--scores',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--weights',nargs='+',type=float);a=p.parse_args()
tok=AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract');docs=[];gold=set()
for l in open('/workspace/data/development.annotations.txt'):
 pm,c,s,e,*_=l.rstrip('\n').split('\t');gold.add((pm,c,int(s),int(e)))
for l in open('/workspace/data/development.abstracts.txt'):
 pm,t,ab=l.rstrip('\n').split('\t')
 for c,text in [('T',t),('A',ab)]:docs.append((pm,c,tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']))
arrays=[]
for path in a.scores:
 z=np.load(path);arr=[z['arr_'+str(i)] for i in range(len(docs))];assert all(x.shape==(len(d[2]),3) for x,d in zip(arr,docs));arrays.append(np.concatenate(arr))
M=np.stack(arrays);lens=[len(d[2]) for d in docs];starts=np.cumsum([0]+lens)
def score(w,b,ib):
 logits=np.tensordot(np.asarray(w,np.float32)/sum(w),M,axes=(0,0));logits[:,1]+=b;logits[:,2]+=ib;pred=set()
 for d,s,e in zip(docs,starts[:-1],starts[1:]):
  if s==e:continue
  labels=tags(logits[s:e]);pm,c,off=d;begin=None;end=None
  for label,(u,v) in zip(labels,off):
   if label!=2 and begin is not None:pred.add((pm,c,begin,end));begin=None
   if label==1 or (label==2 and begin is None):begin=u
   if label:end=v
  if begin is not None:pred.add((pm,c,begin,end))
 return 2*len(pred&gold)/(len(pred)+len(gold))
w=a.weights or [1.]*len(arrays);assert len(w)==len(arrays);b=-.5;ib=1.;best=score(w,b,ib)
print('INITIAL',best,w,b,ib,flush=True)
for round in range(2):
 for bb in [-1.5,-1,-.5,0,.5,1]:
  for ii in [-.5,0,.5,1,1.5,2]:
   f=score(w,bb,ii)
   if f>best:best=f;b=bb;ib=ii;print('BIAS',best,w,b,ib,flush=True)
 for j in range(1,len(w)):
  for v in [0,.5,1,1.5,2,3]:
   nw=w.copy();nw[j]=v;f=score(nw,b,ib)
   if f>best:best=f;w=nw;print('WEIGHT',best,w,b,ib,flush=True)
json.dump(dict(f1=best,weights=w,bias=b,ibias=ib,scores=a.scores),open(a.out,'w'),indent=2);print('BEST',best,w,b,ib,flush=True)
