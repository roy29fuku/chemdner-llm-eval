import argparse,os,json
import numpy as np
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--bias',type=float);p.add_argument('--output',default='/workspace/output/predictions.tsv');p.add_argument('--devonly',action='store_true');p.add_argument('--range',dest='bias_range',type=float,default=.8);p.add_argument('--begin-bias',type=float,default=0);p.add_argument('--average',choices=['log','prob'],default='log');p.add_argument('--weights',nargs='+',type=float);p.add_argument('--constraint',choices=['none','word','bio','both'],default='none');args=p.parse_args()
from constrained import viterbi,path as fast_path
MASKS={}
from project import project
tok=AutoTokenizer.from_pretrained('/workspace/code/tokenizer')
def read(split):
 gold=set();path=f'/workspace/data/{split}.annotations.txt'
 if os.path.exists(path):
  for l in open(path):
   p,s,a,b,*_=l.rstrip('\n').split('\t');gold.add((p,s,int(a),int(b)))
 docs=[]
 for l in open(f'/workspace/data/{split}.abstracts.txt'):
  p,t,a=l.rstrip('\n').split('\t')
  for s,txt in [('T',t),('A',a)]:
   off=tok(txt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];docs.append((p,s,off,txt))
   mask=np.zeros(len(off),np.bool_)
   for i in range(1,len(off)):
    a=off[i][0];b=off[i-1][1];mask[i]=a==b and a>0 and a<len(txt) and txt[a-1].isalpha() and txt[a].isalpha()
   MASKS[(p,s)]=mask
 return docs,gold
def load(split):
 docs,_=read('development' if split=='dev' else split)
 arrays=[]
 for run in args.runs:
  cache=f'{run}/{split}_anchor_logprobs.npz'
  if os.path.exists(cache):
   z=np.load(cache);arrays.append([z[f'arr_{i}'] for i in range(len(z.files))]);continue
  z=np.load(f'{run}/{split}_logits.npz');arr=[z[f'arr_{i}'] for i in range(len(z.files))]
  rt=AutoTokenizer.from_pretrained(run)
  if rt.get_vocab()!=tok.get_vocab():
   arr=[project(lg,rt(txt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'],off,len(txt)) for (pm,s,off,txt),lg in zip(docs,arr)]
  else:
   arr=[lg-np.log(np.exp(lg-lg.max(-1,keepdims=True)).sum(-1,keepdims=True))-lg.max(-1,keepdims=True) for lg in arr]
  np.savez_compressed(cache,*arr)
  arrays.append(arr)
 return [(np.log(np.average(np.exp(v),axis=0,weights=args.weights)) if args.average=='prob' else np.average(v,axis=0,weights=args.weights)) for v in zip(*arrays)]
def decode(docs,arr,bias):
 pred=set()
 for (p,s,offs,txt),logits in zip(docs,arr):
  lg=logits.copy();lg[:,1:]+=bias;lg[:,1]+=args.begin_bias;tags=lg.argmax(-1) if args.constraint=='none' else fast_path(lg,MASKS[(p,s)] if args.constraint in ['word','both'] else np.zeros(len(offs),np.bool_),args.constraint in ['bio','both']);start=None;end=None
  for tag,(a,b) in zip(tags,offs):
   if tag!=2 and start is not None:pred.add((p,s,start,end));start=None
   if tag:
    if start is None:start=a
    end=b
  if start is not None:pred.add((p,s,start,end))
 return pred
if args.bias is None or args.devonly:
 docs,gold=read('development');arr=load('dev');best=(0,0)
 for bias in ([args.bias] if args.bias is not None else np.arange(-args.bias_range,args.bias_range+.01,.1)):
  pred=decode(docs,arr,bias);hit=len(pred&gold);f=2*hit/(len(pred)+len(gold));print(round(bias,2),hit/len(pred),hit/len(gold),f,flush=True)
  if f>best[0]:best=(f,float(bias))
 bias=best[1];print('BEST',best,flush=True)
 devpred=decode(docs,arr,bias)
 with open(args.output+'.dev','w') as f:
  for p,s,a,b in sorted(devpred):f.write(f'{p}\t{s}:{a}:{b}\n')
else:bias=args.bias
if args.devonly:raise SystemExit
arr=load('test');docs,_=read('test');pred=decode(docs,arr,bias)
with open(args.output,'w') as f:
 for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')
print('saved',args.output,len(pred),flush=True)
