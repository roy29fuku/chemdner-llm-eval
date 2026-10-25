import argparse
from pathlib import Path
import numpy as np
from evaluate import load_corpus,decode,write
from lexicon import metric

def softmax(x):
 x=x-x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)

def load_records(path,defaultoffs):
 d=np.load(path);scores=d['logits'].astype(np.float32);lengths=d['lengths'];cuts=np.cumsum(lengths)[:-1]
 offs=np.split(d['offsets'],cuts) if 'offsets' in d else defaultoffs
 return list(zip(np.split(scores,cuts),offs))

def merge(records,docs,weights,method='probability'):
 scores=[];offsets=[]
 for di,(p,s,text) in enumerate(docs):
  if not text:scores.append(np.empty((0,3),np.float32));offsets.append([]);continue
  total=np.zeros((len(text),3),np.float32);points=set();covered=np.zeros(len(text),bool)
  for rec,weight in zip(records,weights):
   raw,offs=rec[di];probs=softmax(raw) if len(raw) else raw
   char=np.zeros((len(text),3),np.float32);char[:,0]=1
   for prob,(a,b) in zip(probs,offs):
    if b<=a:continue
    covered[a:b]=True;char[a:b,0]=prob[0];char[a:b,1]=.000001;char[a:b,2]=prob[1]+prob[2]
    char[a]=prob;points.add(int(a));points.add(int(b))
   total+=weight*(np.log(np.maximum(char,1e-6)) if method=='geometric' else char)
  total/=sum(weights)
  boundaries=sorted(points);off=[(a,b) for a,b in zip(boundaries,boundaries[1:]) if covered[a] and not text[a:b].isspace()]
  v=np.array([total[a] for a,b in off],np.float32)
  if method!='geometric':v=np.log(np.maximum(v,1e-7))
  scores.append(v);offsets.append(off)
 return scores,offsets

def main():
 ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--split',default='development');ap.add_argument('--weights',nargs='+',type=float);ap.add_argument('--bias',type=float);ap.add_argument('--method',default='probability');ap.add_argument('--out');args=ap.parse_args()
 docs,gold,defaultoffs=load_corpus(args.split);records=[load_records(p,defaultoffs) for p in args.files];weights=args.weights or [1]*len(records);scores,offs=merge(records,docs,weights,args.method)
 best=(0,None,None)
 for bias in ([args.bias] if args.bias is not None else [-.3,-.15,0,.15,.3]):
  pred=decode(scores,docs,offs,bias);m=metric(pred,gold);print('bias',bias,'metric',m,flush=True)
  if m[0]>best[0] or not gold:best=(m[0],bias,pred)
 if args.out:write(args.out,best[2])
 print('best',best[:2],flush=True)

if __name__=='__main__':main()
