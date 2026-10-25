import os,json,argparse
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
from post import filter_pred,normalize_pred
BASE='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
def read(root,split):
 docs=[];gold=set()
 for line in open(root+'/'+split+'.abstracts.txt'):
  p,t,a=line.rstrip('\n').split('\t')
  docs.extend([(p,'T',t),(p,'A',a)])
 ann=root+'/'+split+'.annotations.txt'
 if os.path.exists(ann):
  for l in open(ann):
   p,s,a,b,*_=l.rstrip('\n').split('\t');gold.add((p,s,int(a),int(b)))
 return docs,gold

def offsets(docs,name):
 tok=AutoTokenizer.from_pretrained(name)
 return [tok(t,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'] for p,s,t in docs]

def remap(pr,source,target):
 if source==target:return pr
 if not source:return np.zeros((len(target),3),np.float32)
 so=np.asarray(source);ta=np.asarray(target)
 if not len(target):return np.zeros((0,3),np.float32)
 idx=np.searchsorted(so[:,1],ta[:,0],side='right').clip(0,len(source)-1)
 result=pr[idx].copy();inner=so[idx,0]<ta[:,0]
 result[inner,2]+=result[inner,1];result[inner,1]=0
 return result

def decode(probs,docs,off,scale=1):
 pred=set()
 for pr,(p,s,t),ofs in zip(probs,docs,off):
  active=None;pr=pr.copy();pr[:,0]*=scale
  for lab,(a,b) in zip(pr.argmax(-1),ofs):
   if lab in (0,1):
    if active is not None:pred.add((p,s,*active));active=None
   if lab in (1,2):
    if active is None:active=[a,b]
    else:active[1]=b
  if active is not None:pred.add((p,s,*active))
 text={(p,s):t for p,s,t in docs}
 pred=normalize_pred(pred,text)
 pred={x for x in pred if not ((x[2]>0 and text[x[:2]][x[2]-1].isalpha() and text[x[:2]][x[2]].isalpha()) or (x[3]<len(text[x[:2]]) and text[x[:2]][x[3]-1].isalpha() and text[x[:2]][x[3]].isalpha()))}
 return filter_pred(pred,text)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--split',default='development');ap.add_argument('--spec',required=True);ap.add_argument('--out');ap.add_argument('--scale',type=float,default=1);args=ap.parse_args()
 docs,gold=read(args.data,args.split);off=offsets(docs,BASE);cache={BASE:off};total=[np.zeros((len(o),3),np.float32) for o in off];weight=0
 for entry in json.load(open(args.spec)):
  name=entry.get('model',BASE);w=entry.get('weight',1)
  if name not in cache:cache[name]=offsets(docs,name)
  data=np.load(entry['path'])
  for i in range(len(docs)):
   pr=data['arr_'+str(i)]
   assert len(pr)==len(cache[name][i]),(i,len(pr),len(cache[name][i]))
   total[i]+=w*remap(pr,cache[name][i],off[i])
  weight+=w
 total=[p/weight for p in total]
 if gold:
  for scale in [.25,.4,.6,.8,1,1.2,1.5,2]:
   pred=decode(total,docs,off,scale);tp=len(pred&gold)
   print(json.dumps(dict(scale=scale,f1=2*tp/(len(pred)+len(gold)),precision=tp/max(1,len(pred)),recall=tp/len(gold),mentions=len(pred))),flush=True)
 if args.out:
  pred=decode(total,docs,off,args.scale)
  with open(args.out,'w') as f:
   for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')
  print('Wrote',len(pred),'mentions',flush=True)
if __name__=='__main__':main()
