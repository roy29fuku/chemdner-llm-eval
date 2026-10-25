import os,json,argparse
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
from viterbi import tags,typed_tags,bio_posterior
P=argparse.ArgumentParser();P.add_argument('--scores',nargs='+',required=True);P.add_argument('--weights',nargs='+',type=float);P.add_argument('--save-scores');P.add_argument('--typed',action='store_true');P.add_argument('--type-bias',nargs='+',type=float);P.add_argument('--posterior',action='store_true');P.add_argument('--split',default='development');P.add_argument('--bias',type=float,default=0);P.add_argument('--sweep',action='store_true');P.add_argument('--grid',action='store_true');P.add_argument('--ibias',type=float);P.add_argument('--title-bias',type=float);P.add_argument('--title-ibias',type=float);P.add_argument('--title-grid',action='store_true');P.add_argument('--out');P.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');args=P.parse_args()
tok=AutoTokenizer.from_pretrained(args.model,**({'do_lower_case':False} if 'biobert-base-cased' in args.model else {}))
ann={}
if args.split!='test':
 for line in open('/workspace/data/'+args.split+'.annotations.txt'):
  pm,sec,s,e,*_=line.rstrip('\n').split('\t');ann.setdefault((pm,sec),[]).append((int(s),int(e)))
docs=[]
for line in open('/workspace/data/'+args.split+'.abstracts.txt'):
 pm,t,a=line.rstrip('\n').split('\t')
 for sec,text in [('T',t),('A',a)]:
  off=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];docs.append(dict(pm=pm,sec=sec,text=text,offsets=off,gold=ann.get((pm,sec),[])))
arrays=[np.load(path) for path in args.scores]; weights=args.weights or [1.0]*len(arrays)
assert len(weights)==len(arrays)
assert all(len(a.files)==len(docs) for a in arrays)
scores=[sum(w*arr['arr_'+str(i)] for w,arr in zip(weights,arrays))/sum(weights) for i in range(len(docs))]
assert all(len(a)==len(d['offsets']) for a,d in zip(scores,docs)), 'Tokenizer or split mismatch'
if args.type_bias:
 assert len(args.type_bias)==8 and all(a.shape[1]==17 for a in scores)
 shifts=np.zeros(17,np.float32)
 for j,v in enumerate(args.type_bias):shifts[1+2*j:3+2*j]=v
 scores=[a+shifts for a in scores]
if args.posterior:scores=[bio_posterior(a) for a in scores]
if not args.typed and scores and scores[0].shape[1]>3:scores=[np.stack([a[:,0],np.logaddexp.reduce(a[:,1::2],axis=1),np.logaddexp.reduce(a[:,2::2],axis=1)],axis=1) for a in scores]
if args.save_scores:np.savez_compressed(args.save_scores,*scores)
def decode(bias,ibias=None,tbias=None,tibias=None):
 result=set()
 for doc,raw in zip(docs,scores):
  if not len(raw):continue
  logit=raw.copy();bi=tbias if doc['sec']=='T' and tbias is not None else bias;ii=tibias if doc['sec']=='T' and tibias is not None else (bias if ibias is None else ibias);logit[:,1::2]+=bi;logit[:,2::2]+=ii
  labels=typed_tags(logit) if args.typed else tags(logit);begin=None;end=None
  for tag,(s,e) in zip(labels,doc['offsets']):
   if tag!=2 and begin is not None:result.add((doc['pm'],doc['sec'],begin,end));begin=None
   if tag==1 or (tag==2 and begin is None):begin=s
   if tag:end=e
  if begin is not None:result.add((doc['pm'],doc['sec'],begin,end))
 return result
gold={(d['pm'],d['sec'],s,e) for d in docs for s,e in d['gold']}
best=-1;bestpred=None;bestconfig=None
settings=[(b,i) for b in [-1,-.5,0,.5,1] for i in [-1,-.5,0,.5,1]] if args.grid else [(b,args.ibias) for b in ([-1,-.5,0,.25,.5,.75,1,1.5,2] if args.sweep else [args.bias])]
settings=[(b,i,args.title_bias,args.title_ibias) for b,i in settings]
if args.title_grid:settings=[(args.bias,args.ibias,b,i) for b in [-1,-.5,0,.5,1] for i in [-1,-.5,0,.5,1]]
for bias,ibias,tbias,tibias in settings:
 pred=decode(bias,ibias,tbias,tibias)
 f1=0
 if gold:
  tp=len(gold&pred);f1=2*tp/(len(pred)+len(gold));print(bias,ibias,tbias,tibias,'F1',f1,'P',tp/len(pred),'R',tp/len(gold),flush=True)
 if f1>best:best=f1;bestpred=pred;bestconfig=(bias,ibias,tbias,tibias)
print('BEST',best,bestconfig,flush=True)
if args.out:
 pred=bestpred
 with open(args.out,'w') as f:
  for pm,sec,s,e in sorted(pred):f.write(f'{pm}\t{sec}:{s}:{e}\n')
 textmap={(d['pm'],d['sec']):d['text'] for d in docs}
 if gold:
  with open(args.out+'.errors','w') as f:
   for label,spans in [('FP',pred-gold),('FN',gold-pred)]:
    for pm,sec,s,e in sorted(spans):
     text=textmap[pm,sec];f.write(f'{label}\t{pm}\t{sec}\t{s}\t{e}\t{text[s:e]}\t{text[max(0,s-70):min(len(text),e+70)]}\n')
