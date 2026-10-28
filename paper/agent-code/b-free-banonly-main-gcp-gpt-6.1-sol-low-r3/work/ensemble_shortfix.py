import os,json,argparse,re
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
from post_fixed import filter_pred,normalize_pred,ELEMENTS
LEXICON={l.rstrip('\n').split('\t')[4].lower() for l in open('/workspace/data/training.annotations.txt')}
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

def valid_word_span(t,a,b):
 name=t[a:b]
 cut_start=a>0 and t[a-1].isalpha() and t[a].isalpha()
 cut_end=b<len(t) and t[b-1].isalpha() and t[b].isalpha()
 if cut_start:
  if a>0 and t[a-1]=='p' and name in {'Ser','Thr','Tyr','His','Asp','Glu','Ala','Arg','Cys','Ile','Leu','Lys','Met','Phe','Pro','Trp','Val','Asn','Gln','Gly'} and (a<2 or not t[a-2].isalpha()):cut_start=False
  if t[a-1] in {'ν','τ'} and re.fullmatch('[A-Z0-9]+',name):cut_start=False
  if any(t[:a].endswith(head) for head in ['OBJECTIVE','CONCLUSIONS','RESULTS','BACKGROUND','METHODS','PURPOSE']):cut_start=False
  if any(t[:a].endswith(unit) for unit in ['mM','μg','µg']):cut_start=False
 if cut_end:
  if name in ELEMENTS and t[b:].startswith('SOD'):cut_end=False
  if name=='Met' and re.match(r'AP[0-9]',t[b:]):cut_end=False
 return not (cut_start or cut_end)

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
 repaired=set()
 for p,s,a,b in pred:
  t=text[p,s];start=a;end=b
  while start>0 and t[start-1].isalpha():start-=1
  while end<len(t) and t[end].isalpha():end+=1
  if (start!=a or end!=b) and end-start>3 and t[start:end].lower() in LEXICON:a,b=start,end
  repaired.add((p,s,a,b))
 pred={x for x in repaired if valid_word_span(text[x[:2]],x[2],x[3])}
 return filter_pred(pred,text)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--split',default='development');ap.add_argument('--spec',required=True);ap.add_argument('--out');ap.add_argument('--scale',type=float,default=1);args=ap.parse_args()
 if args.split=='test':LEXICON.update(l.rstrip('\n').split('\t')[4].lower() for l in open(args.data+'/development.annotations.txt'))
 docs,gold=read(args.data,args.split);off=offsets(docs,BASE);cache={BASE:off};total=[np.zeros((len(o),3),np.float32) for o in off];weight=0
 for entry in json.load(open(args.spec)):
  name=entry.get('model',BASE);w=entry.get('weight',1)
  if name not in cache:cache[name]=offsets(docs,name)
  data=np.load(entry['path'])
  for i in range(len(docs)):
   pr=data['arr_'+str(i)]
   pr=pr/np.maximum(pr.sum(axis=-1,keepdims=True),1e-9)
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
