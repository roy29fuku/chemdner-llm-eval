import argparse,json,pickle
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from transformers import AutoTokenizer
from viterbi import viterbi
from lexicon import read,metric,build,predict

def load_corpus(split):
 path=Path('/workspace/code/'+split+'_tokens.pkl')
 if path.exists():return pickle.loads(path.read_bytes())
 docs,gold=read(split);tok=AutoTokenizer.from_pretrained('/workspace/tokenizer' if Path('/workspace/tokenizer').exists() else 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');offs=[tok(t,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'] for p,s,t in docs]
 path.write_bytes(pickle.dumps((docs,gold,offs)));return docs,gold,offs

def scores_from_files(files):
 arrays=[]
 for path in files:
  d=np.load(path);arrays.append(d['logits'].astype(np.float32));lens=d['lengths']
 scores=np.mean(arrays,axis=0);cuts=np.cumsum(lens)[:-1]
 return np.split(scores,cuts)

def decode(scores,docs,offs,bias=0):
 pred=set()
 for raw,(p,s,text),offsets in zip(scores,docs,offs):
  if len(raw)==0:continue
  v=raw.copy();v[:,1:]+=bias
  tags=viterbi(v)
  a=None;b=None
  for tag,(st,en) in zip(tags,offsets):
   if tag!=2 and a is not None:
    if a<b:pred.add((p,s,a,b))
    a=None
   if tag==1:a=st;b=en
   elif tag==2:b=en
  if a is not None and a<b:pred.add((p,s,a,b))
 return pred

def write(path,pred):
 with open(path,'w') as f:
  for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--split',default='development');ap.add_argument('--bias',type=float);ap.add_argument('--out');ap.add_argument('--errors');args=ap.parse_args()
 docs,gold,offs=load_corpus(args.split);scores=scores_from_files(args.files);best=(0,None,None)
 for bias in ([args.bias] if args.bias is not None else [-.6,-.3,0,.3,.6]):
  pred=decode(scores,docs,offs,bias);m=metric(pred,gold);print('bias',bias,'metric',m,flush=True)
  if m[0]>best[0] or not gold:best=(m[0],bias,pred)
 pred=best[2]
 if args.out:write(args.out,pred)
 if args.errors and gold:
  texts={(p,s):t for p,s,t in docs};rows=[]
  for typ,spans in [('FP',pred-gold),('FN',gold-pred)]:
   for p,s,a,b in sorted(spans):
    text=texts[p,s];rows.append((typ,p,s,a,b,text[a:b],text[max(0,a-70):min(len(text),b+70)]))
  with open(args.errors,'w') as f:
   for row in rows:f.write('\t'.join(map(str,row))+'\n')
 print('best',best[:2],flush=True)

if __name__=='__main__':main()
