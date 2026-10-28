from lexicon import read,metric
from postprocess import load_preds

def trim(pred,docs):
 texts={(p,s):t for p,s,t in docs};out=set()
 for p,s,a,b in pred:
  text=texts[p,s]
  while a<b and (text[a].isspace() or text[a] in ',;:'):a+=1
  while a<b and (text[b-1].isspace() or text[b-1] in ',.;:®™'):b-=1
  if b-a>=2 and text[b-2:b] in ["'s",'’s']:b-=2
  if a<b:out.add((p,s,a,b))
 return out

if __name__=='__main__':
 import sys
 docs,g=read('development');pred=load_preds(sys.argv[1]);p=trim(pred,docs);print('before',metric(pred,g),'after',metric(p,g),'changed',len(p-pred),'correct added',len((p-pred)&g),'correct removed',len((pred-p)&g))
