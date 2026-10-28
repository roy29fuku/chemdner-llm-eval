from model_fast import *
from collections import defaultdict,Counter
from pathlib import Path
import re,math
train=read_labels('training');dev=read_labels('development')
# tokens/frequency lexicon from only train labeled spans, token and phrase exact matches.
phrase=Counter(); singles=Counter(); cls=Counter()
for anns in train.values():
 for a,b,c,m in anns:
  phrase[m.lower()]+=1;cls[m.lower()]+=1
  if len(re.findall(r'\w+',m))==1: singles[m.lower()]+=1
# create dictionary alternatives, match as token subsequence (word tokens), exact case-insensitive preserving whole-token boundary.
patlist=[]
for m,n in phrase.items():
 if len(m)>1 and any(ch.isalpha() for ch in m):
  # search raw text, with boundaries that allow punctuation chemically internal but no substring in words
  patlist.append((re.compile(r'(?<![A-Za-z0-9])'+re.escape(m)+r'(?![A-Za-z0-9])',re.I),m,n))

def predict(text):
 spans=[]
 for pat,m,n in patlist:
  for mt in pat.finditer(text):spans.append((mt.start(),mt.end(),n,m))
 spans.sort(key=lambda x:(x[0],-(x[1]-x[0]),-x[2]))
 # select longest non-overlap sequentially
 res=[];i=0
 while i<len(spans):
  s,e,n,m=spans[i]; same=[z for z in spans[i:] if z[0]==s]
  z=max(same,key=lambda x:(x[1]-x[0],x[2]));res.append((z[0],z[1]));i+=len(same)
 return set(res)
for kind in ['development']:
 gold=set();pred=set()
 for (p,s),anns in dev.items():
  text=next(x[2] for x in read_articles('development') if x[:2]==(p,s))
  gold|={(p,s,a,b) for a,b,c,m in anns}
  pred|={(p,s,a,b) for a,b in predict(text)}
 tp=len(gold&pred);print('dict',tp,len(pred),len(gold),2*tp/(len(pred)+len(gold)))
