import re, pickle, sys
from model_fast import *
from collections import Counter,defaultdict
from pathlib import Path

def best_labels():
 tr=read_labels('training'); counts=Counter(m.lower() for aa in tr.values() for a,b,c,m in aa)
 pats=[]
 for m,n in counts.items():
  if len(m)>2 and any(ch.isalpha() for ch in m):
   pats.append((m,n))
 # sorted alternation noncapturing; match order dictated longest, avoids ambiguity
 pats.sort(key=lambda z:-len(z[0]))
 return re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(m) for m,n in pats)+r')(?![A-Za-z0-9])',re.I),counts
pat,counts=best_labels()
# validate on dev efficient regex (must confirm)
for kind in ['development']:
 ann=read_labels(kind);txts={(p,s):t for p,s,t in read_articles(kind)}
 g=set();p=set()
 for k,aa in ann.items():
  txt=txts[k]
  g.update((*k,a,b) for a,b,c,m in aa)
  p.update((*k,x.start(),x.end()) for x in pat.finditer(txt))
 tp=len(g&p);print('dictionary',tp,len(p),len(g),2*tp/(len(p)+len(g)),flush=True)
