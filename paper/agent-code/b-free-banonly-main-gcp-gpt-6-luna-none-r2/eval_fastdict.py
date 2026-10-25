from model_fast import *
from collections import Counter
import re,time
tr=read_labels('training'); forms=Counter(m.lower() for aa in tr.values() for a,b,c,m in aa if len(m)>2 and any(ch.isalpha() for ch in m))
terms=sorted(forms,key=lambda x:-len(x));pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in terms)+r')(?![A-Za-z0-9])',re.I)
dv=read_labels('development');txts={(p,s):t for p,s,t in read_articles('development')};g=set();p=set();t=time.time()
for k,aa in dv.items():
 txt=txts[k];g.update((*k,a,b) for a,b,c,m in aa)
 for x in pat.finditer(txt):p.add((*k,x.start(),x.end()))
tp=len(g&p);print('dictionary',tp,len(p),len(g),2*tp/(len(p)+len(g)),time.time()-t,flush=True)
