import re,collections
from model import load_docs,load_ann,evaluate,P
from collections import Counter,defaultdict
tr=load_docs(P+'training.abstracts.txt');dev=load_docs(P+'development.abstracts.txt'); ann=load_ann(P+'training.annotations.txt'); da=load_ann(P+'development.annotations.txt')
def norm(s): return re.sub(r'\s+',' ',s.lower()).strip()
freq=Counter()
for (pmid,sec),spans in ann.items():
 text=tr[pmid][0 if sec=='T' else 1]
 for a,b in spans:freq[norm(text[a:b])]+=1
# tokenization captures exact source offsets
from model import tokenize
for minf in [1,2,3,5]:
 d=defaultdict(list)
 for v,c in freq.items():
  if c>=minf and len(v)>1:
   toks=[x[0].lower() for x in tokenize(v)]
   if toks:d[tuple(toks)].append(v)
 maxlen=max(map(len,d))
 gold={(p,s,a,b) for (p,s),spans in da.items() for a,b in spans};pred=[]
 for pmid,(title,abstract) in dev.items():
  for sec,text in [('T',title),('A',abstract)]:
   toks=tokenize(text); low=[t[0].lower() for t in toks]; cand=[]
   for i in range(len(toks)):
    for n in range(1,min(maxlen,len(toks)-i)+1):
     vs=d.get(tuple(low[i:i+n]))
     if vs:
      cand.append((toks[i][1],toks[i+n-1][2],max(freq[v] for v in vs)))
   cand.sort(key=lambda x:(x[0],-(x[1]-x[0]),-x[2]));keep=[]
   for c in cand:
    ov=[z for z in keep if c[0]<z[1] and c[1]>z[0]]
    if not ov:keep.append(c)
    elif all(c[1]-c[0]>z[1]-z[0] for z in ov):keep=[z for z in keep if z not in ov]+[c]
   pred.extend((pmid,sec,a,b) for a,b,_ in keep)
 print(minf,evaluate(gold,pred),flush=True)
