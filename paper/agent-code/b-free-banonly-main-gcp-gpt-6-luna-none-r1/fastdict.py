import re,collections
from model import *
from model import tokenize
P='/workspace/data/'
tr=load_docs(P+'training.abstracts.txt');dv=load_docs(P+'development.abstracts.txt');test=load_docs(P+'test.abstracts.txt');an=load_ann(P+'training.annotations.txt');da=load_ann(P+'development.annotations.txt')
def norm(s):return re.sub(r'\s+',' ',s.lower()).strip()
freq=collections.Counter()
for (p,s),spans in an.items():
 text=tr[p][0 if s=='T' else 1]
 for a,b in spans:freq[norm(text[a:b])]+=1
def make_dic(minf=1):
 d=collections.defaultdict(list)
 for v,c in freq.items():
  if c>=minf and len(v)>1:
   ts=tuple(t[0].lower() for t in tokenize(v))
   if ts:d[ts].append(v)
 return d
def predict(docs,minf=1,threshold=1,allow_single=False):
 d=make_dic(minf);maxlen=max(map(len,d));out=[]
 for p,(title,abstract) in docs.items():
  for sec,text in [('T',title),('A',abstract)]:
   ts=tokenize(text); low=[t[0].lower() for t in ts]; candidates=[]
   for i in range(len(ts)):
    for n in range(1,min(maxlen,len(ts)-i)+1):
     vs=d.get(tuple(low[i:i+n]))
     if not vs:continue
     score=max(freq[v] for v in vs)
     if score<threshold:continue
     if n==1 and not allow_single: continue
     a,b=ts[i][1],ts[i+n-1][2]
     candidates.append((a,b,score,n))
   candidates.sort(key=lambda z:(z[0],-z[3],-z[2]));chosen=[]
   for c in candidates:
    ol=[z for z in chosen if c[0]<z[1] and c[1]>z[0]]
    if not ol:chosen.append(c)
    elif all((c[1]-c[0]>z[1]-z[0] or (c[1]-c[0]==z[1]-z[0] and c[2]>z[2])) for z in ol):chosen=[z for z in chosen if z not in ol]+[c]
   out.extend((p,sec,a,b) for a,b,*_ in chosen)
 return out
gold={(p,s,a,b) for (p,s),sp in da.items() for a,b in sp}
for minf in [1,2,3]:
 for threshold in [1,2,3,5]:
  pred=predict(dv,minf,threshold,False);print(minf,threshold,evaluate(gold,pred),flush=True)
