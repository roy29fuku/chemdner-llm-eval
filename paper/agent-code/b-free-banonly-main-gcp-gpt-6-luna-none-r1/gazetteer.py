import re,collections
from model import *
from model import tokenize
P='/workspace/data/'
tr=load_docs(P+'training.abstracts.txt');dv=load_docs(P+'development.abstracts.txt');an=load_ann(P+'training.annotations.txt');da=load_ann(P+'development.annotations.txt')
def norm(s):return re.sub(r'\s+',' ',s.lower()).strip()
freq=collections.Counter()
for (p,sec), spans in an.items():
 txt=tr[p][0 if sec=='T' else 1]
 for a,b in spans:freq[norm(txt[a:b])]+=1
D=collections.defaultdict(set)
for v,c in freq.items():
 ts=tuple(t[0].lower() for t in tokenize(v))
 if ts and len(v)>1:D[ts].add(v)

# remove known tokens that function only as ordinary words, unless source chemical labels corroborate multi-token names
for bad in [('in',),('as',)]: D.pop(bad,None)
maxlen=max(map(len,D));print('dict',len(D),'max',maxlen,flush=True)
def predict(docs, minf=1, maxsingle=0, exact=False):
 out=[]
 for p,(title,abstract) in docs.items():
  for sec,text in [('T',title),('A',abstract)]:
   toks=tokenize(text); low=[t[0].lower() for t in toks];cand=[]
   for i in range(len(toks)):
    for n in range(1,min(maxlen,len(toks)-i)+1):
     ts=tuple(low[i:i+n])
     if ts in D:
      v=max(D[ts],key=lambda x:freq[x]);score=freq[v]
      if score < minf or (n==1 and (score<maxsingle or len(low[i])<3)): continue
      cand.append((toks[i][1],toks[i+n-1][2],score,n,v))
   # prune common non-chemical contexts via train negative neighboring contexts learned later
   # Resolve overlap in favor longer spans but allow exact substrings if mention is separately legitimate
   cand.sort(key=lambda x:(x[0],-x[3],-x[2]));keep=[]
   for c in cand:
    ov=[z for z in keep if c[0]<z[1] and c[1]>z[0]]
    if not ov:keep.append(c)
    elif all((c[1]-c[0])>(z[1]-z[0]) for z in ov):keep=[z for z in keep if z not in ov]+[c]
   out.extend((p,sec,a,b) for a,b,*_ in keep)
 return out
if __name__=='__main__':
 gold={(p,s,a,b) for (p,s),sp in da.items() for a,b in sp}
 for threshold in [1]:
  pred=predict(dv,threshold);print(threshold,evaluate(gold,pred),len(pred),flush=True)
