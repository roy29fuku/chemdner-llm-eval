import re,collections,math
from model import load_docs,load_ann,evaluate,P
from collections import Counter,defaultdict

def norm(s): return re.sub(r'\s+',' ',s.lower()).strip()
tr=load_docs(P+'training.abstracts.txt');dev=load_docs(P+'development.abstracts.txt'); ann=load_ann(P+'training.annotations.txt'); da=load_ann(P+'development.annotations.txt')
# extract labeled surfaces by offsets, count exact normalized variants
freq=Counter(); label=Counter()
for (pmid,sec),spans in ann.items():
 text=tr[pmid][0 if sec=='T' else 1]
 for a,b in spans:
  v=text[a:b];freq[norm(v)]+=1
# candidate boundary regex broad, punctuation included only likely name tokens
pat=re.compile(r"(?<![A-Za-z0-9])(?:\(?[A-Za-z0-9α-ωΑ-ΩµμΔδ]+(?:[-+./][A-Za-z0-9α-ωΑ-ΩµμΔδ]+|\([A-Za-z0-9+−-]+\))*)(?:\s+(?:[A-Za-z][A-Za-z0-9-]*|[0-9]+(?:\.[0-9]+)?)){0,4}",re.I)
# Longest matching substrings from sentence/token boundaries via token ngrams exact dictionary.
# Include only dictionary forms frequencies and build per document prediction from exact substring matching.
gold={(p,s,a,b) for (p,s),spans in da.items() for a,b in spans}
for minfreq in [1,2,3,4,5]:
 dic={v for v,c in freq.items() if c>=minfreq and len(v)>1}
 pred=[]
 # whole word matching, prefer longest nonoverlapping matching sequence, but preserve nested names if distinct adjacent
 for pmid,(title,abstract) in dev.items():
  for sec,text in [('T',title),('A',abstract)]:
   low=text.lower()
   cand=[]
   for v in dic:
    # scanning occurrences exact normalized; whitespace-flexible
    rex=re.compile(r'(?<![\w])'+r'\s+'.join(map(re.escape,v.split()))+r'(?![\w])',re.I)
    for m in rex.finditer(text): cand.append((m.start(),m.end(),freq[v]))
   # Resolve overlapping: only longest for same overlapping region
   cand.sort(key=lambda x:(x[0],-(x[1]-x[0]),-x[2]))
   keep=[]
   for c in cand:
    if any(c[0]<z[1] and c[1]>z[0] for z in keep):
     # replace if strictly longer and exact same boundary-ish
     overlaps=[z for z in keep if c[0]<z[1] and c[1]>z[0]]
     if all((c[1]-c[0])>(z[1]-z[0]) for z in overlaps):
      keep=[z for z in keep if z not in overlaps];keep.append(c)
    else:keep.append(c)
   pred.extend((pmid,sec,a,b) for a,b,_ in keep)
 print(minfreq,evaluate(gold,pred),flush=True)
