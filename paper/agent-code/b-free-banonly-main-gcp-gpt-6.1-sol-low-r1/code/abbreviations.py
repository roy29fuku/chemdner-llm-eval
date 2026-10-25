import sys,re
from collections import defaultdict
from dictionary import corpus,gold
split,path=sys.argv[1:3];pred=defaultdict(set)
for l in open(path):
 p,span=l.strip().split('\t');s,a,b=span.split(':');pred[(p,s)].add((int(a),int(b)))
docs=list(corpus(split));bydoc=defaultdict(list)
for p,s,txt in docs:bydoc[p].append((s,txt))
adds=set()
for p,parts in bydoc.items():
 abbrevs=set()
 for s,txt in parts:
  for a,b in pred[(p,s)]:
   m=re.match(r'\s*\(([A-Za-z][A-Za-z0-9+−-]{2,19})\)',txt[b:])
   if not m:continue
   short=m[1];long=txt[a:b]
   if len(long)<=len(short) or sum(c.isalpha() for c in short)<2:continue
   letters=''.join(c.lower() for c in short if c.isalpha())
   large=''.join(c.lower() for c in long if c.isalpha())
   it=iter(large)
   if not all(any(c==d for d in it) for c in letters):continue
   abbrevs.add(short)
 for s,txt in parts:
  for name in abbrevs:
   for m in re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',txt):
    a,b=m.span()
    if not any(a<y and b>x for x,y in pred[(p,s)]):adds.add((p,s,a,b))
if split!='test':
 gs=gold(split);P=sum(map(len,pred.values()));G=sum(map(len,gs.values()));H=sum(len(v&gs[k]) for k,v in pred.items());hit=sum((a,b) in gs[(p,s)] for p,s,a,b in adds)
 print('before',2*H/(P+G),'after',2*(H+hit)/(P+G+len(adds)),'added',len(adds),'correct',hit)
for p,s,a,b in adds:pred[(p,s)].add((a,b))
with open(path+'.abbr','w') as f:
 for (p,s),spans in sorted(pred.items()):
  for a,b in sorted(spans):f.write(f'{p}\t{s}:{a}:{b}\n')
