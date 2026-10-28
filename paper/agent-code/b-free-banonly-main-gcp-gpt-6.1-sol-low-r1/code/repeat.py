import sys,re
from collections import defaultdict,Counter
from dictionary import corpus,gold,build
split,path=sys.argv[1:3];pred=defaultdict(set)
for l in open(path):
 p,span=l.strip().split('\t');s,a,b=span.split(':');pred[(p,s)].add((int(a),int(b)))
docs=list(corpus(split));bydoc=defaultdict(list)
for p,s,txt in docs:bydoc[p].append((s,txt))
stats=build(['training','development'] if split=='test' else ['training'])
adds=set()
for p,parts in bydoc.items():
 names=Counter()
 for s,txt in parts:
  for a,b in pred[(p,s)]:
   name=txt[a:b]
   h,t=stats.get(name.lower(),(0,1))
   if len(name)>=3 and h>=1 and h/t>=.8:names[name]+=1
 for s,txt in parts:
  occupied=pred[(p,s)].copy()
  for name,c in sorted(names.items(),key=lambda x:-len(x[0])):
   for m in re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',txt,0 if name.isupper() else re.I):
    a,b=m.span()
    if not any(a<y and b>x for x,y in occupied):adds.add((p,s,a,b));occupied.add((a,b))
if split!='test':
 gs=gold(split);P=sum(map(len,pred.values()));G=sum(map(len,gs.values()));H=sum(len(v&gs[k]) for k,v in pred.items());hit=sum((a,b) in gs[(p,s)] for p,s,a,b in adds)
 print('before',2*H/(P+G),'after',2*(H+hit)/(P+G+len(adds)),'added',len(adds),'correct',hit)
for p,s,a,b in adds:pred[(p,s)].add((a,b))
with open(path+'.repeat','w') as f:
 for (p,s),spans in sorted(pred.items()):
  for a,b in sorted(spans):f.write(f'{p}\t{s}:{a}:{b}\n')
