import sys,re
from collections import defaultdict,Counter
from dictionary import corpus,gold
split,path=sys.argv[1:3];parts=['training','development'] if split=='test' else ['training'];names=Counter();anns={}
for part in parts:
 gs=gold(part)
 for k,v in gs.items():anns[(part,*k)]=v
 for l in open('/workspace/data/'+part+'.annotations.txt'):
  p,s,a,b,t,c=l.rstrip('\n').split('\t')
  if re.fullmatch('[A-Z][a-z]',t):names[t]+=1
positive=Counter();total=Counter();pattern=re.compile(r'(?<![\w-])([A-Z][a-z])(?![\w(+−-])')
for part in parts:
 for p,s,txt in corpus(part):
  for m in pattern.finditer(txt):
   name=m[1]
   if name not in names:continue
   total[name]+=1;positive[name]+=m.span() in anns.get((part,p,s),set())
valid={n for n in names if positive[n]>=5 and positive[n]/max(total[n],1)>=.95};print('reliable symbols',sorted((n,positive[n],total[n]) for n in valid))
pred=defaultdict(set)
for l in open(path):
 p,sp=l.strip().split('\t');s,a,b=sp.split(':');pred[p,s].add((int(a),int(b)))
adds=set()
for p,s,txt in corpus(split):
 for m in pattern.finditer(txt):
  a,b=m.span()
  if m[1] in valid and not any(a<y and b>x for x,y in pred[p,s]):adds.add((p,s,a,b))
if split!='test':
 gs=gold(split);P=sum(map(len,pred.values()));G=sum(map(len,gs.values()));H=sum(len(v&gs[k]) for k,v in pred.items());hit=sum((a,b) in gs[p,s] for p,s,a,b in adds);print('before',2*H/(P+G),'after',2*(H+hit)/(P+G+len(adds)),'added',len(adds),'correct',hit)
for p,s,a,b in adds:pred[p,s].add((a,b))
with open(path+'.atoms','w') as f:
 for (p,s),spans in sorted(pred.items()):
  for a,b in sorted(spans):f.write(f'{p}\t{s}:{a}:{b}\n')
