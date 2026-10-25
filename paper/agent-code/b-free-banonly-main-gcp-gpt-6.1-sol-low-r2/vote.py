import argparse,json
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--soft',required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--tune',action='store_true');p.add_argument('--add',type=int);p.add_argument('--keep',type=int,default=0);p.add_argument('--out');a=p.parse_args()
def read(path):
 r=set()
 for l in open(path):
  pm,x=l.rstrip().split('\t');sec,s,e=x.split(':');r.add((pm,sec,int(s),int(e)))
 return r
soft=read(a.soft);counts=Counter()
for path in a.models:counts.update(read(path))
gold=set()
if a.tune:
 for l in open('/workspace/data/development.annotations.txt'):
  pm,sec,s,e,*_=l.rstrip('\n').split('\t');gold.add((pm,sec,int(s),int(e)))
settings=[(add,keep) for add in range(2,len(a.models)+2) for keep in range(0,len(a.models))] if a.tune else [(a.add or len(a.models)+1,a.keep)]
best=-1;bp=None;bc=None
for add,keep in settings:
 pred={x for x in soft if counts[x]>=keep}|{x for x,n in counts.items() if n>=add}
 f1=2*len(pred&gold)/(len(pred)+len(gold)) if gold else 0
 added=pred-soft;removed=soft-pred
 print(add,keep,'F1',f1,'added',len(added),len(added&gold),'removed',len(removed),len(removed&gold),flush=True)
 if f1>best:best=f1;bp=pred;bc=(add,keep)
print('BEST',best,bc)
if a.out:
 with open(a.out,'w') as f:
  for pm,sec,s,e in sorted(bp):f.write(f'{pm}\t{sec}:{s}:{e}\n')
 json.dump({'add':bc[0],'keep':bc[1],'f1':best},open(a.out+'.json','w'),indent=2)
