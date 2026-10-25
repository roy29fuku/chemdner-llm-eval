import os,sys,json
import ahocorasick
from collections import defaultdict,Counter
ROOT='/workspace'
def corpus(split):
 for l in open(f'{ROOT}/data/{split}.abstracts.txt'):
  p,t,a=l.rstrip('\n').split('\t')
  for s,txt in [('T',t),('A',a)]:yield p,s,txt

def gold(split):
 out=defaultdict(set)
 for l in open(f'{ROOT}/data/{split}.annotations.txt'):
  p,s,a,b,t,*_=l.rstrip('\n').split('\t');out[(p,s)].add((int(a),int(b)))
 return out

def build(splits):
 names=Counter()
 for split in splits:
  for l in open(f'{ROOT}/data/{split}.annotations.txt'):
   p,s,a,b,t,*_=l.rstrip('\n').split('\t');names[t.lower()]+=1
 auto=ahocorasick.Automaton()
 for name,count in names.items():
  if len(name)>=3:auto.add_word(name,name)
 auto.make_automaton();hit=Counter();total=Counter()
 for split in splits:
  gs=gold(split)
  for p,s,txt in corpus(split):
   for a,b,name in matches(auto,txt):
    total[name]+=1
    if (a,b) in gs[(p,s)]:hit[name]+=1
 return {name:(hit[name],total[name]) for name in names if total[name]}
def matches(auto,txt):
 for e,name in auto.iter(txt.lower()):
  b=e+1;a=b-len(name)
  if (a>0 and (txt[a-1].isalnum() or txt[a-1]=='_')) or (b<len(txt) and (txt[b].isalnum() or txt[b]=='_')):continue
  yield a,b,name
if __name__=='__main__':
 split=sys.argv[1];path=sys.argv[2];full=split=='test';names=build(['training','development'] if full else ['training']);auto=ahocorasick.Automaton()
 for name,(h,t) in names.items():auto.add_word(name,(name,h,t))
 auto.make_automaton()
 preds=defaultdict(set)
 for l in open(path):
  p,span=l.strip().split('\t');s,a,b=span.split(':');preds[(p,s)].add((int(a),int(b)))
 candidates=[]
 for p,s,txt in corpus(split):
  existing=preds[(p,s)]
  for e,(name,h,t) in auto.iter(txt.lower()):
   b=e+1;a=b-len(name)
   if (a>0 and (txt[a-1].isalnum() or txt[a-1]=='_')) or (b<len(txt) and (txt[b].isalnum() or txt[b]=='_')):continue
   if any(a<y and b>x for x,y in existing):continue
   candidates.append((p,s,a,b,name,h,t))
 if not full:
  gs=gold(split);G=sum(map(len,gs.values()));P=sum(map(len,preds.values()));H=sum(len(ps&gs[k]) for k,ps in preds.items())
  best=(2*H/(G+P),0,0)
  for support in [1,2,3,5,10]:
   for ratio in [.9,.95,.98,1]:
    adds={(p,s,a,b) for p,s,a,b,n,h,t in candidates if h>=support and h/t>=ratio};hits=sum((a,b) in gs[(p,s)] for p,s,a,b in adds);f=2*(H+hits)/(G+P+len(adds))
    print(support,ratio,'adds',len(adds),'hits',hits,'F1',f)
    if f>best[0]:best=(f,support,ratio)
  print('BEST',best)
  json.dump(best,open(ROOT+'/code/dict_params.json','w'))
  support,ratio=best[1:]
 else:_,support,ratio=json.load(open(ROOT+'/code/dict_params.json'))
 adds={(p,s,a,b) for p,s,a,b,n,h,t in candidates if support and h>=support and h/t>=ratio}
 for p,s,a,b in adds:preds[(p,s)].add((a,b))
 with open(path+'.dict','w') as f:
  for (p,s),ps in sorted(preds.items()):
   for a,b in sorted(ps):f.write(f'{p}\t{s}:{a}:{b}\n')
 print('written',path+'.dict','added',len(adds))
