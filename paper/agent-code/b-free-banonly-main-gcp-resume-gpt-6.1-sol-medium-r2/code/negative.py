import re,json
from pathlib import Path
from collections import Counter,defaultdict
from lexicon import read,metric
from postprocess import load_preds

def build_negative(splits=['training']):
 cache=Path('/workspace/code/negative_'+'_'.join(splits)+'.json')
 if cache.exists():return json.loads(cache.read_text())
 counts=Counter();positive=Counter()
 for split in splits:
  docs,gold=read(split);bysec=defaultdict(list)
  for p,s,a,b in gold:bysec[p,s].append((a,b))
  for p,s,text in docs:
   for m in re.finditer(r'(?<!\w)[A-Za-z][A-Za-z0-9]{2,}(?!\w)',text):
    term=m.group().lower();a,b=m.span();counts[term]+=1
    if any(a<y and b>x for x,y in bysec[p,s]):positive[term]+=1
 stats={t:(n,positive[t]) for t,n in counts.items()};cache.write_text(json.dumps(stats));return stats

def prune(pred,docs,stats,mincount=5,maxratio=0):
 texts={(p,s):t for p,s,t in docs};out=set()
 for p,s,a,b in pred:
  term=texts[p,s][a:b].lower();n,h=stats.get(term,(0,0))
  if n>=mincount and h/n<=maxratio:continue
  out.add((p,s,a,b))
 return out

if __name__=='__main__':
 import sys
 pred=load_preds(sys.argv[1]);docs,gold=read('development');stats=build_negative();texts={(p,s):t for p,s,t in docs}
 print('baseline',metric(pred,gold))
 for n in [2,3,5,10,20]:
  p=prune(pred,docs,stats,n);removed=pred-p;print(n,metric(p,gold),'removed',len(removed),'removed_correct',len(removed&gold));print(Counter(texts[x[0],x[1]][x[2]:x[3]] for x in removed).most_common(30))
