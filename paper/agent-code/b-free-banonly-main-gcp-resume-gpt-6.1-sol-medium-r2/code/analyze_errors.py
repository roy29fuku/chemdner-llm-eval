from collections import defaultdict,Counter
import sys
from lexicon import read,metric
from postprocess import load_preds
pred=load_preds(sys.argv[1]); docs,gold=read('development');texts={(p,s):t for p,s,t in docs};pp=defaultdict(list);gg=defaultdict(list)
for p,s,a,b in pred:pp[p,s].append((a,b))
for p,s,a,b in gold:gg[p,s].append((a,b))
counts=Counter()
for p,s,a,b in gold-pred:
 counts['boundary_FN' if any(a<y and b>x for x,y in pp[p,s]) else 'missed_FN']+=1
for p,s,a,b in pred-gold:
 counts['boundary_FP' if any(a<y and b>x for x,y in gg[p,s]) else 'spurious_FP']+=1
print('metrics',metric(pred,gold));print(counts)
classes=defaultdict(Counter);miss=defaultdict(Counter)
for line in open('/workspace/data/development.annotations.txt'):
 p,s,a,b,t,c=line.rstrip('\n').split('\t');span=(p,s,int(a),int(b));classes[c]['total']+=1
 if span in pred:classes[c]['correct']+=1
 else:miss[c][t]+=1
for c,n in classes.items():print(c,n,'recall',round(n['correct']/n['total'],4),'common misses',miss[c].most_common(12))
