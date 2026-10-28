exec(open('/workspace/eval_crf.py').read().split("allpred=[]")[0])
from collections import Counter
fc=Counter();tc=Counter(); examples=collections.defaultdict(list)
for (p,s),aa in dev[1].items():
 text=dev[0][p][s];pred=spans_probs(text,forms,m);gold={(a,b,t.lower()) for a,b,t in aa};gpos={(a,b) for a,b,t in aa}
 for a,b,c in pred:
  if (a,b) not in gpos:
   x=text[a:b].lower();fc[x]+=1;examples[x].append((text[max(0,a-50):min(len(text),b+50)],s))
for x,n in fc.most_common(60):
 print(n,repr(x),examples[x][0])
