from gazetteer import *
from collections import Counter,defaultdict
# create prediction and errors, summarize strings of FPs/FNs
pred=predict(dv);gold={(p,s,a,b) for (p,s),sp in da.items() for a,b in sp};fp=Counter();fn=Counter();
for p,s,a,b in set(pred)-gold:
 text=dv[p][0 if s=='T' else 1];fp[text[a:b].lower()]+=1
for p,s,a,b in gold-set(pred):
 text=dv[p][0 if s=='T' else 1];fn[text[a:b].lower()]+=1
print('FP top',fp.most_common(80));print('FN top',fn.most_common(80))
