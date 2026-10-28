from dict_tag import *
import re,collections
tr=load('training');dev=load('development'); forms=build(tr)
# Patterns joined by alternation, longest alternatives first; count thresholds
for threshold in [1,2,3,4,5,10]:
 keys=sorted((x for x,n in forms.items() if n>=threshold),key=len,reverse=True)
 pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x).replace(r'\ ',r'\s+') for x in keys)+r')(?![A-Za-z0-9])',re.I)
 tp=fp=fn=0
 for (p,s),aa in dev[1].items():
  text=dev[0][p][s];pred={(m.start(),m.end()) for m in pat.finditer(text)}; gold={(a,b) for a,b,_ in aa}
  tp+=len(pred&gold); fp+=len(pred-gold);fn+=len(gold-pred)
 print(threshold,tp,fp,fn,2*tp/(2*tp+fp+fn),flush=True)
