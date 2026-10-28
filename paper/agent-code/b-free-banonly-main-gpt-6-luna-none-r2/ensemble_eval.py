exec(open('/workspace/eval_crf.py').read().split("allpred=[]")[0])
import re,collections
# dictionary matching evaluated with support cutoffs; suppress short abbreviations that are non-chemistry overwhelming
short_bad={'as','p','b','d','i','w','ir','re','f','ga','cs','rb'}
def dictpred(text, threshold=4):
 keys=[x for x,n in forms.items() if n>=threshold and x not in short_bad]
 # skip short one char except element symbols with high observed precision
 allowed={'c','n','o','s','h'}
 keys=[x for x in keys if len(x)>1 or x in allowed]
 keys=sorted(keys,key=len,reverse=True)
 pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x).replace(r'\ ',r'\s+') for x in keys)+r')(?![A-Za-z0-9])',re.I)
 return {(m.start(),m.end()) for m in pat.finditer(text)}
TP=FP=FN=0; modes=collections.Counter()
for (p,s),aa in dev[1].items():
 text=dev[0][p][s];pr={ (a,b) for a,b,c in spans_probs(text,forms,m) }; dp=dictpred(text); gold={(a,b) for a,b,_ in aa}
 for mode in ['union','dict_only_novel','replace_lowconf']:
  pass
 pr2=pr|dp
 TP+=len(pr2&gold);FP+=len(pr2-gold);FN+=len(gold-pr2)
print('union',TP,FP,FN,2*TP/(2*TP+FP+FN),flush=True)
# Dict matches not model exact recall and precision
TP=FP=0
for (p,s),aa in dev[1].items():
 tx=dev[0][p][s];dp=dictpred(tx); gold={(a,b) for a,b,_ in aa}; TP+=len(dp&gold); FP+=len(dp-gold)
print('dict',TP,FP,flush=True)
