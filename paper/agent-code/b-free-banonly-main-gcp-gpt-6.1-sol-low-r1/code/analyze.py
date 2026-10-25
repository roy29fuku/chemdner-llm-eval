import sys
from collections import Counter
split='development'
docs={l.split('\t')[0]:l.rstrip('\n').split('\t')[1:] for l in open('/workspace/data/'+split+'.abstracts.txt')}
gold=set()
for l in open('/workspace/data/'+split+'.annotations.txt'):
 p,s,a,b,t,c=l.rstrip('\n').split('\t');gold.add((p,s,int(a),int(b)))
pred=set()
for l in open(sys.argv[1]):
 p,span=l.strip().split('\t');s,a,b=span.split(':');pred.add((p,s,int(a),int(b)))
hit=len(pred&gold);print('P R F1',hit/len(pred),hit/len(gold),2*hit/(len(pred)+len(gold)))
for name,errors in [('FP',pred-gold),('FN',gold-pred)]:
 print(name,len(errors))
 for p,s,a,b in sorted(errors)[:150]:
  txt=docs[p][s=='A'];print(p,s,a,b,repr(txt[a:b]),repr(txt[max(0,a-45):b+45]))
