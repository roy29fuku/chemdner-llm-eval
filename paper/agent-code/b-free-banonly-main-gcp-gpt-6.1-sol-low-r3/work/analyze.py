import numpy as np,sys,os,json
from transformers import AutoTokenizer
from collections import defaultdict,Counter
T=AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
g=defaultdict(list);docs=[];offs=[]
for l in open('/workspace/data/development.annotations.txt'):
 p,s,a,b,*_=l.rstrip('\n').split('\t');g[(p,s)].append((int(a),int(b)))
for l in open('/workspace/data/development.abstracts.txt'):
 p,t,a=l.rstrip('\n').split('\t')
 for s,txt in [('T',t),('A',a)]:
  docs.append((p,s,txt));offs.append(T(txt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'])
gold={(p,s,a,b) for (p,s),sp in g.items() for a,b in sp};text={(p,s):t for p,s,t in docs}
prs=[]
for path in sys.argv[1:]:
 f=np.load(path);prs.append([f['arr_'+str(i)] for i in range(len(docs))])
probs=[sum(x)/len(prs) for x in zip(*prs)]
def decode(scale=1):
 pred=set()
 for pr,(p,s,t),off in zip(probs,docs,offs):
  active=None;pr=pr.copy();pr[:,0]*=scale
  for lab,(a,b) in zip(pr.argmax(-1),off):
   if lab in (0,1):
    if active:pred.add((p,s,*active));active=None
   if lab in (1,2):
    if active is None:active=[a,b]
    else:active[1]=b
  if active:pred.add((p,s,*active))
 return pred
for x in [.6,.8,1,1.2,1.4,1.6,2,2.5,3]:
 pred=decode(x);tp=len(pred&gold);print(x,'F1',2*tp/(len(pred)+len(gold)),'P',tp/len(pred),'R',tp/len(gold),flush=True)
pred=decode();fp=pred-gold;fn=gold-pred
for name,spans in [('FP',fp),('FN',fn)]:
 print(name,Counter(text[(p,s)][a:b] for p,s,a,b in spans).most_common(50))
 with open('/workspace/work/'+name+'.txt','w') as f:
  for p,s,a,b in sorted(spans):
   t=text[(p,s)];f.write(f'{p}\t{s}\t{t[a:b]}\t{t[max(0,a-50):b+50]}\n')
