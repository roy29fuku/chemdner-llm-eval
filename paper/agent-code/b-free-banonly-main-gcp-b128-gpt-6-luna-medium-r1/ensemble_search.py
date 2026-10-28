exec(open('/workspace/ensemble_eval.py').read().split("print('dict'")[0])
# all thresholds and span-level combinations
with open('/workspace/dev_bert_probs.pkl','rb') as f:(rows,probs)=pickle.load(f)
items=[(p,s,t,da.get(p,{}).get(s,[]))for p,s,t in rows]
from transformers import AutoTokenizer
import bert_ner
tok=AutoTokenizer.from_pretrained('distilbert/distilbert-base-cased',use_fast=True)
B={pair:bert_ner.decode_items(items,probs,tok,*pair) for pair in [(1,1),(.75,.75),(.5,.5),(.4,.4),(.3,.3),(.2,.2),(.1,.1),(.5,.25),(.25,.5),(.25,.1),(.4,.1),(.5,.1),(.75,.25)]}
gold={(p,s,a,b)for p,ss in da.items()for s,aa in ss.items()for a,b,_,_ in aa}
sets={}
for th in [.02,.03,.04,.05,.06,.08,.1,.15,.2]:sets[th]={(p,s,c[2],c[3])for(p,s,t,toks,c),x in zip(dm,pr)if x>=th}
for th,d in sets.items():
 print('DICT',th,ner.score(d,gold))
for pair,b in B.items():
 print('BERT',pair,ner.score(b,gold),'union dict@.04',ner.score(b|sets[.04],gold),'intersect',ner.score(b&sets[.04],gold),'union crf',ner.score(b|sets[.04]|crfpred,gold),'dict intersect crf',ner.score(sets[.04]&(crfpred|b),gold))
# best union/intersection candidate thresholds
best=[]
for pair,b in B.items():
 for th,d in sets.items():
  for name,z in [('union',b|d),('intersection',b&d),('union+crf',b|d|crfpred),('union+direct',b|d|direct),('agree2', (b&d)|crfpred),('agree2+direct',(b&d)|direct|crfpred)]:
   sc=ner.score(z,gold);best.append((sc[2],pair,th,name,sc))
for x in sorted(best,reverse=True)[:30]:print('BEST',x)
