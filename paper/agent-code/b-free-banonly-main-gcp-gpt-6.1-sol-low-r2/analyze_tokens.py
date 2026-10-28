from transformers import AutoTokenizer
from collections import Counter
T=AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
ann={}
for line in open('/workspace/data/training.annotations.txt'):
 pm,sec,s,e,txt,cls=line.rstrip('\n').split('\t');ann.setdefault((pm,sec),[]).append((int(s),int(e),txt))
bad=[];n=0
for line in open('/workspace/data/training.abstracts.txt'):
 pm,t,a=line.rstrip('\n').split('\t')
 for sec,text in [('T',t),('A',a)]:
  off=T(text,return_offsets_mapping=True,add_special_tokens=False)['offset_mapping']; ss={s for s,e in off};ee={e for s,e in off}
  for s,e,txt in ann.get((pm,sec),[]):
   n+=1
   if s not in ss or e not in ee:bad.append((txt,s not in ss,e not in ee))
print(n,len(bad),Counter((s,e) for t,s,e in bad));print(bad[:100])
