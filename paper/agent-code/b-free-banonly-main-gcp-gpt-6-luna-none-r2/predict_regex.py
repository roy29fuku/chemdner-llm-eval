import re,os
from collections import Counter
forms=Counter()
for line in open('/workspace/data/training.annotations.txt',encoding='utf8'):
 p,s,a,b,m,c=line.rstrip('\n').split('\t');m=m.lower()
 if len(m)>2 and any(ch.isalpha() for ch in m): forms[m]+=1
terms=sorted(forms,key=lambda x:-len(x))
pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in terms)+r')(?![A-Za-z0-9])',re.I)
rows=[]
for line in open('/workspace/data/test.abstracts.txt',encoding='utf8'):
 pmid,title,abstract=line.rstrip('\n').split('\t')
 for sec,txt in [('T',title),('A',abstract)]:
  for m in pat.finditer(txt):rows.append(f'{pmid}\t{sec}:{m.start()}:{m.end()}')
os.makedirs('/workspace/output',exist_ok=True)
open('/workspace/output/predictions.tsv','w').write('\n'.join(rows)+'\n')
print(len(rows))
