import re,os
from collections import Counter
forms=Counter()
for line in open('/workspace/data/training.annotations.txt',encoding='utf8'):
 p,s,a,b,m,c=line.rstrip('\n').split('\t');m=m.lower()
 if len(m)>2 and any(ch.isalpha() for ch in m): forms[m]+=1
terms=sorted(forms,key=lambda x:-len(x))
# Chunk alternation pattern to keep regex compilation and evaluation bounded.
pats=[]
for j in range(0,len(terms),500):
 pats.append(re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in terms[j:j+500])+r')(?![A-Za-z0-9])',re.I))
formula=re.compile(r'(?<![A-Za-z0-9])[A-Z](?:[a-z]?)(?:\d{0,3})(?:[A-Z][a-z]?(?:\d{0,3}))*?(?:\s*\([0-9IVXivx]+[+-]?\)|[0-9]*[+-])?(?![A-Za-z0-9])')
rows=[]
for line in open('/workspace/data/test.abstracts.txt',encoding='utf8'):
 pmid,title,abstract=line.rstrip('\n').split('\t')
 for sec,txt in [('T',title),('A',abstract)]:
  spans=set()
  for pat in pats:
   spans.update((m.start(),m.end()) for m in pat.finditer(txt))
  for m in formula.finditer(txt):
   z=m.group(0)
   if (any(c.isdigit() for c in z) or '+' in z or '-' in z or z in {'C','N','O','S','P','K','F','I','V','W','U','B','H','D','T','Y','Z','X'}) and len(z)<30:spans.add((m.start(),m.end()))
  for a,b in sorted(spans):rows.append(f'{pmid}\t{sec}:{a}:{b}')
os.makedirs('/workspace/output',exist_ok=True)
open('/workspace/output/predictions.tsv','w').write('\n'.join(rows)+'\n')
print(len(rows))
