from model_fast import *
import re,pickle,os
from collections import Counter
tr=read_labels('training'); forms=Counter(m.lower() for aa in tr.values() for a,b,c,m in aa if len(m)>2 and any(ch.isalpha() for ch in m))
terms=sorted(forms,key=lambda x:-len(x))
pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in terms)+r')(?![A-Za-z0-9])',re.I)
# Rule-based general chemical patterns complement the training mention dictionary, filtered by string and context.
# Keep only formula-like terms/element symbols where these are strong and exact boundaries.
formula=re.compile(r'(?<![A-Za-z0-9])[A-Z](?:[a-z]?)(?:\d{0,3})(?:[A-Z][a-z]?(?:\d{0,3}))*?(?:\s*\([0-9IVXivx]+[+-]?\)|[0-9]*[+-])?(?![A-Za-z0-9])')
# Explicit high-frequency chemical nouns from CHEBI vocabulary are excluded; instead use trained terms.
rows=[]
with open('/workspace/data/test.abstracts.txt',encoding='utf8') as f:
 for line in f:
  pmid,title,abstract=line.rstrip('\n').split('\t')
  for sec,txt in [('T',title),('A',abstract)]:
   spans={(m.start(),m.end()) for m in pat.finditer(txt)}
   # formula recognizer only for strongly chemical string forms, training occurrences govern exact mention forms.
   for m in formula.finditer(txt):
    z=m.group(0)
    if (z.lower(),) in []: pass
    # accept compact mixed-case formulas containing digit, charge, common symbols in trained formula label set
    if (any(c.isdigit() for c in z) or '+' in z or '-' in z or z in {'C','N','O','S','P','K','F','I','V','W','U','B','H','D','T','Y','Z','X'}) and len(z)<30:
     spans.add((m.start(),m.end()))
   for a,b in sorted(spans): rows.append(f'{pmid}\t{sec}:{a}:{b}')
os.makedirs('/workspace/output',exist_ok=True)
open('/workspace/output/predictions.tsv','w',encoding='utf8').write('\n'.join(rows)+'\n')
print('wrote',len(rows))
