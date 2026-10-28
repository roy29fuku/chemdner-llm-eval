import re, collections, os
root='/workspace/data/'
counts=collections.Counter();articles={}
for split in ['training','development']:
 for line in open(root+split+'.abstracts.txt'):
  p,t,a=line.rstrip('\n').split('\t');articles[p]={'T':t,'A':a}
 for line in open(root+split+'.annotations.txt'):
  p,s,a,b,text,*_=line.rstrip('\n').split('\t')
  counts[text.lower()]+=1
terms={t for t,n in counts.items() if n>=2 and len(t)>=3}
pattern=re.compile(r'(?<![\w])(?:'+ '|'.join(re.escape(t) for t in sorted(terms,key=lambda t:(-len(t),t)))+r')(?![\w])',re.I)
occ=collections.Counter()
for doc in articles.values():
 for text in doc.values():
  occ.update(m.group().lower() for m in pattern.finditer(text))
terms={t for t in terms if counts[t]/max(occ[t],1)>=.8}
pattern=re.compile(r'(?<![\w])(?:'+ '|'.join(re.escape(t) for t in sorted(terms,key=lambda t:(-len(t),t)))+r')(?![\w])',re.I)
os.makedirs('/workspace/output',exist_ok=True)
with open('/workspace/output/predictions.tsv','w') as f:
 for line in open(root+'test.abstracts.txt'):
  p,t,a=line.rstrip('\n').split('\t')
  for sec,text in [('T',t),('A',a)]:
   for m in pattern.finditer(text):f.write(f'{p}\t{sec}:{m.start()}:{m.end()}\n')
print('Wrote fallback with',len(terms),'terms')
