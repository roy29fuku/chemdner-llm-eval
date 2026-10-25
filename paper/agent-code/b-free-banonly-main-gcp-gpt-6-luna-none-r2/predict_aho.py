import ahocorasick,re,os
from collections import Counter
forms=Counter()
for line in open('/workspace/data/training.annotations.txt',encoding='utf8'):
 p,s,a,b,m,c=line.rstrip('\n').split('\t');m=m.lower()
 if len(m)>2 and any(ch.isalpha() for ch in m): forms[m]+=1
A=ahocorasick.Automaton()
for m,n in forms.items():A.add_word(m,(m,n))
A.make_automaton()
rows=[]
for line in open('/workspace/data/test.abstracts.txt',encoding='utf8'):
 pmid,title,abstract=line.rstrip('\n').split('\t')
 for sec,txt in [('T',title),('A',abstract)]:
  lo=txt.lower()
  for end,(m,n) in A.iter(lo):
   start=end-len(m)+1; stop=end+1
   if (start==0 or not txt[start-1].isalnum()) and (stop==len(txt) or not txt[stop].isalnum()): rows.append((pmid,sec,start,stop))
# de-dupe exact hits
rows=sorted(set(rows),key=lambda x:(int(x[0]),x[1],x[2],x[3]))
os.makedirs('/workspace/output',exist_ok=True)
with open('/workspace/output/predictions.tsv','w') as f:
 for p,s,a,b in rows:f.write(f'{p}\t{s}:{a}:{b}\n')
print(len(rows))
