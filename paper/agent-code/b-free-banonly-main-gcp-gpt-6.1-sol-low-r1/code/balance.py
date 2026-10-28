import sys,re
from dictionary import corpus,gold
split,path=sys.argv[1:3];texts={(p,s):txt for p,s,txt in corpus(split)};before=set();out=set()
for l in open(path):
 p,span=l.strip().split('\t');s,a,b=span.split(':');a=int(a);b=int(b);before.add((p,s,a,b));txt=texts[p,s]
 while b>a and txt[b-1] in ',;:':b-=1
 for left,right in [('(',')'),('[',']')]:
  while b>a and txt[b-1]==right and txt[a:b].count(right)>txt[a:b].count(left):b-=1
  while b>a and txt[b-1]==left:b-=1
  if left=='(' and a<b and txt[a]==left and ')' not in txt[a:b] and re.fullmatch(r'\([A-Za-z][A-Za-z0-9+\- ]*',txt[a:b]):a+=1
  while a<b and b<len(txt) and txt[b]==right and txt[a:b].count(left)>txt[a:b].count(right):b+=1
 while a<b and txt[a].isspace():a+=1
 while b>a and txt[b-1].isspace():b-=1
 if a<b:out.add((p,s,a,b))
if split!='test':
 gs=gold(split);G={(p,s,a,b) for (p,s),sp in gs.items() for a,b in sp}
 print('before',2*len(before&G)/(len(before)+len(G)),'after',2*len(out&G)/(len(out)+len(G)),'changed',len(before^out))
with open(path+'.balanced','w') as f:
 for p,s,a,b in sorted(out):f.write(f'{p}\t{s}:{a}:{b}\n')
