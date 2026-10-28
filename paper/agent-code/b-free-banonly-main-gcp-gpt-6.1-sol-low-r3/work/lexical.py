import ahocorasick,json,os
from collections import Counter,defaultdict
from post import valid

def lower(text):return ''.join(c.lower() if len(c.lower())==1 else c for c in text)
def build(root,splits):
 counts=Counter();raw={}
 for split in splits:
  for l in open(root+'/'+split+'.annotations.txt'):
   p,s,a,b,name,*_=l.rstrip('\n').split('\t')
   if valid(name) and len(name)>=3:
    key=lower(name);counts[key]+=1;raw[key]=name
 A=ahocorasick.Automaton()
 for name,n in counts.items():
  if n>=2:A.add_word(name,name)
 A.make_automaton();denom=Counter()
 for split in splits:
  for l in open(root+'/'+split+'.abstracts.txt'):
   p,*texts=l.rstrip('\n').split('\t')
   for text in texts:
    for end,name in A.iter(lower(text)):
     a=end+1-len(name);b=end+1
     if (a and text[a-1].isalnum()) or (b<len(text) and text[b].isalnum()):continue
     denom[name]+=1
 out={name:dict(pos=n,total=denom[name],ratio=n/max(1,denom[name])) for name,n in counts.items() if n>=2}
 return out

def make_automaton(stats,ratio=.98,mincount=5):
 A=ahocorasick.Automaton()
 for name,v in stats.items():
  if v['ratio']>=ratio and v['pos']>=mincount:A.add_word(name,name)
 A.make_automaton();return A

def apply(pred,docs,probs,offs,A,confidence=.1):
 import numpy as np
 out=set(pred);bydoc=defaultdict(list)
 for p,s,a,b in pred:bydoc[p,s].append((a,b))
 for (p,s,text),pr,off in zip(docs,probs,offs):
  spans=bydoc[p,s];oa=np.asarray(off)
  if not len(oa):continue
  matches=[]
  for end,name in A.iter(lower(text)):
   a=end+1-len(name);b=end+1
   if (a and text[a-1].isalnum()) or (b<len(text) and text[b].isalnum()):continue
   if not valid(text[a:b]):continue
   if any(a<y and b>x for x,y in spans):continue
   left=np.searchsorted(oa[:,1],a,side='right');right=np.searchsorted(oa[:,0],b,side='left')
   if right<=left:continue
   conf=float((1-pr[left:right,0]).mean())
   if conf>=confidence:matches.append((b-a,a,b))
  for _,a,b in sorted(matches,reverse=True):
   if not any(a<y and b>x for x,y in spans):out.add((p,s,a,b));spans.append((a,b))
 return out
if __name__=='__main__':
 import sys
 json.dump(build('/workspace/data',sys.argv[1:] or ['training']),open('/workspace/work/lex_stats.json','w'))
