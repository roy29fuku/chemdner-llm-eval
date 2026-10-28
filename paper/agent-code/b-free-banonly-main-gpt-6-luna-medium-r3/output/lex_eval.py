import sys,collections,pickle
sys.path.insert(0,'/workspace/output')
import solve
from pathlib import Path
train=solve.read_articles(solve.ROOT+'training.abstracts.txt'); trann=solve.read_anns(solve.ROOT+'training.annotations.txt')
dev=solve.read_articles(solve.ROOT+'development.abstracts.txt'); dvann=solve.read_anns(solve.ROOT+'development.annotations.txt')
lex=solve.term_lexicon(trann);gold=solve.gold_set(dev,dvann)
modelpred=set()
for ln in open('/workspace/output/dev_predictions.tsv'):
 pm,spec=ln.rstrip().split('\t'); sec,s,e=spec.split(':');modelpred.add((pm,sec,int(s),int(e)))
print('crf',len(modelpred),len(gold),len(modelpred&gold))
# Candidate matches and surfaces/freqs
cands=collections.defaultdict(set); exact_counts=collections.Counter()
trie={}
for term,n in lex.items():
 node=trie
 for w in term:node=node.setdefault(w,{})
 node[None]=(n,term)
for pm,(t,a) in dev.items():
 for sec,txt in [('T',t),('A',a)]:
  toks=solve.tokenize(txt); words=[w.lower() for w,_,_ in toks]
  for i in range(len(words)):
   node=trie;j=i
   while j<len(words) and words[j] in node:
    node=node[words[j]];j+=1
    if None in node:
     n,term=node[None];key=(pm,sec,toks[i][1],toks[j-1][2]);cands[n].add(key);exact_counts[n]+=1
for freq in [1,2,3,4,5,10]:
 cand=set().union(*(v for n,v in cands.items() if n>=freq))
 tp=len(cand&gold);p=tp/max(1,len(cand));r=tp/len(gold);f=2*p*r/(p+r)
 union=modelpred|cand;utp=len(union&gold);up=utp/len(union);ur=utp/len(gold);uf=2*up*ur/(up+ur)
 print('minfreq',freq,'lex',len(cand),tp,round(p,3),round(r,3),round(f,3),'union',len(union),utp,round(up,3),round(ur,3),round(uf,3),'newtp',len((cand-modelpred)&gold))
