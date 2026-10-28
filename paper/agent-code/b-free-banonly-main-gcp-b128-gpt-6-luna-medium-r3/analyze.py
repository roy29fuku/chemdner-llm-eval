import ner,collections
from pathlib import Path
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt')
dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt'); da=ner.read_annotations(ner.DATA/'development.annotations.txt')
def goldset(abs,ann): return {(pm,s,b,e) for pm,sec in ann.items() for s,spans in sec.items() for b,e,*_ in spans}
def lexpred(abs,lex):
 out=set()
 for pm,sec in abs.items():
  for s,txt in sec.items():
   toks=ner.tokenized(txt); ws=[w.casefold() for w,b,e in toks]
   for phrase,freq in lex.items():
    L=len(phrase)
    for i in range(len(toks)-L+1):
     if tuple(ws[i:i+L])==phrase: out.add((pm,s,toks[i][1],toks[i+L-1][2]))
 return out
for source_abs,source_ann,target_abs,target_ann,name in [(tr,ta,dv,da,'tr->dv'),(dv,da,tr,ta,'dv->tr')]:
 lex=ner.build_lexicon(source_ann); pred=lexpred(target_abs,lex); g=goldset(target_abs,target_ann)
 print(name,'lexsize',len(lex),'score',ner.score(g,pred))
 # length specific
 for n in [1,2,3,5]:
  ll={k:v for k,v in lex.items() if v>=n}; pp=lexpred(target_abs,ll);print('freq>=',n,ner.score(g,pp))
