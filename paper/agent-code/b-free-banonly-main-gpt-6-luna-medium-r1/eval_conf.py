import collections
import pycrfsuite, ner

tr=ner.read_articles(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt')
dv=ner.read_articles(ner.DATA/'development.abstracts.txt'); da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.lexicon_from(tr,ta)
tagger=pycrfsuite.Tagger(); tagger.open('/workspace/output/train_only.crfsuite')
gold=set(); entries=[]
for pmid,sections in dv.items():
 for sec,text in sections.items():
  toks=ner.tokenize(text); x=ner.features(toks,lex); tags=tagger.tag(x)
  gold.update((pmid,sec,s,e) for s,e,*_ in da.get((pmid,sec),[]))
  probs=[tagger.marginal(t,i) for i,t in enumerate(tags)]
  for s,e in ner.get_spans(toks,tags):
   inds=[i for i,(_,a,b) in enumerate(toks) if a>=s and b<=e]
   ent=(pmid,sec,s,e)
   entries.append((ent,min(probs[i] for i in inds),sum(probs[i] for i in inds)/len(inds),probs[inds[0]],len(inds)))
print('gold/pred',len(gold),len(entries))
for field,name in ((1,'min'),(2,'mean'),(3,'start')):
 best=(0,None)
 for cut in [i/100 for i in range(20,100)]:
  pred={x[0] for x in entries if x[field]>=cut}
  tp=len(pred&gold); p=tp/max(1,len(pred)); r=tp/len(gold); f=2*p*r/(p+r)
  if f>best[0]:best=(f,(cut,p,r,tp,len(pred)))
 print(name,'best',best)
 for cut in ([.3,.4,.5,.6,.7,.8,.9] if field==2 else []):
  pred={x[0] for x in entries if x[field]>=cut}
  tp=len(pred&gold);p=tp/max(1,len(pred));r=tp/len(gold);print(' ',cut,round(2*p*r/(p+r),4),round(p,4),round(r,4),len(pred))
