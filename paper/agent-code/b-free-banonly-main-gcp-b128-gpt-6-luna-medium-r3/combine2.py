import ner,re
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt');ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')

def evaluate(trainabs,trainann,evabs,evann,name):
 lex=ner.build_lexicon(trainann);X,Y,_,_=ner.examples(trainabs,trainann,lex);Xd,Yd,loc,_=ner.examples(evabs,evann,lex)
 crf=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0,c2=.1,max_iterations=150,all_possible_transitions=True);crf.fit(X,Y);yp=crf.predict(Xd)
 gold=set();cp=set()
 for (pm,s,t,toks),ys,pl in zip(loc,Yd,yp):
  for b,e in ner.spans_from_labels(t,toks,ys):gold.add((pm,s,b,e))
  for b,e in ner.spans_from_labels(t,toks,pl):cp.add((pm,s,b,e))
 lexp=set()
 for pm,ss in evabs.items():
  for s,t in ss.items():
   toks=ner.tokenized(t);ws=[w.casefold() for w,b,e in toks]
   for phr,freq in lex.items():
    L=len(phr)
    for i in range(len(toks)-L+1):
     obs=tuple(ws[i:i+L]);
     if obs==phr or (len(obs)==len(phr) and obs[:-1]==phr[:-1] and (obs[-1]==phr[-1]+'s' or (phr[-1].endswith(('s','x','ch','sh')) and obs[-1]==phr[-1]+'es') or (phr[-1].endswith('y') and obs[-1]==phr[-1][:-1]+'ies'))):
      lexp.add((pm,s,toks[i][1],toks[i+L-1][2]))
 suffix=('one','ene','ose','ine','ol','amide','amine','ide','ium','yl')
 sp=set()
 for pm,ss in evabs.items():
  for s,t in ss.items():
   for w,b,e in ner.tokenized(t):
    low=w.casefold()
    if any(low.endswith(x) and len(low)>len(x)+1 for x in suffix):sp.add((pm,s,b,e))
 upper=set()
 for pm,ss in evabs.items():
  for s,t in ss.items():
   for w,b,e in ner.tokenized(t):
    if w.isupper() and len(w)>=3 and len(w)<=8 and any(c.isalpha() for c in w):upper.add((pm,s,b,e))
 print('\n',name,'crf',ner.score(gold,cp),'lex',ner.score(gold,lexp),'suff',ner.score(gold,sp),'upper',ner.score(gold,upper))
 for x,pp in [('crf+s',cp|sp),('crf+lex+s',cp|lexp|sp),('crf+lex+s+u',cp|lexp|sp|upper)]:print(x,ner.score(gold,pp))
for a,b,c,d,n in [(tr,ta,dv,da,'tr->dv'),(dv,da,tr,ta,'dv->tr')]:evaluate(a,b,c,d,n)
