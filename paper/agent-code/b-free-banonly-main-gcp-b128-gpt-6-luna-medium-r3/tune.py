import ner, itertools
from pathlib import Path
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.build_lexicon(ta);X,Y,_,_=ner.examples(tr,ta,lex);Xd,Yd,loc,_=ner.examples(dv,da,lex)
gold=set()
for (pm,s,t,toks),ys in zip(loc,Yd):
 for b,e in ner.spans_from_labels(t,toks,ys):gold.add((pm,s,b,e))
for c1,c2 in [(0,0.01),(0,.1),(.01,.01),(.01,.1),(.1,.1),(0,1),(.1,1),(1,1)]:
 crf=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=c1,c2=c2,max_iterations=150,all_possible_transitions=True)
 crf.fit(X,Y)
 m=crf.predict_marginals(Xd)
 for tb,ti in [(.3,.3),(.4,.3),(.5,.4),(.6,.5),(.2,.2),(.15,.15),(.4,.2),(.25,.4)]:
  p=set();n=0
  for (pm,s,t,toks),row in zip(loc,m):
   labels=[]
   for d in row:
    labels.append('B' if d.get('B',0)>=tb else 'I' if d.get('I',0)>=ti else 'O')
   for b,e in ner.spans_from_labels(t,toks,labels):p.add((pm,s,b,e))
  print((c1,c2,tb,ti),ner.score(gold,p)[:3],len(p))
