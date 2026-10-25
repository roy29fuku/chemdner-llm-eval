import ner,re,collections
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt');ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.build_lexicon(ta);X,Y,_,_=ner.examples(tr,ta,lex);Xd,Yd,loc,_=ner.examples(dv,da,lex)
crf=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0,c2=.1,max_iterations=150,all_possible_transitions=True);crf.fit(X,Y);yp=crf.predict(Xd)
gold=set();cp=set()
for (pm,s,t,toks),ys,pl in zip(loc,Yd,yp):
 for b,e in ner.spans_from_labels(t,toks,ys):gold.add((pm,s,b,e))
 for b,e in ner.spans_from_labels(t,toks,pl):cp.add((pm,s,b,e))

def lp(variants=False):
 out=set()
 for pm,ss in dv.items():
  for s,t in ss.items():
   toks=ner.tokenized(t);ws=[w.casefold() for w,b,e in toks]
   for phr,freq in lex.items():
    L=len(phr)
    if L>len(toks):continue
    for i in range(len(toks)-L+1):
     obs=tuple(ws[i:i+L]); ok=obs==phr
     if variants and not ok:
      # Final-token plural morphology; accept only common regular inflections.
      root=obs[:-1]; last=obs[-1]; base=phr[-1]
      ok=(root==phr[:-1] and ((last==base+'s') or (base.endswith(('s','x','ch','sh')) and last==base+'es') or (base.endswith('y') and last==base[:-1]+'ies')))
     if ok:out.add((pm,s,toks[i][1],toks[i+L-1][2]))
 return out
lexp=lp();lexvar=lp(True)
print('crf',ner.score(gold,cp),'lex',ner.score(gold,lexp),'var',ner.score(gold,lexvar))
# thresholds list for one-token suffixes
def suffp(suffs):
 out=set()
 for pm,ss in dv.items():
  for s,t in ss.items():
   for w,b,e in ner.tokenized(t):
    low=w.casefold()
    if any(low.endswith(suf) and len(low)>len(suf)+1 for suf in suffs):out.add((pm,s,b,e))
 return out
for suffs in [('one','ene','ose'),('one','ene','ose','ine','ol'),('one','ene','ose','ine','ol','amide','amine'),('one','ene','ose','ine','ol','amide','amine','ide','ium'),('one','ene','ose','ine','ol','amide','amine','ide','ium','yl')]:
 su=suffp(suffs); print('suffix',suffs,ner.score(gold,su))
 for name,other in [('crf',cp),('lex',lexp),('lexvar',lexvar),('crf+lex',cp|lexp),('crf+lexvar',cp|lexvar)]:
  # add suffix but prioritize exact matching. If overlaps conflicting, set union exact.
  print('  ',name,'+suf',ner.score(gold,other|su))
