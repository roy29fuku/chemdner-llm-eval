import ner,abbr
for ab,an,name in [(ner.read_abstracts(ner.DATA/'training.abstracts.txt'),ner.read_annotations(ner.DATA/'training.annotations.txt'),'train'),(ner.read_abstracts(ner.DATA/'development.abstracts.txt'),ner.read_annotations(ner.DATA/'development.annotations.txt'),'dev')]:
 g={(pm,s,b,e) for pm,ss in an.items() for s,sp in ss.items() for b,e,*_ in sp};p=set();defs=0
 for pm,ss in ab.items():
  for s,t in ss.items():
   pp,dd=abbr.abbreviation_candidates(t); defs+=len(dd)
   for b,e in pp:p.add((pm,s,b,e))
 print(name,'definitions',defs,'abbr spans',len(p),'gold',sum(1 for x in g if x[2:]),'score',ner.score(g,p))
 print('examples')
 n=0
 for pm,s,b,e in sorted(p&g):
  print(pm,s,repr(ab[pm][s][b:e]));n+=1
  if n>20:break
 print('missed gold all caps')
 n=0
 for pm,s,b,e in sorted(g-p):
  t=ab[pm][s][b:e]
  if t.upper()==t and len(t)>=3:
   print(pm,s,repr(t));n+=1
   if n>20:break
