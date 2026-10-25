import ner,re,collections
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')

def goldset(ann): return {(p,s,b,e) for p,ss in ann.items() for s,sp in ss.items() for b,e,*_ in sp}
def build(abs, ann):
 out=set(); counts=collections.Counter()
 for pm,ss in abs.items():
  for sec,text in ss.items():
   toks=ner.tokenized(text)
   for w,b,e in toks:
    lw=w.casefold(); alpha=''.join(x for x in lw if x.isalpha()); p=None
    if re.search(r'\d',w) and re.search('[A-Z]',w) and (len(w)>1):p='digit'
    if w.isupper() and 2<=len(w)<=8 and any(c.isalpha() for c in w):p='upper'
    if alpha.endswith(('ol','one','ide','ate','ine','ium','ose','yl','oxy','azole','amide','acid','ester','itol','amine','phenyl','benzene','pyridine','quinone','terpene','sterol','lipid','saccharide','alkaloid','flavonoid','polyphenol','hydrocarbon','nucleotide','peptide','hydrocarbon','phenol','oxide','alcohol','ketone','ether','amine','nitrile','thiol','polymer','triglyceride','phosphate','glucoside','sulfate','sulphate')):p='suffix'
    if re.match(r'^(?:[A-Z]?\d+|\d+[A-Z])',w) or '-' in w and any(c.isdigit() for c in w):p='systematic'
    if p:out.add((pm,sec,b,e));counts[p]+=1
 print('counts',counts)
 print('scores each union',ner.score(goldset(ann),out))
 for p in counts:
  pp=set()
  for pm,ss in abs.items():
   for sec,text in ss.items():
    for w,b,e in ner.tokenized(text):
     lw=w.casefold();alpha=''.join(x for x in lw if x.isalpha()); ok=False
     if p=='digit':ok=bool(re.search(r'\d',w) and re.search('[A-Z]',w) and len(w)>1)
     if p=='upper':ok=w.isupper() and 2<=len(w)<=8 and any(c.isalpha() for c in w)
     if p=='suffix':ok=alpha.endswith(('ol','one','ide','ate','ine','ium','ose','yl','oxy','azole','amide','acid','ester','itol','amine','phenyl','benzene','pyridine','quinone','terpene','sterol','lipid','saccharide','alkaloid','flavonoid','polyphenol','hydrocarbon','nucleotide','peptide','phenol','oxide','alcohol','ketone','ether','nitrile','thiol','polymer','triglyceride','phosphate','glucoside','sulfate','sulphate'))
     if p=='systematic':ok=bool(re.match(r'^(?:[A-Z]?\d+|\d+[A-Z])',w) or '-' in w and any(c.isdigit() for c in w))
     if ok: pp.add((pm,sec,b,e))
  print(p,ner.score(goldset(ann),pp))
 return out
for a,b,name in [(dv,da,'dev'),(tr,ta,'train')]:
 print('\n',name);build(a,b)
