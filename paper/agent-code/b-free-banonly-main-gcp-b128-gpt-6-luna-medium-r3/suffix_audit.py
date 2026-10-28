import ner,collections,re
tr=ner.read_abstracts(ner.DATA/'training.abstracts.txt');ta=ner.read_annotations(ner.DATA/'training.annotations.txt');dv=ner.read_abstracts(ner.DATA/'development.abstracts.txt');da=ner.read_annotations(ner.DATA/'development.annotations.txt')
S=('ine','ol','one','ium','ide','ate','ose','ane','ene','yne','yl','oxy','azole','amide','acid','ester','itol','amine','phenyl','benzene','pyridine','purine','pyrimidine','quinone','terpene','sterol','lipid','saccharide','alkaloid','flavonoid','polyphenol','hydrocarbon','nucleotide','peptide','phenol','oxide','alcohol','ketone','ether','nitrile','thiol','polymer','triglyceride','phosphate','glucoside','sulfate','sulphate','olide','imine','amine','anone','anilide','benzyl','benzoate','chromone','coumarin','carboxyl','naphthalene','indole','imidazole','benzimidazole','pyrrole','piperidine','piperazine','morpholine','steroid','terpenoid','aldehyde','isocyanate','isothiocyanate','cyanide','chloride','bromide','iodide','fluoride','hydroxide','carbonate','sulfonate','sulphide','sulfide','ether','amine','diol','triol','glutamate','aspartate','asparagine','serine','glutamine','alanine','glycine','tyrosine','methionine','tryptophan','lysine','arginine','histidine','valine','leucine','isoleucine','proline','threonine','cysteine','phenylalanine','thymine','adenine','guanine','cytosine','uracil','ribose','deoxyribose','sugar','sugars')
# per candidate suffix, exact match only
for ab,ann,name in [(dv,da,'dev'),(tr,ta,'train')]:
 gold={(pm,s,b,e) for pm,ss in ann.items() for s,sp in ss.items() for b,e,*_ in sp}
 print('\n',name)
 for suf in S:
  pred=set()
  for pm,secs in ab.items():
   for sec,text in secs.items():
    for w,b,e in ner.tokenized(text):
     if w.casefold().endswith(suf) and len(w)>len(suf)+1:pred.add((pm,sec,b,e))
  tp=len(pred&gold)
  if tp or (len(pred)>0):print(suf,'n',len(pred),'tp',tp,'p',round(tp/len(pred),2),'r',round(tp/len(gold),3))
