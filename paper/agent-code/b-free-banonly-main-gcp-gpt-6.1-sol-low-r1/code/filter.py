import sys
from dictionary import corpus,gold
split,path=sys.argv[1:3]
black=set('water aqueous lead gold proton protons organic inorganic lipid lipids protein proteins polymer polymers isomer isomers enantiomer enantiomers monomer monomers dimer dimers glycoside glycosides oligosaccharide oligosaccharides aglycone aglycones antibiotic antibiotics drug drugs compound compounds analogue analogues analog analogs inhibitor inhibitors agonist agonists antioxidant antioxidants hormone hormones toxin toxins metabolite metabolites solvent solvents reagent reagents buffer buffers catalyst catalysts electrophile electrophiles cation cations anion anions atom atoms ion ions molecule molecules insulin DNA mRNA collagen starch cellulose glycogen peptidoglycan peptidoglycans glycoprotein glycoproteins melanin'.lower().split())
texts={(p,s):txt for p,s,txt in corpus(split)};out=set();removed=set()
for l in open(path):
 p,span=l.strip().split('\t');s,a,b=span.split(':');a=int(a);b=int(b);text=texts[(p,s)][a:b]
 if text.lower() in black:removed.add((p,s,a,b))
 else:out.add((p,s,a,b))
if split!='test':
 gs=gold(split);G={(p,s,a,b) for (p,s),sp in gs.items() for a,b in sp};P=out|removed
 print('before',2*len(P&G)/(len(P)+len(G)),'after',2*len(out&G)/(len(out)+len(G)),'removed',len(removed),'correct removals',len(removed-G))
with open(path+'.filtered','w') as f:
 for p,s,a,b in sorted(out):f.write(f'{p}\t{s}:{a}:{b}\n')
