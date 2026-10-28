import argparse
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--pred',required=True);p.add_argument('--split',default='development');p.add_argument('--out',required=True);a=p.parse_args()
# Whole-mention exclusions explicitly listed in the task guidelines. Longer
# specific chemical names containing these words are preserved.
words='lead gold water aqueous organic inorganic proton atom ion molecule polymer isomer enantiomer monomer dimer lipid protein functional-group aglycone oligosaccharide glycoside insulin DNA mRNA collagen starch cellulose glycogen peptidoglycan glycoprotein melanin analogue inhibitor agonist antioxidant drug compound hormone antibiotic toxin metabolite solvent reagent buffer catalyst electrophile cation anion acid base alkali'.lower().split()
candidates=set(words)|{x+'s' for x in words};candidates|={'functional group','functional groups'}
observed=Counter()
for split in ['training','development']:
 for l in open('/workspace/data/'+split+'.annotations.txt'):
  fields=l.rstrip('\n').split('\t');observed[fields[4].casefold()]+=1
# Keep any expression actually labeled as a chemical in supplied data.
forbidden={x for x in candidates if not observed[x]}
texts={}
for l in open('/workspace/data/'+a.split+'.abstracts.txt'):
 pm,t,ab=l.rstrip('\n').split('\t');texts[pm,'T']=t;texts[pm,'A']=ab
pred=set();removed=set()
for l in open(a.pred):
 pm,x=l.rstrip().split('\t');c,s,e=x.split(':');s,e=int(s),int(e);span=(pm,c,s,e)
 (removed if texts[pm,c][s:e].casefold() in forbidden else pred).add(span)
print('removed',len(removed),flush=True)
if a.split=='development':
 gold=set()
 for l in open('/workspace/data/development.annotations.txt'):
  pm,c,s,e,*_=l.rstrip('\n').split('\t');gold.add((pm,c,int(s),int(e)))
 for desc,q in [('before',pred|removed),('after',pred)]:print(desc,2*len(q&gold)/(len(q)+len(gold)),flush=True)
with open(a.out,'w') as f:
 for pm,c,s,e in sorted(pred):f.write(f'{pm}\t{c}:{s}:{e}\n')
