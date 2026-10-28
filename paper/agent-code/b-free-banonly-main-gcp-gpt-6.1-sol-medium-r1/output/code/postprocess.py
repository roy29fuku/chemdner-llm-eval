from ner import *
from lexicon import matches, build_automaton, lower
import re

# Explicit excluded concepts from the supplied guidelines. Match entire mentions.
EXCLUDED = set('water waters ice steam proton protons lead gold organic inorganic aqueous lipid lipids protein proteins polymer polymers isomer isomers enantiomer enantiomers monomer monomers dimer dimers oligosaccharide oligosaccharides glycoside glycosides functional_group functional_groups aglycone aglycones insulin dna rna mrna collagen starch cellulose glycogen peptidoglycan peptidoglycans glycoprotein glycoproteins melanin compound compounds drug drugs inhibitor inhibitors agonist agonists antioxidant antioxidants hormone hormones antibiotic antibiotics toxin toxins metabolite metabolites solvent solvents reagent reagents buffer buffers catalyst catalysts electrophile electrophiles cation cations anion anions heparin chitosan hemoglobin pectin polysaccharide polysaccharides'.split())
EXCLUDED |= {'functional group','functional groups'}
ELEMENTS=set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D T'.split())
SHORT_FORMULAS=set('NO CO OH WC NH CH SH CS HD CD CN IO HO OD Et Me Ph'.split())

def filter_predictions(pred,docs,short=False,excluded=True):
    byid={d['pmid']:d for d in docs};result=set()
    for p,s,a,b in pred:
        d=byid[p];text=(d['title'] if s=='T' else d['abstract'])[a:b]
        if excluded and lower(text) in EXCLUDED:continue
        if short and len(text)<=2 and text.isalpha() and text not in ELEMENTS and text not in SHORT_FORMULAS:continue
        if short and text.replace('(','').replace(')','') in {'E2','T3','T4'}:continue
        result.add((p,s,a,b))
    return result

def lexicon_boost(docs,scores,lexicon,strength=0.5,min_count=2,min_precision=.95,role_context=False):
    names={n for n,v in lexicon.items() if len(n)>=3 and v['exact']>=min_count and v['precision']>=min_precision}
    automaton=build_automaton(names)
    result=[]
    for d,score in zip(docs,scores):
        added=np.zeros_like(score)
        for sec,text in [('T',d['title']),('A',d['abstract'])]:
            indices=[i for i,s in enumerate(d['sections']) if s==sec]
            starts=[d['offsets'][i][0] for i in indices];ends=[d['offsets'][i][1] for i in indices]
            for a,b,name in matches(automaton,text):
                if role_context and not re.match(r'\s+(?:compounds?|derivatives?|analogues?|groups?)\b',text[b:],re.I):continue
                i=bisect.bisect_left(starts,a);j=bisect.bisect_left(ends,b)
                if i==len(starts) or j==len(ends) or starts[i]!=a or ends[j]!=b:continue
                if i==j:added[indices[i],4]=max(added[indices[i],4],strength)
                else:
                    added[indices[i],1]=max(added[indices[i],1],strength)
                    added[indices[j],3]=max(added[indices[j],3],strength)
                    for k in indices[i+1:j]:added[k,2]=max(added[k,2],strength)
        result.append(score+added)
    return result

def propagate(pred,docs,min_length=4):
    byid=defaultdict(list)
    for p,s,a,b in pred:byid[p].append((s,a,b))
    result=set(pred)
    for d in docs:
        chosen=byid[d['pmid']]
        names=set()
        for s,a,b in chosen:
            t=(d['title'] if s=='T' else d['abstract'])[a:b]
            if len(t)>=min_length:names.add(lower(t))
        if not names:continue
        auto=build_automaton(names)
        for sec,text in [('T',d['title']),('A',d['abstract'])]:
            intervals=[(a,b) for s,a,b in chosen if s==sec]
            for a,b,name in sorted(matches(auto,text),key=lambda m:-(m[1]-m[0])):
                if any(a<bb and b>aa for aa,bb in intervals):continue
                result.add((d['pmid'],sec,a,b))
                intervals.append((a,b))
    return result

