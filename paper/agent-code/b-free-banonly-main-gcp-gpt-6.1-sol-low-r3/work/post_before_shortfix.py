import re
ELEMENTS=set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D T'.split())
EXCLUDE=set('acid base metal alkali peptides peptide its this these those the an of and or to with from than it them that which who tri di tetra mono poly'.split()) | set('lead gold water aqueous organic inorganic proton protons atom atoms ion ions molecule molecules polymer polymers isomer isomers enantiomer enantiomers monomer monomers dimer dimers lipid lipids protein proteins functional group aglycone oligosaccharide glycoside insulin DNA RNA mRNA collagen starch cellulose glycogen peptidoglycans glycoproteins melanin compound compounds analogue analogues inhibitor inhibitors agonist agonists antioxidant antioxidants drug drugs hormone hormones antibiotic antibiotics toxin toxins metabolite metabolites solvent solvents reagent reagents buffer buffers catalyst catalysts electrophile cation anion reduced oxidized'.lower().split())
def valid(name):
 if not any(c.isalnum() for c in name) or name.isdigit():return False
 if name.lower() in EXCLUDE:return False
 if len(name)<=2 and name not in ELEMENTS and not re.fullmatch(r'[HCNOSFPIDT][23456789]|CO|NO|OH|NH|CH',name):return False
 return True

def filter_pred(pred,text):return {x for x in pred if valid(text[x[:2]][x[2]:x[3]])}

def normalize_pred(pred,text):
 out=set()
 for p,s,a,b in pred:
  t=text[p,s]
  while a<b and t[a] in ' \t\n,;:':a+=1
  while a<b and t[b-1] in ' \t\n,;:.®™':b-=1
  if t[a:b].endswith(("'s",'’s')):b-=2
  if t[a:b].lower().startswith('non-') and not t[a:b].lower().startswith('non-ortho'):a+=4
  if a<b and t[b-1]=='-' and re.match(r'(?:dependent|induced|based|binding|mediated|treated|containing|labeled|loaded|coated|related|sensitive|ATPase|kinase)\b',t[b:],re.I):b-=1
  while a<b and t[a]=='(' and t[a:b].count('(')>t[a:b].count(')'):a+=1
  while a<b and t[b-1]==')' and t[a:b].count(')')>t[a:b].count('('):b-=1
  if a<b and t[a]=='(' and t[b-1]==')':
   depth=0;firstclose=None
   for i,c in enumerate(t[a:b]):
    if c=='(':depth+=1
    if c==')':
     depth-=1
     if depth==0:firstclose=i;break
   if firstclose==b-a-1:a+=1;b-=1
  if a<b:out.add((p,s,a,b))
 return out
