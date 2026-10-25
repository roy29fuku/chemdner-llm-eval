import re, json, math, collections, unicodedata, os, sys
DATA='/workspace/data/'

def load_articles(fn):
 out=[]
 for line in open(DATA+fn,encoding='utf8'):
  p=line.rstrip('\n').split('\t',2)
  out.append((p[0],p[1],p[2]))
 return out

def load_ann(fn):
 d=collections.defaultdict(set)
 for l in open(DATA+fn,encoding='utf8'):
  p=l.rstrip('\n').split('\t'); pm,sec,a,b,txt,cl=p
  d[pm].add((sec,int(a),int(b)))
 return d

# frequency-backed phrase lexicon from training + dev
freq=collections.Counter(); classes=collections.defaultdict(collections.Counter)
for fn in ['training.annotations.txt','development.annotations.txt']:
 for l in open(DATA+fn,encoding='utf8'):
  p=l.rstrip('\n').split('\t'); form=p[4].lower()
  freq[form]+=1; classes[form][p[5]]+=1
# high-precision atomic common chemical nouns / families drawn from common terminology, not a recognizer
EXTRA='''acetate acetic acid acetone acetyl acetylcholine adenine adenosine adrenaline alanine albumin alcohol aldehyde alkaloid alkane alkene alkyne alkyl amide amine amino acid ammonia ammonium aniline anthocyanin arabinose arginine arsenic ascorbate ascorbic acid aspartate aspartic acid aspirin atom? atropine benzene benzoate benzoic acid benzyl bile acid biotin bisphenol bromide butanol caffeine calcite calcium carbonate camphor carbohydrate carbon carbon dioxide carbon monoxide carbonic acid carboxylate carboxylic acid carotenoid casein catechol catechin cellulose ceramide cetyl chitosan chlorate chloride chlorophyll cholesterol choline chromate citric acid citrate cobalt codeine collagen copper corticosteroid cortisol coumarin creatine cyanide cyclohexane cysteine cytidine cytosine deoxyribose deoxyribonucleic acid dextran dextrose dioxane dioxin dopamine doxorubicin drug? ethanol ethyl ethylene ethylene glycol eugenol fatty acid flavonoid formaldehyde formate formic acid fructose fucose fumarate fumaric acid galactose galactosamine gallate gamma-aminobutyric acid glucose glutamate glutamic acid glutamine glutathione glycerol glycine glycol glycogen guanine guanosine heme histamine histidine hydroxide ibuprofen imidazole indole insulin? iodine iodide iron isopropanol ketone lactose lactate lactic acid leucine ligand lipid? linoleic acid lysine magnesium malate malic acid maltose manganese melatonin menthol methacrylate methane methanol methyl methylamine methylation methionine methylparaben mineral? morphine myristic acid nadh nad+ naproxen naphthalene nicotine nitrate nitric oxide nitrite nitrogen nitrobenzene nitrogen oxide nucleotide oleic acid oxalate oxalic acid oxide oxygen palmitate palmitic acid paraffin paracetamol paraquat phenol phenolic phosphate phosphoric acid phosphorus phthalate picric acid piperidine platinum polyamine polymer? polyphenol polysaccharide? potassium progesterone proline propanol propylene glycol protein? purine pyridine pyrimidine pyruvate quercetin quinine quinone resveratrol retinol riboflavin ribose riboflavin ribonucleic acid RNA? salicylate salicylic acid serine silica silicon silver sodium sorbitol starch? stearate stearic acid steroid sucrose sulfate sulfur sulfonamide tannin taurine testosterone threonine thymidine thymine tocopherol toluene trans-fat triglyceride tryptophan tyrosine uracil urea uric acid valine vanillin vitamin A vitamin B vitamin C vitamin D vitamin E vitamin K xanthine xylene zinc'''.split()
# only straightforward words/multiword; remove concepts and obvious excluded
stop=set('atom drug insulin polymer lipid protein starch polysaccharide'.split())
for w in EXTRA:
 if w.endswith('?'): continue
 if w in stop: continue
 freq[w.lower()]=max(freq[w.lower()],3)

# Trie over normalized chemical mention strings keeps matching linear in text length.
# Preserve original offsets by mapping each lowercase-character index to original index.
class TrieNode(dict):
    __slots__=('end',)
    def __init__(self): super().__init__(); self.end=False
trie=TrieNode(); lexforms=[]
NOISY={'no','co','in','as','or','at','is','it','he','h','n','c','s','o','ca','mg','al','and','of','to','by','on','up','do','be','we','all','may','can','was','are','not','use','used','cell','cells','acid','base','salt','metal','drug','drugs','compound','compounds','protein','proteins','water','group','groups','dna','insulin','control','within','provide','membrane','peptide','appropriate','evaluate','investigate','indicate','determine','demonstrate','aggregate','gas','nucleotide','patient','treatment','assessment','management','development','improvement','measurement','response','involvement','experiment','environment','prevalence','incidence','resistance','sensitivity','activity','toxicity','mortality','community','human','different','significant','associated','increased','decreased','compared','reported','observed','induced','treated','isolated','identified','analysis','diagnosis','disease','syndrome','enzyme','receptor','transporter','inhibitor','activator','derivative','mixture','antibiotic','antibacterial','antioxidant','antitumor','antimicrobial','antiviral','antifungal','anticonvulsant'}
for f,n in freq.items():
    if len(f)<2 or n<2 or (len(f)<=3 and n<6) or f in NOISY or f.endswith('-') or f in {'pro','ligand','collagen','methyl','methylation','evaluate','insulin','protein'}: continue
    # do not match generic labels / obvious non-CEM classes even when annotations occasionally contain them
    node=trie
    for ch in f: node=node.setdefault(ch,TrieNode())
    node.end=True; lexforms.append(f)

def lex_matches(text):
    low=text.lower(); found=[]; n=len(low)
    for i,ch in enumerate(low):
        if not (ch.isalnum() or ch=='_') and ch not in 'αβγδεζηθικλμνξοπρστυφχψω': continue
        if i and (low[i-1].isalnum() or low[i-1]=='_'): continue
        node=trie; j=i; last=-1
        while j<n and low[j] in node:
            node=node[low[j]]; j+=1
            if node.end and (j==n or not (low[j].isalnum() or low[j]=='_')): last=j
        if last>i: found.append((i,last))
    # nested hits collapse to longest start-end spans. Distinct adjacent names preserved.
    found.sort(key=lambda z:(z[0],-(z[1]-z[0])))
    out=[]
    for st,en in found:
        if out and st==out[-1][0]: continue
        if out and st<out[-1][1]: continue
        out.append((st,en))
    return out
# chemical formula/name structured patterns, with enough composition to avoid ordinary words
formula_patterns=[
 r'(?<![A-Za-z0-9])(?:\[(?:\d{1,3}H?|\d{14}C|[A-Z][a-z]?\+?)[^\]]{0,30}\]|\(\d{1,3}(?:H|C)\)|\(\d{1,3}\)[HC])',
 r'(?<![A-Za-z0-9])(?:\d{1,3}[- ]?)?(?:[A-Z][a-z]?\d{0,3}){2,}(?:\(aq\)|\(s\)|\(l\)|\(g\))?(?:\d*[+-]|\([IVX]+\))?(?![A-Za-z0-9])',
 r'(?<![A-Za-z])(?:[A-Z]{2,10}\d{0,4}(?:[-/][A-Z0-9]+)*)(?![A-Za-z])',
 r'(?<!\w)(?:[A-Z][a-z]?\+|[A-Z][a-z]?\d*[+-]|[A-Z][a-z]?\([IVX]+\))(?!\w)',
 r'(?<!\w)(?:[A-Z][a-z]?/[A-Z][a-z]?)(?!\w)',
]
pat_re=[re.compile(p) for p in formula_patterns]
# chemical systematic morphology, prefer tokens with characteristic chemistry suffixes/prefixes
chemword=re.compile(r'(?<!\w)(?:[0-9]+(?:-[0-9]+)*(?:,\s*[0-9]+)*(?:-[A-Za-z]+)?-)?(?:[A-Za-zα-ωΑ-Ω]+(?:-[A-Za-z0-9α-ωΑ-Ω]+){0,5})(?!\w)')
chem_suffix=re.compile(r'(?:ol|one|ones|olide|ide|ate|ite|ine|in|ene|ane|yne|ose|ose|acid|oxide|amine|amide|ester|ether|azole|azine|olol|phenol|quinone|glycoside|terpene|terpenoid|alkaloid|flavonoid|steroid|toxin|mycin|cycline|caine|pril|sartan|olol|mab|tinib|navir|vir)$',re.I)
# Avoid generic linguistic endings where they don't imply a structure

def predict(sec):
 spans=set()
 # exact phrase lexicon longest matching; overlaps prioritize longest candidate locally
 candidates=[]
 for st,en in lex_matches(sec):
  f=sec[st:en].lower(); n=freq.get(f,0)
  
  if len(f)<=3 and n<10 and f not in {'n','no','c','co','s','o','h','ca','mg','as','k','al','p','f','au','fe','zn','cu','hg','mn','na','cl','cd','cr','pd','si','se','br','i','b','li','ni','pt','ag'}: continue
  if f in {'pd','na','cd','cl','mn','cr','gd','se','pt','fe','ser'}: continue
  if f in {'tea','ir','1h','2h','hydroxy','ethyl','amino','phenyl','glycogen','cb1','sar','mmp','ester','acetate','silymarin','his','casein','chitosan','alkaloid','statins','mda','doi','carbonic','cannabinoid','fatty','rt-pcr','gc-ms','sirt1','5-ht2a','t2dm','ec50','pc12','raw264','foxo1','fmo3','atm','mdr1','bmp-2','lc-ms/ms','phenolic','iron','alcohol','trp','sodium','bile','chloride','phosphate','dihydroxy','arf','oxide','gnrh'}: continue
  if f in {'protein','insulin','collagen','peptide','dna','nucleotide','prostate','medicine','metabolite','hormone','methylation','control','provide','membrane','demonstrate','indicate','investigate','determine','whether','together','evaluate','substrate','purpose','despite','examine','baseline','ligand','cellulose','albumin'}: continue
  candidates.append((st,en,n, 'lex'))
 # regex formula/codes: filter ordinary terms
 for pr in pat_re:
  for m in pr.finditer(sec):
   st,en=m.span(); s=m.group()
   if len(s)<2: continue
   if s.startswith('[') and s.endswith(']') and s[1:-1].lower() in freq: continue
   # Formula must have digit, symbol punctuation, uppercase multi-letter token, or chemical charge
   if not (any(c.isdigit() for c in s) or any(c in '+-[]().' for c in s) or (s.isupper() and len(s)>2 and s.lower() in freq)):
    continue
   # suppress standard abbreviations/initialisms unless chemical-looking or learned lex form
   if not re.fullmatch(r'(?:HTR\d+[A-Z]?|EGFR|PIK3CA|GAD|HAM-A|CGI-I|BLCO|MS/MS|FVCs?|C57BL.*|CYP\w+|MCF[-]?\d+|PI3K|COX[-]?\d+|IL[-]?\d+|TNF[-]?|TGF[-]?|NF[-]?|IC\d+|IR|TEA|E2|NOR|CYP.*|HIV[-]?\d+|ERK\d+/\d+|TRPA\d+|HO[-]?\d+|HIF[-]?\d+|MDA[-]?MB[-]?\d+|IFN[-]?|BL/6|N-|CB1|ERK\d+|MDA|ABC[A-Z]\d+|BL/6[A-Z]?|HER\d+|CD\d+|MAO-B|H2AX|RT-PCR|GC-MS|SIRT\d+|T2DM|EC\d+|5-HT\d+[A-Z]?|LC-MS/MS|MDR\d+|PC\d+|RAW\d+|FOXO\d+|FMO\d+|ATM|BMP-\d+|SAR|MMP|PLA|PET|PPI|TRP|GC-MS|SIRT\d+|5-HT\d+[A-Z]?|T2DM|EC\d+)', s): candidates.append((st,en,3,'pat'))
 # morphology only with informative suffix and a chemically suggestive multi-character token
 for m in []: # avoid broad morphology false positives
  s=m.group(); st,en=m.span()
  if len(s)<7 or not chem_suffix.search(s): continue
  if s.lower() in {'patient','treatment','assessment','management','development','improvement','measurement','response','involvement','experiment','environment','prevalence','incidence','resistance','sensitivity','activity','toxicity','mortality','community','human','different','significant','associated','increased','decreased','compared','reported','observed','induced','treated','isolated','identified','analysis','diagnosis','disease','syndrome','enzyme','receptor','transporter','inhibitor','activator','compound','derivative','mixture','mixtures','antibiotic','antibacterial','antioxidant','antitumor','antimicrobial','antiviral','antifungal','anticonvulsant'}: continue
  # require recognizable chemical structure indicators; suffix list includes common English-ish endings
  candidates.append((st,en,1,'morph'))
 # greedily resolve spans: longest first per overlapping component, lexical preferred at same span
 candidates.sort(key=lambda z: (z[0], -(z[1]-z[0]), -z[2]))
 accepted=[]
 for st,en,n,typ in candidates:
  # lexicon longest span among overlaps; patterns supersede when formula is longer
  overlap=[i for i,(a,b,*_) in enumerate(accepted) if st<b and en>a]
  if not overlap:
   accepted.append((st,en,n,typ)); continue
  # if contained in already accepted then skip
  if all(st>=accepted[i][0] and en<=accepted[i][1] for i in overlap): continue
  # if new span contains prior spans, replace only if actual lexical or formula
  if all(st<=accepted[i][0] and en>=accepted[i][1] for i in overlap) and typ in ('lex','pat'):
   for i in reversed(overlap): accepted.pop(i)
   accepted.append((st,en,n,typ))
 # exact span output (will be tuned with val)
 return sorted((a,b) for a,b,*_ in accepted)

def evaluate(articles, gold):
 tp=fp=fn=0; per=[]
 for pm,t,a in articles:
  pred=set()
  for sec,s in [('T',t),('A',a)]:
   pred|={(sec,x,y) for x,y in predict(s)}
  g=gold[pm]; tp+=len(pred&g); fp+=len(pred-g); fn+=len(g-pred)
  per.append((pm,pred,g))
 p=tp/(tp+fp); r=tp/(tp+fn); f=2*p*r/(p+r)
 return p,r,f,tp,fp,fn,per

if __name__=='__main__':
 dev=load_articles('development.abstracts.txt'); gold=load_ann('development.annotations.txt')
 p,r,f,tp,fp,fn,per=evaluate(dev,gold)
 print('dev',p,r,f,tp,fp,fn,'lex',len(lexforms))
 # collect false examples for diagnostics
 fp_items=[]; fn_items=[]
 d={pm:(t,a) for pm,t,a in dev}
 for pm,p,g in per:
  t,a=d[pm]
  for sec,x,y in p-g:
   fp_items.append((sec[t if False else 0] if False else pm, sec,x,y,(t if sec=='T' else a)[x:y]))
  for sec,x,y in g-p:
   fn_items.append((pm,sec,x,y,(t if sec=='T' else a)[x:y]))
 print('FP examples',fp_items[:80])
 print('FN examples',fn_items[:100])
