import gzip,re,sys,collections,pickle
sys.path.insert(0,'/workspace');import ner
CACHE='/workspace/chebi_lexicon.pkl'
try: lex=pickle.load(open(CACHE,'rb')); print('cache',len(lex))
except FileNotFoundError:
 entries=[];cur=None
 for line in gzip.open('/workspace/chebi.obo.gz','rt',encoding='utf8',errors='replace'):
  line=line.rstrip('\n')
  if line=='[Term]':
   if cur:entries.append(cur)
   cur={'syn':[],'obsolete':False}
  elif line.startswith('['):
   if cur:entries.append(cur)
   cur=None
  elif cur is not None:
   if line.startswith('id: '):cur['id']=line[4:]
   elif line.startswith('name: '):cur['name']=line[6:]
   elif line.startswith('is_obsolete: true'):cur['obsolete']=True
   elif line.startswith('synonym: '):
    m=re.match(r'synonym: "(.*?)" (\w+)',line)
    if m and m.group(2)=='EXACT':cur['syn'].append(m.group(1))
 lex=collections.defaultdict(set)
 for en in entries:
  if en['obsolete']:continue
  for phrase in [en.get('name',''),*en['syn']]:
   tok=ner.tokenize(phrase); seq=tuple(t[0].lower() for t in tok)
   if seq and len(phrase)<100 and len(seq)<=12 and any(c.isalpha() for c in phrase):lex[seq].add(phrase)
 pickle.dump(dict(lex),open(CACHE,'wb'),protocol=4);print('parsed',len(lex))

BLACK={'water','water vapor','water vapour','ice','lead','gold','organic','inorganic','atom','ion','molecule','polymer','isomer','enantiomer','monomer','dimer','lipid','protein','proteins','functional group','aglycone','oligosaccharide','glycoside','hormone','antibiotic','toxin','metabolite','solvent','reagent','buffer','catalyst','electrophile','cation','anion','drug','compound','compounds','analogue','analog','inhibitor','agonist','antioxidant','electron','role','group','aptamer','fuel','ligand','activator','antioxidants','lipoprotein','lipoproteins','polypeptide','polypeptides','dna','rna','mrna','starch','cellulose','glycogen','peptidoglycan','glycoprotein','glycoproteins','nucleic acid','single-stranded dna','double-stranded dna','alkali','acid','base','metal','salt','lipid','lipids'}
BAD_CONTAINS=('protein','polypeptide','lipoprotein','glycoprotein','glycogen','cellulose','starch','peptidoglycan','nucleic acid','single-stranded dna','double-stranded dna')

def allowed(seq, phrases):
 phrase=' '.join(seq).strip()
 if phrase in BLACK or any(x in phrase for x in BAD_CONTAINS):return False
 if len(phrase)<=2 and not (len(phrase)==1 and phrase in {'n','c','o','s','p','f','i','k','v','y','u','w','b','h'}):return False
 return True

def candidates(text):
 toks=ner.tokenize(text); low=[x[0].lower() for x in toks]; out=[]
 for i in range(len(toks)):
  for n in range(1,min(12,len(toks)-i)+1):
   seq=tuple(low[i:i+n])
   if seq in lex and allowed(seq,lex[seq]):
    out.append((i,i+n,toks[i][1],toks[i+n-1][2],seq))
 # greedy maximum span per leftmost unoccupied token
 chosen=[]; i=0; by=collections.defaultdict(list)
 for c in out:by[c[0]].append(c)
 while i<len(toks):
  choices=by.get(i)
  if choices:
   c=max(choices,key=lambda c:(c[1]-c[0],c[3]-c[2]))
   chosen.append((c[2],c[3],c[4])); i=c[1]
  else:i+=1
 return chosen,out

if __name__=='__main__':
 arts=ner.load_articles('development'); anns=ner.load_annotations('development');gold={(p,s,a,b)for p,ss in anns.items()for s,aa in ss.items()for a,b,_,_ in aa}
 for greedy in (False,True):
  pred=set(); fp=[]
  for p,secs in arts.items():
   for sec,text in secs.items():
    chosen,all_=candidates(text)
    for a,b,seq in chosen if greedy else [(x[2],x[3],x[4]) for x in all_]:
     z=(p,sec,a,b);pred.add(z)
     if z not in gold:fp.append((text[a:b],seq))
  print('greedy',greedy,'pred',len(pred),'score',ner.score(pred,gold),'FP',len(fp));print('sample',fp[:20])
