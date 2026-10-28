import re,json,argparse
from collections import defaultdict,Counter
from pathlib import Path
from lexicon import read,build,match,metric

def load_preds(path):
 pred=set()
 for line in open(path):
  p,span=line.rstrip().split('\t');s,a,b=span.split(':');pred.add((p,s,int(a),int(b)))
 return pred

def propagation(pred,docs,minchars=3,mincount=1,case_sensitive=True):
 bysec=defaultdict(list);texts={(p,s):t for p,s,t in docs};counts=defaultdict(Counter)
 for p,s,a,b in pred:
  bysec[p,s].append((a,b))
  name=texts[p,s][a:b]
  if len(name)>=minchars and len(name)<=100 and '\n' not in name:counts[p][name]+=1
 out=set(pred)
 for p,s,text in docs:
  existing=list(bysec[p,s])
  for name,n in sorted(counts[p].items(),key=lambda x:(-len(x[0]),x[0])):
   if n<mincount:continue
   pat=re.compile(re.escape(name),0 if case_sensitive else re.I)
   for m in pat.finditer(text):
    a,b=m.span()
    if a and text[a-1].isalnum():continue
    if b<len(text) and text[b].isalnum():continue
    if any(a<y and b>x for x,y in existing):continue
    out.add((p,s,a,b));existing.append((a,b))
 return out

def abbreviation(pred,docs):
 out=set(pred);bysec=defaultdict(list);alias=defaultdict(set)
 for p,s,a,b in pred:bysec[p,s].append((a,b))
 for p,s,text in docs:
  for a,b in bysec[p,s]:
   name=text[a:b]
   # Adjacent bracketed acronyms, grounded by the letters in the long name.
   m=re.match(r'\s*\(([A-Za-z][A-Za-z0-9+\-]{2,14})\)',text[b:])
   if m:
    short=m.group(1);alnum=''.join(c.lower() for c in name if c.isalnum());letters=''.join(c.lower() for c in short if c.isalpha());j=0
    for c in alnum:
     if j<len(letters) and c==letters[j]:j+=1
    if j==len(letters) and len(name)>len(short):alias[p].add(short)
 for p,s,text in docs:
  spans=bysec[p,s]
  for short in sorted(alias[p],key=lambda x:(-len(x),x)):
   for m in re.finditer(re.escape(short),text):
    a,b=m.span()
    if a and text[a-1].isalnum():continue
    if b<len(text) and text[b].isalnum():continue
    if any(a<y and b>x for x,y in spans):continue
    out.add((p,s,a,b));spans.append((a,b))
 return out

def dictionary(pred,docs,auto,stats,threshold=.98,mincount=5):
 out=set(pred);bysec=defaultdict(list)
 for p,s,a,b in pred:bysec[p,s].append((a,b))
 for p,s,text in docs:
  for a,b,t in match(auto,text):
   h,n,f=stats[t]
   if h<mincount or h/max(1,n)<threshold:continue
   if any(a<y and b>x for x,y in bysec[p,s]):continue
   out.add((p,s,a,b))
 return out

def exclusions(pred,docs,positive_terms=None):
 terms=set('organic inorganic water proton lead gold aqueous insulin dna mrna collagen starch cellulose glycogen peptidoglycans glycoproteins melanin atom ion molecule polymer isomer enantiomer monomer dimer lipid protein glycoside oligosaccharide aglycone acid base metal alkali analogue inhibitor agonist antioxidant drug compound hormone antibiotic toxin metabolite solvent reagent buffer catalyst electrophile cation anion'.split())
 terms|={t+'s' for t in terms}
 positives=set(positive_terms) if positive_terms is not None else set()
 if positive_terms is None:
  for line in open('/workspace/data/training.annotations.txt'):
   positives.add(line.rstrip('\n').split('\t')[4].lower())
 terms-=positives; texts={(p,s):t for p,s,t in docs}
 return {x for x in pred if texts[x[0],x[1]][x[2]:x[3]].lower() not in terms}

def expansion(pred,docs,auto,stats,threshold=.95,mincount=3):
 out=set(pred);bysec=defaultdict(list)
 for p,s,a,b in pred:bysec[p,s].append((a,b))
 for p,s,text in docs:
  matches=[]
  for a,b,t in match(auto,text):
   h,n,f=stats[t]
   if h<mincount or h/max(1,n)<threshold:continue
   inner=[(x,y) for x,y in bysec[p,s] if a<y and b>x]
   if not inner or any(x<a or y>b for x,y in inner):continue
   if (a,b) in inner:continue
   matches.append((a,b,inner))
  for a,b,inner in sorted(matches,key=lambda x:x[1]-x[0],reverse=True):
   if all((p,s,x,y) in out for x,y in inner):
    for x,y in inner:out.remove((p,s,x,y))
    out.add((p,s,a,b))
 return out

def write(path,pred):
 with open(path,'w') as f:
  for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('pred');ap.add_argument('--split',default='development');ap.add_argument('--out');ap.add_argument('--propagate',type=int,default=0);ap.add_argument('--abbreviation',action='store_true');ap.add_argument('--dict',action='store_true');ap.add_argument('--exclude',action='store_true');ap.add_argument('--expand',action='store_true');args=ap.parse_args()
 docs,gold=read(args.split);pred=load_preds(args.pred);print('baseline',metric(pred,gold),flush=True)
 if args.split!='test':
  for count in [1,2,3]:
   for case in [True,False]:
    p=propagation(pred,docs,mincount=count,case_sensitive=case);print('propagation',count,case,metric(p,gold),'additions',len(p-pred),'correct',len((p-pred)&gold),flush=True)
  p=exclusions(pred,docs);print('exclusions',metric(p,gold),'removed',len(pred-p),flush=True)
  p=abbreviation(pred,docs);print('abbreviations',metric(p,gold),'additions',len(p-pred),'correct',len((p-pred)&gold),flush=True)
  auto,stats=build(['training'])
  for th in [.9,.95,.98,1]:
   for count in [2,5,10]:
    p=expansion(pred,docs,auto,stats,th,count);print('expansion',th,count,metric(p,gold),'additions',len(p-pred),'correct',len((p-pred)&gold),'removed correct',len((pred-p)&gold),flush=True)
   for count in [2,5,10]:
    p=dictionary(pred,docs,auto,stats,th,count);print('dictionary',th,count,metric(p,gold),'additions',len(p-pred),'correct',len((p-pred)&gold),flush=True)
 if args.propagate:pred=propagation(pred,docs,mincount=args.propagate)
 if args.abbreviation:pred=abbreviation(pred,docs)
 if args.dict:
  auto,stats=build(['training','development'] if args.split=='test' else ['training']);pred=dictionary(pred,docs,auto,stats)
 if args.expand:
  auto,stats=build(['training','development'] if args.split=='test' else ['training']);pred=expansion(pred,docs,auto,stats)
 if args.exclude:pred=exclusions(pred,docs)
 if args.out:write(args.out,pred)

if __name__=='__main__':main()
