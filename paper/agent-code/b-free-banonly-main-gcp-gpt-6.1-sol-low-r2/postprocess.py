import argparse,re,json
from collections import defaultdict,Counter
P=argparse.ArgumentParser();P.add_argument('--split',default='development');P.add_argument('--pred',required=True);P.add_argument('--out');P.add_argument('--lexicon',action='store_true');P.add_argument('--repeat',action='store_true');P.add_argument('--mincount',type=int,default=2);P.add_argument('--minlen',type=int,default=3);args=P.parse_args()
texts={};articles=defaultdict(dict)
for line in open('/workspace/data/'+args.split+'.abstracts.txt'):
 pm,t,a=line.rstrip('\n').split('\t')
 for sec,text in [('T',t),('A',a)]:texts[pm,sec]=text;articles[pm][sec]=text
pred=set()
for line in open(args.pred):
 pm,span=line.rstrip().split('\t');sec,s,e=span.split(':');pred.add((pm,sec,int(s),int(e)))
gold=set()
if args.split!='test':
 for line in open('/workspace/data/'+args.split+'.annotations.txt'):
  pm,sec,s,e,*_=line.rstrip('\n').split('\t');gold.add((pm,sec,int(s),int(e)))
def score(p):
 if gold:
  tp=len(p&gold);print('f1',2*tp/(len(p)+len(gold)),'precision',tp/len(p),'recall',tp/len(gold),'n',len(p),flush=True)
score(pred);added=set()
occupied=defaultdict(list)
for pm,sec,s,e in pred:occupied[pm,sec].append((s,e))
if args.repeat:
 bypm=defaultdict(Counter)
 for pm,sec,s,e in pred:
  name=texts[pm,sec][s:e]
  if len(name)>=args.minlen and len(name)<=100:bypm[pm][name]+=1
 for pm,names in bypm.items():
  for name,n in names.items():
   if n<args.mincount:continue
   pat=re.compile(r'(?<!\w)'+re.escape(name)+r'(?!\w)')
   for sec,text in articles[pm].items():
    for m in pat.finditer(text):
     span=(pm,sec,m.start(),m.end())
     if not any(s<m.end() and e>m.start() for s,e in occupied[pm,sec]):added.add(span)
if args.lexicon:
 # Learn chemical surface forms solely from supplied training annotations.
 names=Counter()
 for line in open('/workspace/data/training.annotations.txt'):
  pm,sec,s,e,name,*_=line.rstrip('\n').split('\t');names[name]+=1
 candidates={n for n,c in names.items() if c>=4 and len(n)>=3 and len(n)<=100}
 pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(candidates,key=len,reverse=True))+r')(?!\w)')
 total=Counter()
 for line in open('/workspace/data/training.abstracts.txt'):
  pm,t,a=line.rstrip('\n').split('\t')
  for text in [t,a]:
   for m in pattern.finditer(text):total[m.group()]+=1
 reliable={n for n in candidates if names[n]/max(total[n],1)>=.98}
 pat=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(reliable,key=len,reverse=True))+r')(?!\w)')
 for (pm,sec),text in texts.items():
  for m in pat.finditer(text):
   span=(pm,sec,m.start(),m.end())
   if not any(s<m.end() and e>m.start() for s,e in occupied[pm,sec]):added.add(span)
print('added',len(added),'correct',len(added&gold),'precision',len(added&gold)/max(len(added),1),flush=True);pred|=added;score(pred)
if args.out:
 with open(args.out,'w') as f:
  for pm,sec,s,e in sorted(pred):f.write(f'{pm}\t{sec}:{s}:{e}\n')
