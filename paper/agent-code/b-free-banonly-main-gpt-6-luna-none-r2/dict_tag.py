import re,collections,sys,os
ROOT='/workspace'
def load(base):
 texts={}
 for l in open(f'{ROOT}/data/{base}.abstracts.txt',encoding='utf8'):
  p,t,a=l.rstrip('\n').split('\t');texts[p]={'T':t,'A':a}
 anns=collections.defaultdict(list)
 try:
  for l in open(f'{ROOT}/data/{base}.annotations.txt',encoding='utf8'):
   p,s,b,e,tx,cl=l.rstrip('\n').split('\t');anns[(p,s)].append((int(b),int(e),tx))
 except FileNotFoundError: pass
 return texts,anns

def build(train,dev=None):
 # dictionary name variants from training mentions. high support and remove overly generic term based on context later
 forms=collections.Counter()
 for (p,s),aa in train[1].items():
  for b,e,tx in aa:
   key=tx.lower().strip()
   if key: forms[key]+=1
 return forms

def detect(text, forms, minfreq=1):
 # token sequences allow punctuation/connectors as in original spellings
 out=[]
 candidates=[x for x,n in forms.items() if n>=minfreq]
 # escape, longest scan using case insensitive regex with exact boundaries
 # patterns with whitespace flexible
 pats=[]
 for x in candidates:
  # escape but normalize whitespace
  pat=re.escape(x)
  pat=pat.replace(r'\ ',r'\s+')
  pat=r'(?<![A-Za-z0-9])'+pat+r'(?![A-Za-z0-9])'
  try:
   for m in re.finditer(pat,text,re.I): pats.append((m.start(),m.end(),x,forms[x]))
  except: pass
 # resolve overlaps: favor more frequent then longer
 pats.sort(key=lambda z:(z[0],-(z[3]),-(z[1]-z[0])))
 chosen=[]
 for item in sorted(pats,key=lambda z:(-z[3],-(z[1]-z[0]),z[0])):
  a,b,_,_=item
  if any(not (b<=c or a>=d) for c,d in chosen): continue
  chosen.append((a,b))
 return sorted(chosen)

if __name__=='__main__':
 tr=load('training'); dev=load('development'); forms=build(tr)
 # eval by support cutoff, case sensitivity, most common false positive review
 for threshold in [1,2,3,4,5,10]:
  tp=fp=fn=0
  for (p,s),aa in dev[1].items():
   text=dev[0][p][s]; pred=detect(text,forms,threshold); gold={(a,b) for a,b,_ in aa}
   pr=set(pred);tp+=len(pr&gold);fp+=len(pr-gold);fn+=len(gold-pr)
  print(threshold,tp,fp,fn,2*tp/(2*tp+fp+fn))
