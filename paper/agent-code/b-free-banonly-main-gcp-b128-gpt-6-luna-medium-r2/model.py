import re, sys, math, random, collections
BASE='/workspace/data/'
TOKEN_RE=re.compile(r'\w+|[^\w\s]', re.UNICODE)

class Doc:
 def __init__(self,pmid,sections): self.pmid=pmid; self.sections=sections

def load(split):
 docs=[]
 for line in open(BASE+split+'.abstracts.txt',encoding='utf8'):
  p,t,a=line.rstrip('\n').split('\t',2); docs.append(Doc(p,{'T':t,'A':a}))
 anns=collections.defaultdict(list)
 for line in open(BASE+split+'.annotations.txt',encoding='utf8'):
  p,s,b,e,txt,c=line.rstrip('\n').split('\t'); anns[p,s].append((int(b),int(e),txt,c))
 for d in docs:
  for s,text in d.sections.items():
   toks=[(m.start(),m.end(),m.group()) for m in TOKEN_RE.finditer(text)]
   labels=['O']*len(toks)
   for b,e,_,_ in anns[d.pmid,s]:
    ix=[i for i,(x,y,w) in enumerate(toks) if x<e and y>b]
    if ix:
     labels[ix[0]]='B'
     for j in ix[1:]: labels[j]='I'
   yield d,s,text,toks,labels

def shape(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 return re.sub(r'(.)\1+',r'\1',s)[:15]
def word_features(toks,i):
 w=toks[i][2]; lo=w.lower(); n=len(w)
 f=[]
 f += ['bias','w='+lo,'shape='+shape(w),'len='+str(min(n,12))]
 for k in (1,2,3,4,5):
  if n>=k: f += ['pre%d=%s'%(k,lo[:k]),'suf%d=%s'%(k,lo[-k:])]
 if w.isupper(): f+=['upper']
 if w.istitle(): f+=['title']
 if w.islower(): f+=['lower']
 if any(c.isdigit() for c in w): f+=['hasdigit']
 if any(c.isalpha() for c in w) and any(c.isdigit() for c in w): f+=['alnum']
 if any(c in '-−–' for c in w): f+=['hyphen']
 if any(c in 'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔ' for c in w): f+=['greek']
 if len(w)==1 and w.isalpha(): f+=['singlealpha']
 # alphabetic and token patterns
 for c in set(w):
  if not c.isalnum(): f += ['char='+c]
 for delta in (-2,-1,1,2):
  j=i+delta
  if 0<=j<len(toks):
   v=toks[j][2].lower()
   f += ['w%d=%s'%(delta,v)]
   if delta in (-1,1):
    f += ['shape%d=%s'%(delta,shape(toks[j][2]))]
  else: f+=['edge%d'%delta]
 # nearest separators and sentence position
 if i==0 or toks[i-1][2] in '.!?': f+=['sentstart']
 if i+1==len(toks) or toks[i+1][2] in '.!?': f+=['sentend']
 return f

def all_features(toks,i,prev):
 f=word_features(toks,i)
 f += ['prev='+prev]
 if i: f += ['prevwtag='+toks[i-1][2].lower()+'|'+prev]
 return f

class Perceptron:
 def __init__(self): self.W=collections.defaultdict(collections.Counter)
 def predict_one(self,feats,bias=None):
  bias=bias or {}
  scores={c:sum(self.W[c][f] for f in feats)+bias.get(c,0) for c in ('O','B','I')}
  return max(('O','B','I'),key=lambda c:scores[c])
 def predict(self,toks,bias=None):
  out=[]; prev='O'
  for i in range(len(toks)):
   pred=self.predict_one(all_features(toks,i,prev),bias)
   if pred=='I' and prev=='O': pred='B'
   out.append(pred); prev=pred
  return out
 def train(self, examples, epochs=20):
  # online structured perceptron; averaged across epochs
  snap=collections.defaultdict(collections.Counter)
  for ep in range(epochs):
   random.seed(104+ep); order=list(range(len(examples))); random.shuffle(order)
   for ix in order:
    toks,gold=examples[ix]
    pred=[]; prevp='O'; prevg='O'
    for i in range(len(toks)):
     pf=all_features(toks,i,prevp); guess=self.predict_one(pf)
     if guess=='I' and prevp=='O': guess='B'
     gf=all_features(toks,i,prevg); true=gold[i]
     if guess!=true:
      for f in gf: self.W[true][f]+=1
      for f in pf: self.W[guess][f]-=1
     pred.append(guess); prevp=guess; prevg=true
   for c in ('O','B','I'):
    snap[c].update(self.W[c])
  # average epoch endpoint weights
  for c in ('O','B','I'):
   self.W[c]=collections.Counter({f:v/epochs for f,v in snap[c].items() if v})

def spans_from_tags(toks,tags):
 out=[]; st=None; en=None
 for (a,b,w),tag in zip(toks,tags):
  if tag=='B':
   if st is not None: out.append((st,en))
   st,en=a,b
  elif tag=='I':
   if st is None: st=a
   en=b
  else:
   if st is not None: out.append((st,en)); st=en=None
 if st is not None: out.append((st,en))
 return out

def get_dictionary(splits):
 d=collections.Counter()
 for split in splits:
  for line in open(BASE+split+'.annotations.txt',encoding='utf8'):
   p,s,b,e,txt,c=line.rstrip('\n').split('\t'); d[txt]+=1
 return d

def dict_matches(text,d):
 out=[]
 for term,freq in d.items():
  if len(term.strip())<2: continue
  if len(term)<=3 and not term.isupper() and not any(c.isdigit() for c in term):
   # Short ordinary words are too ambiguous; formula-like tokens remain valid.
   continue
  flags=re.I
  if len(term)<=3: flags=0
  pat=re.compile(r'(?<!\w)'+re.escape(term)+r'(?!\w)',flags)
  for m in pat.finditer(text): out.append((m.start(),m.end()))
 return out

def evaluate(train_splits,devsplit='development', use_dict=True):
 examples=[]; tr_gold=set()
 for split in train_splits:
  for d,s,text,toks,labels in load(split): examples.append((toks,labels))
 model=Perceptron(); model.train(examples,25)
 dic=get_dictionary(train_splits)
 tp=fp=fn=0
 for d,s,text,toks,gold in load(devsplit):
  gs=set(spans_from_tags(toks,gold)); ps=set(spans_from_tags(toks,model.predict(toks)))
  if use_dict: ps.update(dict_matches(text,dic))
  tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps)
 print('train',train_splits,'TP FP FN',tp,fp,fn,'P/R/F1',tp/(tp+fp or 1),tp/(tp+fn or 1),2*tp/(2*tp+fp+fn or 1))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='eval':
  evaluate(['training'])
 else:
  # Train combined and write predictions
  examples=[]
  for split in ('training','development'):
   for d,s,text,toks,labels in load(split): examples.append((toks,labels))
  m=Perceptron(); m.train(examples,30)
  dic=get_dictionary(['training','development'])
  import os
  os.makedirs('/workspace/output',exist_ok=True)
  with open('/workspace/output/predictions.tsv','w',encoding='utf8') as out:
   for d,s,text,toks,labels in load('test'):
    ps=set(spans_from_tags(toks,m.predict(toks)))
    ps.update(dict_matches(text,dic))
    for a,b in sorted(ps): out.write(f'{d.pmid}\t{s}:{a}:{b}\n')
