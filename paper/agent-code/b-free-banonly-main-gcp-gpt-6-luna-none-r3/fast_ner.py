import re, json, math, os, collections, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack, csr_matrix

DATA='/workspace/data'
def abstracts(name):
 d={}
 for l in open(DATA+'/'+name,encoding='utf8'):
  p,t,a=l.rstrip('\n').split('\t',2); d[p]=(t,a)
 return d
def annotations(name):
 d=collections.defaultdict(list)
 for l in open(DATA+'/'+name,encoding='utf8'):
  p,s,a,b,txt,cl=l.rstrip('\n').split('\t'); d[(p,s)].append((int(a),int(b)))
 return d
train=abstracts('training.abstracts.txt'); dev=abstracts('development.abstracts.txt'); test=abstracts('test.abstracts.txt')
ta=annotations('training.annotations.txt'); da=annotations('development.annotations.txt')
# Treat every character offset as a binary classification decision, with features centered on word/punctuation context.
def word_spans(text): return [(m.start(),m.end(),m.group()) for m in re.finditer(r"[A-Za-z0-9]+(?:[.+/#-][A-Za-z0-9]+)*|[^\w\s]",text)]
def features(text, spans):
 words=[]
 for a,b,w in spans: words.append((a,b,w))
 feats=[]; targets=[]; offsets=[]
 # features each word candidate for boundary decisions: chemistry clues in word itself and context
 for i,(a,b,w) in enumerate(words):
  low=w.lower(); f={}
  def add(k,v=1): f[k]=v
  for n in range(1, min(6,len(low))+1):
   add('pref%d:%s'%(n,low[:n])); add('suf%d:%s'%(n,low[-n:]))
  for ch in set(low): add('char:'+ch)
  add('w:'+low); add('shape:'+re.sub('[A-Z]','A',re.sub('[a-z]','a',re.sub('[0-9]','0',w))))
  add('len',min(len(w),25)); add('upper',w.isupper()); add('title',w.istitle()); add('digits',any(c.isdigit() for c in w)); add('hyphen','-' in w); add('slash','/' in w); add('plus','+' in w)
  for d in (-2,-1,1,2):
   if 0<=i+d<len(words):
    z=words[i+d][2].lower(); add(f'ctx{d}:'+z)
    add(f'ctxshape{d}:'+re.sub('[A-Z]','A',re.sub('[a-z]','a',re.sub('[0-9]','0',z))))
  for d in (-1,0,1):
   if 0<=i+d<len(words):
    z=words[i+d][2].lower()
    for n in range(2,min(5,len(z))+1):
     add(f'ctxsuf{d}:{z[-n:]}')
  # punctuation/nearby morphology
  add('prefixchar',text[max(0,a-1):a]); add('suffixchar',text[b:b+1])
  feats.append(f); offsets.append((a,b))
 return feats,offsets

def labels_for(text, spans, ann):
 y=[]
 for a,b,w in spans:
  # 0 outside, 1 continuation, 2 beginning; exact token boundaries learned as BIO
  ys=[(x<=a and b<=z) for x,z in ann]
  y.append(0 if not any(ys) else 1)
 for i,(a,b,w) in enumerate(spans):
  if y[i]==1 and (i==0 or y[i-1]!=1 or not any(x<=spans[i-1][0] and spans[i-1][1]<=z for x,z in ann if x<=a and b<=z)):
   # Use actual start offset: beginning if annotation begins in this token
   if any(x==a for x,z in ann): y[i]=2
 return y

Xdict=[]; yy=[]; meta=[]
for p,(t,a) in train.items():
 for s,text in [('T',t),('A',a)]:
  sp=word_spans(text); fs,ofs=features(text,sp); Xdict.extend(fs); yy.extend(labels_for(text,sp,ta.get((p,s),[])))
print('train tokens',len(yy),collections.Counter(yy),flush=True)
vec=DictVectorizer(dtype=np.float32); X=vec.fit_transform(Xdict).astype(np.float32); X.indices=X.indices.astype(np.int32); X.indptr=X.indptr.astype(np.int32)
clf=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=15,tol=1e-4,class_weight={0:1,1:2,2:2},random_state=17,n_jobs=-1)
clf.fit(X,np.array(yy,dtype=np.int8)); del Xdict,X,yy
print('fit done',flush=True)
# Evaluation exact token span via contiguous B/I predictions; report threshold on max probability for positive class

def predict_text(text, threshold=.55):
 sp=word_spans(text); fs,ofs=features(text,sp); xx=vec.transform(fs); xx.indices=xx.indices.astype(np.int32); xx.indptr=xx.indptr.astype(np.int32); probs=clf.predict_proba(xx); classes=list(clf.classes_)
 # Flatten B/I scores; BIO agreement handled by stitching
 pchem=probs[:,classes.index(1)]+probs[:,classes.index(2)]
 y=(pchem>=threshold)
 # split on predicted B where present, otherwise breaks
 out=[]; i=0
 while i<len(y):
  if not y[i]: i+=1; continue
  j=i+1
  while j<len(y) and y[j] and probs[j,classes.index(2)]<pchem[j]: j+=1
  out.append((ofs[i][0],ofs[j-1][1])); i=j
 return out

def score(articles, anns, thr):
 tp=fp=fn=0
 for p,(t,a) in articles.items():
  for s,text in [('T',t),('A',a)]:
   gold=set(anns.get((p,s),[])); pred=set(predict_text(text,thr)); tp+=len(gold&pred);fp+=len(pred-gold);fn+=len(gold-pred)
 prec=tp/(tp+fp or 1); rec=tp/(tp+fn or 1); return prec,rec,2*prec*rec/(prec+rec or 1),tp,fp,fn
for th in [.35,.45,.55,.65,.75]: print('THR',th,score(dev,da,th),flush=True)
# optional train dictionary postprocess evaluation
def mkdict(minc=2):
 c=collections.Counter()
 for p,(t,a) in train.items():
  for s,text in [('T',t),('A',a)]:
   for x,y in ta.get((p,s),[]): c[text[x:y].lower()]+=1
 return {w for w,n in c.items() if n>=minc and len(w)>=2}

def adddict(text, base, dic):
 # word boundary-like case-insensitive matches; restrict to phrases with at least one letter
 out=set(base)
 for w in dic:
  if not re.search('[a-z]',w): continue
  for m in re.finditer(r'(?<![A-Za-z0-9])'+re.escape(w)+r'(?![A-Za-z0-9])',text,re.I): out.add((m.start(),m.end()))
 return out

def score_dict(thr, mindict=2):
 dic=mkdict(mindict); tp=fp=fn=0
 for p,(t,a) in dev.items():
  for s,text in [('T',t),('A',a)]:
   gold=set(da.get((p,s),[])); pred=adddict(text,predict_text(text,thr),dic); tp+=len(gold&pred);fp+=len(pred-gold);fn+=len(gold-pred)
 pr=tp/(tp+fp or 1);rc=tp/(tp+fn or 1);return pr,rc,2*pr*rc/(pr+rc or 1),len(dic)
for m in (2,3,5,10): print('DICT',m,score_dict(.45,m),flush=True)
