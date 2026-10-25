import re, collections, numpy as np, os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
DATA='/workspace/data'; OUT='/workspace/output'
def abstracts(name):
 d={}
 for l in open(DATA+'/'+name,encoding='utf8'):
  p,t,a=l.rstrip('\n').split('\t',2);d[p]=(t,a)
 return d
def annotations(name):
 d=collections.defaultdict(list)
 for l in open(DATA+'/'+name,encoding='utf8'):
  p,s,a,b,txt,cl=l.rstrip('\n').split('\t');d[(p,s)].append((int(a),int(b)))
 return d
train=abstracts('training.abstracts.txt');dev=abstracts('development.abstracts.txt');test=abstracts('test.abstracts.txt');ta=annotations('training.annotations.txt');da=annotations('development.annotations.txt')
def word_spans(text): return [(m.start(),m.end(),m.group()) for m in re.finditer(r"[A-Za-z0-9]+(?:[.+/#-][A-Za-z0-9]+)*|[^\w\s]",text)]
def features(text,spans):
 fs=[]
 for i,(a,b,w) in enumerate(spans):
  low=w.lower();f={}
  def add(k,v=1):f[k]=v
  for n in range(1,min(6,len(low))+1):add('p%d:%s'%(n,low[:n]));add('s%d:%s'%(n,low[-n:]))
  for ch in set(low):add('c:'+ch)
  add('w:'+low);add('shape:'+re.sub('[A-Z]','A',re.sub('[a-z]','a',re.sub('[0-9]','0',w))))
  add('len',min(len(w),25));add('upper',w.isupper());add('title',w.istitle());add('digit',any(c.isdigit() for c in w));add('hyphen','-' in w);add('slash','/' in w);add('plus','+' in w)
  for d in (-2,-1,1,2):
   if 0<=i+d<len(spans):
    z=spans[i+d][2].lower();add(f'ctx{d}:'+z);add(f'ctxshape{d}:'+re.sub('[A-Z]','A',re.sub('[a-z]','a',re.sub('[0-9]','0',z))))
  for d in (-1,0,1):
   if 0<=i+d<len(spans):
    z=spans[i+d][2].lower()
    for n in range(2,min(5,len(z))+1):add(f'ctxsuf{d}:{z[-n:]}')
  add('prechar',text[max(0,a-1):a]);add('postchar',text[b:b+1]);fs.append(f)
 return fs
def ys(text,sp,ann):
 y=[]
 for a,b,w in sp:
  y.append(1 if any(x<=a and b<=z for x,z in ann) else 0)
  # 2 signifies beginning
  if y[-1] and any(x==a for x,z in ann):y[-1]=2
 return y
D=[];Y=[]
for p,(t,a) in train.items():
 for s,text in [('T',t),('A',a)]:
  sp=word_spans(text);D.extend(features(text,sp));Y.extend(ys(text,sp,ta.get((p,s),[])))
print('vectorizing',len(Y),flush=True)
vec=DictVectorizer(dtype=np.float32);X=vec.fit_transform(D);X.indices=X.indices.astype(np.int32);X.indptr=X.indptr.astype(np.int32)
clf=SGDClassifier(loss='log_loss',alpha=2e-6,max_iter=15,tol=1e-4,class_weight={0:1,1:2,2:2},random_state=17,n_jobs=-1)
clf.fit(X,np.asarray(Y,dtype=np.int8)); del X,D,Y
classes=list(clf.classes_); TH=.45

def predict(text):
 sp=word_spans(text);xx=vec.transform(features(text,sp));xx.indices=xx.indices.astype(np.int32);xx.indptr=xx.indptr.astype(np.int32)
 pr=clf.predict_proba(xx); chem=pr[:,classes.index(1)]+pr[:,classes.index(2)]; y=chem>=TH
 out=[];i=0
 while i<len(y):
  if not y[i]:i+=1;continue
  j=i+1
  while j<len(y) and y[j] and pr[j,classes.index(2)]<chem[j]:j+=1
  out.append((sp[i][0],sp[j-1][1]));i=j
 return out
# precise dictionary phrases observed at least twice, only retain on contexts that score at least weakly chemical.
ct=collections.Counter()
for p,(t,a) in train.items():
 for s,text in [('T',t),('A',a)]:
  for x,y in ta.get((p,s),[]):ct[text[x:y].lower()]+=1
# Do not inject dictionary matches; token classifier already receives lexical supervision and better handles boundaries.
path=OUT+'/predictions.tsv'; n=0
with open(path,'w',encoding='utf8') as f:
 for p,(title,abstract) in test.items():
  for sec,text in [('T',title),('A',abstract)]:
   for a,b in predict(text):f.write(f'{p}\t{sec}:{a}:{b}\n');n+=1
print('wrote',n,'mentions')
