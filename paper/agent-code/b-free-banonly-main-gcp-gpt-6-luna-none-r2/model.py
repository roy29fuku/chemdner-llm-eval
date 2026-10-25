import re, json, pickle, math, unicodedata, sys, os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import CRF
from sklearn.metrics import precision_recall_fscore_support

TOKEN_RE=re.compile(r"[A-Za-z]+(?:[0-9]+[A-Za-z]*)*(?:['’][A-Za-z]+)?|[0-9]+(?:[.,][0-9]+)*(?:[A-Za-z]+)?|[^\w\s]", re.UNICODE)

def tokenize(text):
 return [(m.group(0),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]

def read_articles(kind):
 out=[]
 with open(f'/workspace/data/{kind}.abstracts.txt',encoding='utf8') as f:
  for line in f:
   p,t,a=line.rstrip('\n').split('\t'); out.extend([(p,'T',t),(p,'A',a)])
 return out

def read_labels(kind):
 d=defaultdict(list)
 with open(f'/workspace/data/{kind}.annotations.txt',encoding='utf8') as f:
  for line in f:
   p,s,a,b,mention,cl=line.rstrip('\n').split('\t');d[(p,s)].append((int(a),int(b),cl,mention))
 return d

def to_bio(toks, anns):
 lab=['O']*len(toks)
 for a,b,cl,m in anns:
  inds=[i for i,(_,x,y) in enumerate(toks) if x>=a and y<=b]
  if not inds or toks[inds[0]][1]!=a or toks[inds[-1]][2]!=b: continue
  for j,i in enumerate(inds): lab[i]=('B-' if j==0 else 'I-')+cl
 return lab

def shape(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 return re.sub(r'(A+|a+|0+)',lambda m:m.group(0)[0],s)

def features(toks,i):
 w,a,b=toks[i]; low=w.lower(); n=len(toks)
 f={'bias':1,'w':low,'raw':w,'shape':shape(w),'pref1':low[:1],'pref2':low[:2],'pref3':low[:3],'pref4':low[:4], 'suf1':low[-1:],'suf2':low[-2:],'suf3':low[-3:],'suf4':low[-4:], 'len':min(len(w),20), 'isupper':w.isupper(),'istitle':w.istitle(),'islower':w.islower(),'isdigit':w.isdigit(), 'hasdigit':any(c.isdigit() for c in w),'hasalpha':any(c.isalpha() for c in w),'hasdash':'-' in w,'hasplus':'+' in w,'hasparen':any(c in w for c in '()[]'),'hasGreek':any('GREEK' in unicodedata.name(c,'') for c in w)}
 for d in range(1,4):
  for j in (i-d,i+d):
   if 0<=j<n:
    ww=toks[j][0]
    f[f'{j-i}:w']=ww.lower(); f[f'{j-i}:shape']=shape(ww)
    f[f'{j-i}:pre']=ww.lower()[:3];f[f'{j-i}:suf']=ww.lower()[-3:]
    if d==1:
     f[f'{j-i}:upper']=ww.isupper();f[f'{j-i}:title']=ww.istitle();f[f'{j-i}:digit']=any(c.isdigit() for c in ww)
 for d in [1,2]:
  if i-d>=0 and i+d<n:
   f[f'bi{d}']=toks[i-d][0].lower()+'_'+toks[i+d][0].lower()
 return f

def seq_features(toks):
 return [features(toks,i) for i in range(len(toks))]

def build(kind):
 ann=read_labels(kind); X=[]; Y=[]; raw=[]
 for p,s,text in read_articles(kind):
  toks=tokenize(text)
  if not toks: continue
  X.append(seq_features(toks)); Y.append(to_bio(toks,ann.get((p,s),[]))); raw.append((p,s,text,toks,ann.get((p,s),[])))
 return X,Y,raw

def score(gold,pred):
 tp=len(gold&pred); fp=len(pred-gold); fn=len(gold-pred)
 return tp,fp,fn,2*tp/max(1,2*tp+fp+fn),tp/max(1,tp+fp),tp/max(1,tp+fn)

def spans_from_labels(toks, labels):
 out=[]; st=None; en=None
 for i,l in enumerate(labels+['O']):
  if l.startswith('B-') or l=='O':
   if st is not None: out.append((toks[st][1],toks[en][2]))
   st=i if l.startswith('B-') else None; en=i if l.startswith('B-') else None
  elif l.startswith('I-'):
   if st is not None: en=i
  else:
   if st is not None: out.append((toks[st][1],toks[en][2]));st=None;en=None
 return out

if __name__=='__main__':
 X,Y,_=build('training');print('train seqs',len(X),'tokens',sum(map(len,X)))
 crf=CRF(algorithm='lbfgs',c1=.05,c2=.1,max_iterations=120,all_possible_transitions=True,verbose=True)
 crf.fit(X,Y)
 with open('/workspace/model.pkl','wb') as f:pickle.dump(crf,f)
