import re, sys, unicodedata, os, json, math, collections
import sklearn_crfsuite
from sklearn.metrics import precision_recall_fscore_support

P='/workspace/data/'
def load_docs(path):
 d={}
 for line in open(path,encoding='utf8'):
  pmid,title,abstract=line.rstrip('\n').split('\t',2)
  d[pmid]=(title,abstract)
 return d

def tokenize(text):
 # preserve offsets; biomedical tokens often contain punctuation and chemical symbols
 pat=re.compile(r"[A-Za-z]+(?:[0-9]+[A-Za-z]*)*(?:[.+/][A-Za-z0-9]+)*|[0-9]+(?:[.,][0-9]+)*(?:[-+][A-Za-z0-9]+)*|[^\w\s]",re.UNICODE)
 return [(m.group(),m.start(),m.end()) for m in pat.finditer(text)]

def shapes(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 s=re.sub('A+','A+',s);s=re.sub('a+','a+',s);s=re.sub('0+','0+',s)
 return s

def token_features(tok,i,toks,text,section):
 w,s,e=tok; low=w.lower(); n=len(toks)
 f={'bias':1.0,'word.lower':low,'word':w,'shape':shapes(w),'prefix1':low[:1],'prefix2':low[:2],'prefix3':low[:3],'prefix4':low[:4], 'suffix1':low[-1:],'suffix2':low[-2:],'suffix3':low[-3:],'suffix4':low[-4:], 'len':min(len(w),20),'isupper':w.isupper(),'istitle':w.istitle(),'isdigit':w.isdigit(),'hasdigit':any(c.isdigit() for c in w),'hasalpha':any(c.isalpha() for c in w),'section':section}
 # punctuation character hints
 for ch in '-+./=()[],:':
  if ch in w:f['has'+ch]=True
 if i==0:f['BOS']=True
 if i==n-1:f['EOS']=True
 for d in (-2,-1,1,2):
  j=i+d
  if 0<=j<n:
   x=toks[j][0]; xl=x.lower(); tag=('p' if d<0 else 'n')+str(abs(d))
   f[tag+'.lower']=xl; f[tag+'.shape']=shapes(x)
   if abs(d)==1:
    f[tag+'.istitle']=x.istitle();f[tag+'.isupper']=x.isupper();f[tag+'.hasdigit']=any(c.isdigit() for c in x)
 # common context
 return f

def docs_to_seq(docs,anns=None):
 X=[];Y=[]; infos=[]
 for pmid,(title,abstract) in docs.items():
  for section,text in [('T',title),('A',abstract)]:
   toks=tokenize(text); features=[token_features(t,i,toks,text,section) for i,t in enumerate(toks)]
   X.append(features); infos.append((pmid,section,text,toks))
   if anns is not None:
    spans=anns.get((pmid,section),[]); starts={a for a,b in spans}; ends={b for a,b in spans}
    tags=[]
    # token overlapping spans, exact boundaries should align most of time
    for w,s,e in toks:
     overlaps=[(a,b) for a,b in spans if s<b and e>a]
     if not overlaps: tags.append('O')
     else:
      # assume contiguous mention; entity type generic
      tags.append('B' if s==overlaps[0][0] else 'I')
    Y.append(tags)
 return X,Y,infos

def load_ann(path):
 a=collections.defaultdict(list)
 for line in open(path,encoding='utf8'):
  pmid,sec,s,e,txt,cl=line.rstrip('\n').split('\t')
  a[(pmid,sec)].append((int(s),int(e)))
 return a

def tags_to_spans(info,tags):
 pmid,sec,text,toks=info; spans=[]; start=None; end=None
 for (w,s,e),tag in zip(toks,tags):
  if tag=='B' or (tag=='I' and start is None):
   if start is not None:spans.append((start,end))
   start=s;end=e
  elif tag=='I': end=e
  else:
   if start is not None:spans.append((start,end));start=None;end=None
 if start is not None:spans.append((start,end))
 return [(pmid,sec,a,b) for a,b in spans]

def evaluate(gold,pred):
 g=set(gold);p=set(pred);tp=len(g&p);pr=tp/max(1,len(p));re=tp/max(1,len(g));f=2*pr*re/max(1e-10,pr+re)
 return (pr,re,f,tp,len(p),len(g))

if __name__=='__main__':
 train=load_docs(P+'training.abstracts.txt'); dev=load_docs(P+'development.abstracts.txt')
 ann=load_ann(P+'training.annotations.txt'); da=load_ann(P+'development.annotations.txt')
 X,Y,_=docs_to_seq(train,ann);Xd,Yd,infos=docs_to_seq(dev,da)
 print('loaded sequences',len(X),sum(map(len,X)), flush=True)
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--maxiter',type=int,default=100);ap.add_argument('--c1',type=float,default=.05);ap.add_argument('--c2',type=float,default=.1);ap.add_argument('--trainall',action='store_true');args=ap.parse_args()
 crf=sklearn_crfsuite.CRF(algorithm='lbfgs',c1=args.c1,c2=args.c2,max_iterations=args.maxiter,all_possible_transitions=True,verbose=True)
 crf.fit(X,Y)
 yp=crf.predict(Xd)
 gold=[];pred=[]
 for info,tags in zip(infos,Yd):
  gold.extend([(info[0],info[1],a,b) for a,b in da[(info[0],info[1])]])
  pred.extend(tags_to_spans(info,tags))
 print('DEV',evaluate(gold,pred))
