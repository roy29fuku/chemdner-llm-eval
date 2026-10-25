import re, pickle, unicodedata
from collections import defaultdict,Counter
from sklearn_crfsuite import CRF
TOKEN_RE=re.compile(r"[A-Za-z]+(?:[0-9]+[A-Za-z]*)*(?:['’][A-Za-z]+)?|[0-9]+(?:[.,][0-9]+)*(?:[A-Za-z]+)?|[^\w\s]", re.UNICODE)
def tokenize(text):return [(m.group(0),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]
def read_articles(kind):
 out=[]
 for line in open(f'/workspace/data/{kind}.abstracts.txt',encoding='utf8'):
  p,t,a=line.rstrip('\n').split('\t');out.extend([(p,'T',t),(p,'A',a)])
 return out
def read_labels(kind):
 d=defaultdict(list)
 for line in open(f'/workspace/data/{kind}.annotations.txt',encoding='utf8'):
  p,s,a,b,m,cl=line.rstrip('\n').split('\t');d[(p,s)].append((int(a),int(b),cl,m))
 return d
def shape(w):return ''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
def features(toks,i):
 w=toks[i][0]; low=w.lower(); n=len(toks); sh=shape(w)
 f={'w':low,'shape':sh,'pref2':low[:2],'pref3':low[:3],'pref4':low[:4],'suf2':low[-2:],'suf3':low[-3:],'suf4':low[-4:], 'len':min(len(w),20),'upper':w.isupper(),'title':w.istitle(),'digit':any(c.isdigit() for c in w),'alpha':any(c.isalpha() for c in w),'dash':'-' in w,'plus':'+' in w,'greek':any('GREEK' in unicodedata.name(c,'') for c in w)}
 for off in (-2,-1,1,2):
  j=i+off
  if 0<=j<n:
   ww=toks[j][0];f[str(off)+':w']=ww.lower();f[str(off)+':sh']=shape(ww)
   if abs(off)==1:f[str(off)+':up']=ww.isupper();f[str(off)+':d']=any(c.isdigit() for c in ww)
 if i>0 and i+1<n:f['bi']=toks[i-1][0].lower()+'_'+toks[i+1][0].lower()
 return f
def bio(toks,anns):
 y=['O']*len(toks)
 for a,b,c,m in anns:
  ids=[i for i,t in enumerate(toks) if t[1]>=a and t[2]<=b]
  if ids and toks[ids[0]][1]==a and toks[ids[-1]][2]==b:
   for k,i in enumerate(ids):y[i]=('B-' if k==0 else 'I-')+c
 return y
def build(k):
 anns=read_labels(k); X=[];Y=[];raw=[]
 for p,s,text in read_articles(k):
  ts=tokenize(text);X.append([features(ts,i) for i in range(len(ts))]);Y.append(bio(ts,anns.get((p,s),[])));raw.append((p,s,text,ts,anns.get((p,s),[])))
 return X,Y,raw
def spans(ts,ys):
 out=[];st=None;en=None
 for i,y in enumerate(ys+['O']):
  if y.startswith('B-') or y=='O':
   if st is not None:out.append((ts[st][1],ts[en][2]))
   st=i if y.startswith('B-') else None;en=i if y.startswith('B-') else None
  elif y.startswith('I-'):
   if st is not None:en=i
 return out
