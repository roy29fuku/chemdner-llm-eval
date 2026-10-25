import os,re,pickle,collections
ROOT='/workspace'
TOK=re.compile(r"[A-Za-z]+(?:[0-9]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",re.UNICODE)
def toks(text):return [(m.group(),m.start(),m.end()) for m in TOK.finditer(text)]
def shape(w):
 s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
 return re.sub(r'(A+)','A',re.sub(r'(a+)','a',re.sub(r'(0+)','0',s)))[:12]
def features(text,forms):
 ts=toks(text);out=[]
 for i,(w,b,e) in enumerate(ts):
  wl=w.lower();f={'bias':1.0,'w':wl,'shape':shape(w),'isupper':w.isupper(),'istitle':w.istitle(),'isdigit':w.isdigit(),'hasdigit':any(c.isdigit() for c in w),'prefix1':wl[:1],'prefix2':wl[:2],'prefix3':wl[:3],'suffix2':wl[-2:],'suffix3':wl[-3:],'len':min(len(w),20),'hasGreek':any(ord(c)>127 for c in w),'hasHyphen':'-' in w,'chem_suffix':any(wl.endswith(x) for x in ['ol','one','ate','ide','ine','ium','yl','ose','ase','ene','ane','azole','amine','acid','oxide','oate','thiol','phenyl']),'chem_prefix':any(wl.startswith(x) for x in ['hydroxy','amino','methyl','ethyl','chloro','bromo','fluoro','nitro','oxo','acetyl','benzyl']),'dict_'+wl:forms.get(wl,0)>0,'BOS':i==0,'EOS':i==len(ts)-1}
  for j in [-2,-1,1,2]:
   if 0<=i+j<len(ts):z=ts[i+j][0];f[f'{j}:w']=z.lower();f[f'{j}:shape']=shape(z)
  out.append(f)
 return ts,out

def predict_spans(text,forms,model):
 ts,ff=features(text,forms); labs=[]
 for st in range(0,len(ts),180):labs.extend(model.predict_single(ff[st:st+180]))
 spans=[];start=last=None
 for (w,b,e),lab in zip(ts,labs):
  if lab=='B':
   if start is not None:spans.append((start,last))
   start=b;last=e
  elif lab=='I':
   if start is None:start=b
   last=e
  else:
   if start is not None:spans.append((start,last));start=last=None
 if start is not None:spans.append((start,last))
 # collapse any overlapping outputs if chunk boundary caused transition split
 out=[]
 for a,b in spans:
  if out and a<out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return out

def load_abs(path):
 d={}
 for l in open(path,encoding='utf8'):
  p,t,a=l.rstrip('\n').split('\t');d[p]={'T':t,'A':a}
 return d

def load_forms():
 c=collections.Counter()
 for l in open(ROOT+'/data/training.annotations.txt',encoding='utf8'):
  p,s,b,e,t,cl=l.rstrip('\n').split('\t');c[t.lower()]+=1
 return c

if __name__=='__main__':
 texts=load_abs(ROOT+'/data/test.abstracts.txt');forms=load_forms();model=pickle.load(open(ROOT+'/model/crf.pkl','rb'))
 os.makedirs(ROOT+'/output',exist_ok=True)
 with open(ROOT+'/output/predictions.tsv','w',encoding='utf8') as f:
  for p,ss in texts.items():
   for s,text in ss.items():
    for a,b in predict_spans(text,forms,model):
     if 0<=a<b<=len(text):f.write(f'{p}\t{s}:{a}:{b}\n')
