import sys,re,unicodedata
sys.path.insert(0,'/workspace');import ner,chebi_eval as ch
from collections import Counter,defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def cands_for(text, knownseq=(), add_known_matches=False):
 toks=ner.tokenize(text); low=[x[0].lower() for x in toks]; by=defaultdict(list)
 for i in range(len(toks)):
  for n in range(1,min(12,len(toks)-i)+1):
   seq=tuple(low[i:i+n])
   base=seq
   if base not in ch.lex and base and len(base[-1])>4 and base[-1].endswith('s'):
    base=base[:-1]+(base[-1][:-1],)
   if (base in ch.lex and ch.allowed(base,ch.lex[base])) or (add_known_matches and base in knownseq): by[(i,i+n)].append((seq,base))
 return toks,[(i,j,toks[i][1],toks[j-1][2],seq,base) for i,j in by for seq,base in by[(i,j)]]

def featurize(text,toks,c,is_title,known):
 i,j,s,e,seq,base=c; term=' '.join(seq); base_term=' '.join(base); chars=text[s:e]; words=[x[0] for x in toks]
 prev=words[i-1].lower() if i else '<BOS>'; foll=words[j].lower() if j<len(words) else '<EOS>'
 alpha=''.join(x for x in chars.lower() if x.isalpha())
 f={'term':term,'base.term':base_term,'plural_variant':seq!=base,'title':is_title,'ntok':min(j-i,10),'nchar':min(len(chars),60),'seen_positive':term in known,'base_seen_positive':base_term in known,
    'prev':prev,'next':foll,'prev_next':prev+'|'+foll,
    'shape':ner.shape(chars),'upper':chars.isupper(),'initial_upper':chars[:1].isupper(),
    'digit':any(x.isdigit() for x in chars),'hyphen':any(x in chars for x in '-−'),
    'greek':any('GREEK' in unicodedata.name(x,'') for x in chars),
    'exact_annotation_string':term in known}
 for n in (2,3,4,5,6):
  if len(alpha)>=n: f['prefix'+str(n)]=alpha[:n];f['suffix'+str(n)]=alpha[-n:]
 # relaxed morphology clues
 for suf in ('ine','ol','one','ide','ate','ose','acid','yl','amine','amide','phenol','benz','glycoside','sulfate'):
  f['suffix?'+suf]=alpha.endswith(suf);f['contains?'+suf]=suf in alpha
 for w in seq:
  if w in {'and','or',',',';'}:f['list.punct']=True
 return f

def get_data(names, known_source=None, add_known_matches=False):
 X=[];y=[];meta=[]; docs={};anns=defaultdict(lambda:defaultdict(list))
 for name in names:
  docs.update(ner.load_articles(name)); aa=ner.load_annotations(name)
  for p,ss in aa.items():
   for sec,ls in ss.items(): anns[p][sec].extend(ls)
 source_anns=anns if known_source is None else known_source
 known={a[2].lower() for ss in source_anns.values() for ls in ss.values() for a in ls}
 knownseq={tuple(x[0].lower() for x in ner.tokenize(term)) for term in known if ner.tokenize(term)}
 for p,secs in docs.items():
  for sec,text in secs.items():
   toks,cs=cands_for(text,knownseq,add_known_matches); gold={(a,b) for a,b,_,_ in anns.get(p,{}).get(sec,[])}
   for c in cs:
    f=featurize(text,toks,c,sec=='T',known); X.append(f);y.append(int((c[2],c[3]) in gold));meta.append((p,sec,text,toks,c))
 return X,np.array(y),meta,docs,anns,known

if __name__=='__main__':
 X,y,tm,_,_,known=get_data(['training']); ta=ner.load_annotations('training');DX,dy,dm,docs,anns,_=get_data(['development'],ta)
 print('candidate dataset',len(y),sum(y),len(dy),sum(dy))
 vec=DictVectorizer();xx=vec.fit_transform(X);dxx=vec.transform(DX)
 for z in [xx,dxx]:z.indices=z.indices.astype(np.int32);z.indptr=z.indptr.astype(np.int32)
 gold={(p,s,a,b)for p,ss in anns.items()for s,aa in ss.items()for a,b,_,_ in aa}
 for cw in [None,'balanced']:
  clf=LogisticRegression(C=.15,max_iter=250,class_weight=cw,solver='lbfgs');clf.fit(xx,y);probs=clf.predict_proba(dxx)[:,list(clf.classes_).index(1)]
  print('weight',cw)
  for th in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
   pred={(p,s,c[2],c[3])for (p,s,t,toks,c),pr in zip(dm,probs)if pr>=th}
   print(th,ner.score(pred,gold))
