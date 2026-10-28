import pickle,sys
sys.path.insert(0,'/workspace');import ner
from candidate_model import get_data
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

train=ner.load_annotations('training')
X,y,_,_,_,_=get_data(['training'])
DX,dy,dm,dev,da,_=get_data(['development'],train)
v=DictVectorizer();xx=v.fit_transform(X);dd=v.transform(DX)
for z in (xx,dd):z.indices=z.indices.astype(np.int32);z.indptr=z.indptr.astype(np.int32)
clf=LogisticRegression(C=.15,max_iter=250,class_weight=None,solver='lbfgs').fit(xx,y)
pr=clf.predict_proba(dd)[:,list(clf.classes_).index(1)]
dictpred={(p,s,c[2],c[3])for(p,s,t,toks,c),x in zip(dm,pr)if x>=.04}

seqs={tuple(t[0].lower() for t in ner.tokenize(a[2])) for ss in train.values() for ls in ss.values() for a in ls}
direct=set()
for p,secs in dev.items():
 for sec,text in secs.items():
  toks=ner.tokenize(text);lo=[t[0].lower() for t in toks]
  for i in range(len(lo)):
   for n in range(1,min(12,len(lo)-i)+1):
    seq=tuple(lo[i:i+n]);base=seq
    if seq not in seqs and len(seq[-1])>4 and seq[-1].endswith('s'):base=seq[:-1]+(seq[-1][:-1],)
    if base in seqs and toks[i+n-1][2]-toks[i][1]>=3:direct.add((p,sec,toks[i][1],toks[i+n-1][2]))

crf,known=ner.fit(['training']);cx,cy,meta=ner.build_data(dev,da,known);labs=crf.predict(cx)
crfpred={(p,s,a,b)for(p,s,t,toks,ann),ll in zip(meta,labs)for a,b in ner.spans_from_labels(toks,ll)}

with open('/workspace/dev_bert_probs.pkl','rb') as f:(rows,probs)=pickle.load(f)
items=[]
for p,s,text in rows:items.append((p,s,text,da.get(p,{}).get(s,[])))
gold={(p,s,a,b)for p,ss in da.items()for s,aa in ss.items()for a,b,_,_ in aa}
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('distilbert/distilbert-base-cased',use_fast=True)
bert=ner.score(set(),gold)
import bert_ner
bertpred=bert_ner.decode_items(items,probs,tokenizer,.25,.1)
print('dict',ner.score(dictpred,gold))
print('direct',ner.score(direct,gold))
print('crf',ner.score(crfpred,gold))
print('bert',ner.score(bertpred,gold))
for name,pred in [('dict+direct',dictpred|direct),('dict+crf',dictpred|crfpred),('dict+direct+crf',dictpred|direct|crfpred),('dict+direct+bert',dictpred|direct|bertpred),('dict+crf+bert',dictpred|crfpred|bertpred),('all',dictpred|direct|crfpred|bertpred)]:print(name,ner.score(pred,gold))
