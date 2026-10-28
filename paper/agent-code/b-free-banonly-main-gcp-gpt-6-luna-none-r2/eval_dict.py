from model_fast import *
import pickle,time
pat=pickle.load(open('/workspace/dictpat.pkl','rb'))
dv=read_labels('development'); texts={(p,s):txt for p,s,txt in read_articles('development')}
gold=set();pred=set(); t=time.time()
for k,anns in dv.items():
 txt=texts[k]
 for a,b,c,m in anns:gold.add((*k,a,b))
 for mt in pat.finditer(txt):pred.add((*k,mt.start(),mt.end()))
tp=len(gold&pred);print('dict',tp,len(pred),len(gold),2*tp/(len(pred)+len(gold)),time.time()-t,flush=True)
