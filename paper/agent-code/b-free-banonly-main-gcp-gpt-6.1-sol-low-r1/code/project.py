import numpy as np

def project(logits,source,target,n):
 x=logits-logits.max(-1,keepdims=True);pr=np.exp(x);pr/=pr.sum(-1,keepdims=True)
 ep=np.zeros(n,np.float32);bp=np.zeros(n,np.float32);used=np.zeros(n,np.float32)
 for (a,b),p in zip(source,pr):
  if b<=a:continue
  e=p[1]+p[2];ep[a:b]=e;used[a:b]=1;bp[a]=p[1]/max(e,1e-10)
 out=[]
 for a,b in target:
  e=ep[a:b].sum()/max(used[a:b].sum(),1);fraction=bp[a] if a<n else 0
  out.append([1-e,e*fraction,e*(1-fraction)])
 return np.log(np.clip(out,1e-5,1))
