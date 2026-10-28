import numpy as np
from numba import njit

@njit
def path_decode(scores,bias):
    n=scores.shape[0]; back=np.zeros((n,5),np.int32);dp=np.zeros(5,np.float32)
    for j in range(5):dp[j]=scores[0,j]+(bias if j==0 else 0)+(0 if j in (0,1,4) else -10000)
    for i in range(1,n):
        new=np.empty(5,np.float32)
        for j in range(5):
            best=-1e20;arg=0
            for k in range(5):
                allowed=((k==0 or k==3 or k==4) and (j==0 or j==1 or j==4)) or ((k==1 or k==2) and (j==2 or j==3))
                v=dp[k]+(0 if allowed else -10000)
                if v>best:best=v;arg=k
            new[j]=best+scores[i,j]+(bias if j==0 else 0);back[i,j]=arg
        dp=new
    best=-1e20;last=0
    for j in (0,3,4):
        if dp[j]>best:best=dp[j];last=j
    out=np.zeros(n,np.int32);out[n-1]=last
    for i in range(n-1,0,-1):out[i-1]=back[i,out[i]]
    return out

def decode(doc,scores,bias=0):
    result=set()
    for sec in ('T','A'):
        inds=[i for i,x in enumerate(doc['offsets']) if x[0]==sec]
        if not inds:continue
        tags=path_decode(scores[inds].astype(np.float32),bias);start=None
        for i,tag in zip(inds,tags):
            _,a,b=doc['offsets'][i]
            if tag in (1,4):start=a
            if tag in (3,4) and start is not None:
                result.add((sec,start,b));start=None
    return result

def evaluate(docs,logits,bias=0):
    tp=fp=fn=0
    for d,s in zip(docs,logits):
        pred=decode(d,s,bias);gold=d['gold'];tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return dict(f1=2*p*r/max(p+r,1e-10),precision=p,recall=r,tp=tp,fp=fp,fn=fn)
