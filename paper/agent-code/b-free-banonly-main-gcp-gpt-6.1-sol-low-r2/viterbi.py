import numpy as np
from numba import njit
@njit(cache=True)
def tags(logit):
 n=len(logit);back=np.zeros((n,3),np.int32);dp=logit[0].copy();dp[2]=-10000
 for i in range(1,n):
  nd=np.zeros(3,np.float32)
  for j in range(3):
   best=-1e20;prev=0
   for k in range(3):
    v=dp[k]
    if k==0 and j==2:v-=10000
    if v>best:best=v;prev=k
   nd[j]=best+logit[i,j];back[i,j]=prev
  dp=nd
 labels=np.zeros(n,np.int32);labels[n-1]=np.argmax(dp)
 for i in range(n-1,0,-1):labels[i-1]=back[i,labels[i]]
 return labels

@njit(cache=True)
def typed_tags(logit):
 n,c=logit.shape;back=np.zeros((n,c),np.int32);dp=logit[0].copy()
 for j in range(2,c,2):dp[j]=-10000
 for i in range(1,n):
  nd=np.empty(c,np.float32);top=np.argmax(dp)
  for j in range(c):
   if j>0 and j%2==0:
    prev=j-1 if dp[j-1]>=dp[j] else j
   else:prev=top
   nd[j]=dp[prev]+logit[i,j];back[i,j]=prev
  dp=nd
 labels=np.zeros(n,np.int32);state=np.argmax(dp)
 for i in range(n-1,-1,-1):
  labels[i]=0 if state==0 else (1 if state%2 else 2)
  if i:state=back[i,state]
 return labels

@njit(cache=True)
def lse(values):
 m=np.max(values)
 return m+np.log(np.sum(np.exp(values-m)))

@njit(cache=True)
def bio_posterior(logit):
 n,c=logit.shape
 if n==0:return np.empty((0,3),np.float32)
 f=np.empty((n,c),np.float32);b=np.zeros((n,c),np.float32);f[0]=logit[0]
 for j in range(2,c,2):f[0,j]=-10000
 for i in range(1,n):
  allprev=lse(f[i-1])
  for j in range(c):
   if j>0 and j%2==0:
    f[i,j]=np.logaddexp(f[i-1,j-1],f[i-1,j])+logit[i,j]
   else:f[i,j]=allprev+logit[i,j]
 for i in range(n-2,-1,-1):
  vals=logit[i+1]+b[i+1];base=lse(vals[np.array([0]+list(range(1,c,2)))])
  b[i,0]=base
  for j in range(1,c,2):
   v=np.logaddexp(base,vals[j+1]);b[i,j]=v;b[i,j+1]=v
 norm=lse(f[n-1]);out=np.empty((n,3),np.float32)
 for i in range(n):
  p=f[i]+b[i]-norm;out[i,0]=p[0];out[i,1]=lse(p[1::2]);out[i,2]=lse(p[2::2])
 return out
