import numpy as np
try:
 from numba import njit
except ImportError:
 def njit(*a,**kw):return lambda f:f

@njit(cache=True)
def path(lg,inside,valid):
 n=len(lg);back=np.zeros((n,3),np.int8);score=lg[0].copy()
 for i in range(1,n):
  nxt=np.empty(3,dtype=lg.dtype)
  for t in range(3):
   best=-1e30;prev=0
   for r in range(3):
    penalty=0.
    if valid and r==0 and t==2:penalty=-10000.
    if inside[i] and not ((r==0 and t==0) or (r>0 and t==2)):penalty=-10000.
    val=score[r]+penalty
    if val>best:best=val;prev=r
   nxt[t]=lg[i,t]+best;back[i,t]=prev
  score=nxt
 tags=np.empty(n,np.int8);tag=np.argmax(score);tags[-1]=tag
 for i in range(n-1,0,-1):tag=back[i,tag];tags[i-1]=tag
 return tags

def viterbi(logits,offs,text,bias=0,word=True,valid=True):
 if len(offs)==0:return []
 lg=logits.copy();lg[:,1:]+=bias
 inside=np.zeros(len(offs),np.bool_)
 if word:
  for i in range(1,len(offs)):
   a=offs[i][0];b=offs[i-1][1]
   inside[i]=a==b and a>0 and a<len(text) and text[a-1].isalpha() and text[a].isalpha()
 return path(lg,inside,valid)
