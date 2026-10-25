import numpy as np

def viterbi(v):
 rows=v.tolist(); n=len(rows)
 if not n:return []
 d0,d1,_=rows[0];d2=-10000.;back=bytearray(n*3)
 for k in range(1,n):
  if d0>=d1 and d0>=d2:m=d0;arg=0
  elif d1>=d2:m=d1;arg=1
  else:m=d2;arg=2
  if d1>=d2:mi=d1;argi=1
  else:mi=d2;argi=2
  back[3*k]=arg;back[3*k+1]=arg;back[3*k+2]=argi
  x0,x1,x2=rows[k];d0=m+x0;d1=m+x1;d2=mi+x2
 if d0>=d1 and d0>=d2:tag=0
 elif d1>=d2:tag=1
 else:tag=2
 tags=bytearray(n);tags[-1]=tag
 for k in range(n-1,0,-1):tag=back[3*k+tag];tags[k-1]=tag
 return tags
