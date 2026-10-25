import numpy as np

def viterbi(logits,offs,text,bias=0,word=True,valid=True):
 if len(offs)==0:return []
 lg=logits.copy();lg[:,1:]+=bias
 scores=lg[0].copy();back=[]
 for i in range(1,len(offs)):
  transition=np.zeros((3,3),np.float32)
  if valid:transition[0,2]=-10000
  a=offs[i][0];b=offs[i-1][1]
  if word and a==b and a>0 and a<len(text) and text[a-1].isalpha() and text[a].isalpha():
   transition[:]=-10000;transition[0,0]=0;transition[1:,2]=0
  m=scores[:,None]+transition;ptr=m.argmax(0);scores=lg[i]+m[ptr,np.arange(3)];back.append(ptr)
 tag=int(scores.argmax());tags=[tag]
 for ptr in back[::-1]:tag=int(ptr[tag]);tags.append(tag)
 return tags[::-1]
