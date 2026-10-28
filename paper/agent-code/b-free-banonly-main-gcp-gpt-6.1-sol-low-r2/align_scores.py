# Align token scores from one general pretrained tokenizer onto another.
import argparse
import numpy as np
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--source-model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--target-model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract');p.add_argument('--split',default='development');p.add_argument('--scores',required=True);p.add_argument('--out',required=True);a=p.parse_args()
src=AutoTokenizer.from_pretrained(a.source_model,**({'do_lower_case':False} if 'biobert-base-cased' in a.source_model else {}));dst=AutoTokenizer.from_pretrained(a.target_model);raw=np.load(a.scores);out=[];idx=0
for line in open('/workspace/data/'+a.split+'.abstracts.txt'):
 pm,t,ab=line.rstrip('\n').split('\t')
 for text in [t,ab]:
  so=src(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];do=dst(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];sc=raw['arr_'+str(idx)];idx+=1;aligned=np.zeros((len(do),3),np.float32);j=0
  for i,(s,e) in enumerate(do):
   while j<len(so) and so[j][1]<=s:j+=1
   k=j;parts=[]
   while k<len(so) and so[k][0]<e:
    u,v=so[k];l=sc[k].copy();w=min(e,v)-max(s,u)
    if w>0:
     if u<s:
      l[2]=np.logaddexp(l[1],l[2]);l[1]=l[0]-12
     elif u>s:
      ent=np.logaddexp(l[1],l[2]);l[1]=ent;l[2]=ent
     parts.append((w,l))
    k+=1
   if parts:aligned[i]=sum(w*l for w,l in parts)/sum(w for w,l in parts)
   else:aligned[i]=[10,-10,-10]
  out.append(aligned)
np.savez_compressed(a.out,*out);print('aligned',len(out))
