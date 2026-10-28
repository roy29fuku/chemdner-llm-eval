import os,sys
import torch
sys.path.insert(0,'/home/agent')
import ner_transformer as n
from transformers import AutoTokenizer,AutoModelForTokenClassification

tok=AutoTokenizer.from_pretrained('/home/agent/scibert-chem',use_fast=True)
m=AutoModelForTokenClassification.from_pretrained('/home/agent/scibert-chem').cuda();torch.set_num_threads(4)
items=n.make_items('development',tok);lb=n.infer_items(m,items,tok);dic=n.dictionary(['training'])
best=[]
for b in [-4,-3,-2,-1,0,1,2,3,4]:
 for i in [-4,-3,-2,-1,0,1,2,3,4]:
  f=n.eval_preds(items,lb,dic,(0,b,i))
  best.append((f,b,i))
print('BEST',sorted(best,reverse=True)[:15])
