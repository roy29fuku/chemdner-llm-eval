import os,sys
sys.path.insert(0,os.path.dirname(__file__))
import torch
import hf_ner as h
from transformers import AutoTokenizer,AutoModelForTokenClassification

model_dir=sys.argv[1]
texts,keys,anns=h.load('development')
tok=AutoTokenizer.from_pretrained(model_dir,use_fast=True)
examples=h.encode(texts,keys,anns,tok)
model=AutoModelForTokenClassification.from_pretrained(model_dir)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
best=h.evaluate(model,examples,keys,anns,device)
with open('/home/agent/dev_bert.tsv','w') as f:
    for p,s,a,b in sorted(best[2],key=lambda x:(int(x[0]),x[1],x[2],x[3])):
        f.write(f'{p}\t{s}:{a}:{b}\n')
with open('/home/agent/dev_bias.txt','w') as f:f.write(str(best[1])+'\n')
