import torch
from bert_ner import *
args=sys.argv[1]
articles=read_abstracts('data/training.abstracts.txt');ann=read_annotations('data/training.annotations.txt')
tok=AutoTokenizer.from_pretrained(args,use_fast=True);model=AutoModelForTokenClassification.from_pretrained(args);device=torch.device('cuda');model.to(device)
g=goldset(articles,ann)
for th in [.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80]:
 p=predset(predict(model,tok,articles,batch_size=32,threshold=th,device=device));print(th,exact_score(g,p),flush=True)
