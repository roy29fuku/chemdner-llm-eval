import os, json, random, math, collections, argparse
import numpy as np
import torch
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

ap=argparse.ArgumentParser(); ap.add_argument('--data',default='/workspace/data'); ap.add_argument('--model',default='google-bert/bert-base-cased'); ap.add_argument('--out',default='/workspace/output/bert_best'); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--maxlen',type=int,default=512); ap.add_argument('--stride',type=int,default=128); ap.add_argument('--final',action='store_true'); ap.add_argument('--final-bias',type=float,default=0.0); ap.add_argument('--bucket',default=os.getenv('GCS_BUCKET','')); args=ap.parse_args()
LABELS={'O':0,'B-CHEM':1,'I-CHEM':2}; ID2LABEL={v:k for k,v in LABELS.items()}
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(8,os.cpu_count() or 1))

def read_articles(path):
 d={}
 with open(path,encoding='utf8') as f:
  for line in f:
   p,t,a=line.rstrip('\n').split('\t',2); d[p]=(t,a)
 return d

def read_anns(path):
 d=collections.defaultdict(lambda:{'T':[],'A':[]})
 with open(path,encoding='utf8') as f:
  for line in f:
   p,sec,s,e,txt,cl=line.rstrip('\n').split('\t',5);d[p][sec].append((int(s),int(e)))
 for p in d:
  for sec in d[p]:d[p][sec].sort()
 return d

def get_chunks(tokenizer,text,spans,maxlen,stride):
 enc=tokenizer(text,truncation=True,max_length=maxlen,stride=stride,return_overflowing_tokens=True,return_offsets_mapping=True)
 chunks=[]
 for wi in range(len(enc['input_ids'])):
  offsets=enc['offset_mapping'][wi]; labs=[]
  for start,end in offsets:
   if end<=start:labs.append(-100);continue
   label=0
   for s,e in spans:
    if s<=start and end<=e:
     label=1 if start==s else 2
     break
    if s>start:break
   labs.append(label)
  item={k:enc[k][wi] for k in ('input_ids','attention_mask')}
  if 'token_type_ids' in enc:item['token_type_ids']=enc['token_type_ids'][wi]
  item['offset_mapping']=offsets;item['labels']=labs
  chunks.append(item)
 return chunks

def make_items(articles,anns,tokenizer,maxlen,stride):
 out=[]
 for p,(title,abstract) in articles.items():
  for sec,text in [('T',title),('A',abstract)]:out.extend(get_chunks(tokenizer,text,anns.get(p,{}).get(sec,[]),maxlen,stride))
 return out

class Collate:
 def __init__(self,tokenizer):self.tokenizer=tokenizer
 def __call__(self,items):
  labels=[x['labels'] for x in items]; offsets=[x['offset_mapping'] for x in items]
  feats=[{k:v for k,v in x.items() if k not in ('labels','offset_mapping')} for x in items]
  batch=self.tokenizer.pad(feats,padding=True,return_tensors='pt')
  L=batch['input_ids'].shape[1]; labs=torch.full((len(items),L),-100,dtype=torch.long)
  for i,ys in enumerate(labels):labs[i,:len(ys)]=torch.tensor(ys)
  batch['labels']=labs;batch['offset_mapping']=offsets
  return batch

def goldset(articles,anns):
 return {(p,sec,s,e) for p,x in anns.items() for sec,spans in x.items() for s,e in spans}

def persist_dir(path,prefix):
 if not args.bucket:return
 from google.cloud import storage
 client=storage.Client();bucket=client.bucket(args.bucket)
 for name in os.listdir(path):
  full=os.path.join(path,name)
  if os.path.isfile(full):bucket.blob(prefix+'/'+name).upload_from_filename(full)
 print('persisted checkpoint to gs://'+args.bucket+'/'+prefix,flush=True)

def eval_model(model,items,meta,collate,batch_size,gold):
 loader=DataLoader(items,batch_size=batch_size,shuffle=False,collate_fn=collate)
 all_logits=[]
 model.eval()
 with torch.no_grad():
  for batch in loader:
   offsets=batch.pop('offset_mapping'); batch.pop('labels'); batch={k:v.to(DEVICE) for k,v in batch.items()}
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if DEVICE.type=='cuda' else nullcontext()
   with ctx: logits=model(**batch).logits.float()
   logits=logits.cpu().numpy()
   for i in range(len(offsets)):
    row=[]
    for (s,e),lg in zip(offsets[i],logits[i]):
     if e>s:row.append((s,e,lg))
    all_logits.append(row)
 # Merge overlapping windows at original tokenizer pieces and order by original offset.
 grouped=[[] for _ in meta.docs]
 for chunk_i,docidx in enumerate(meta.chunk_to_doc):grouped[docidx].append(all_logits[chunk_i])
 merged=[]
 for chunk_rows in grouped:
  sums={}; counts=collections.Counter()
  for row in chunk_rows:
   for s,e,lg in row:
    key=(s,e); sums[key]=sums.get(key,np.zeros(3,dtype=np.float64))+lg;counts[key]+=1
  merged.append([(s,e,sums[(s,e)]/counts[(s,e)]) for s,e in sorted(sums)])
 best=(0,None,None,None)
 for bias in [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]:
  pred=set()
  for doc,(p,sec,text) in zip(merged,meta.docs):
   labs=[]
   for s,e,lg in doc:
    v=lg.copy();v[1:]+=bias;labs.append(int(v.argmax()))
   current=None;end=None
   for (s,e,_),lab in zip(doc,labs):
    if lab==1:
     if current is not None:pred.add((p,sec,current,end))
     current=s;end=e
    elif lab==2:
     if current is None:current=s
     end=e
    elif current is not None:
     pred.add((p,sec,current,end));current=None;end=None
   if current is not None:pred.add((p,sec,current,end))
  tp=len(pred&gold);pr=tp/max(1,len(pred));rc=tp/max(1,len(gold));f=2*pr*rc/(pr+rc)
  print('threshold',bias,'pred',len(pred),'tp',tp,'P/R/F1',round(pr,4),round(rc,4),round(f,4),flush=True)
  if f>best[0]:best=(f,bias,pred,merged)
 return best

class Meta:
 def __init__(self,docs,chunks):
  self.docs=docs;self.chunk_to_doc=chunks

def main():
 train=read_articles(os.path.join(args.data,'training.abstracts.txt')); trann=read_anns(os.path.join(args.data,'training.annotations.txt'))
 dev=read_articles(os.path.join(args.data,'development.abstracts.txt')); dvann=read_anns(os.path.join(args.data,'development.annotations.txt'))
 if args.final:
  train.update(dev);trann.update(dvann)
 tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
 print('tokenizer',tok.__class__.__name__,flush=True)
 tr_items=[]
 for p,(title,abstract) in train.items():
  for sec,text in [('T',title),('A',abstract)]:tr_items.extend(get_chunks(tok,text,trann.get(p,{}).get(sec,[]),args.maxlen,args.stride))
 print('training chunks',len(tr_items),'tokens',sum(len(x['input_ids']) for x in tr_items),flush=True)
 dv_items=[];chunk_to_doc=[];docs=[]
 for p,(title,abstract) in dev.items():
  for sec,text in [('T',title),('A',abstract)]:
   docs.append((p,sec,text));chunks=get_chunks(tok,text,dvann.get(p,{}).get(sec,[]),args.maxlen,args.stride)
   dv_items.extend(chunks);chunk_to_doc.extend([len(docs)-1]*len(chunks))
 meta=Meta(docs,chunk_to_doc);collate=Collate(tok)
 train_loader=DataLoader(tr_items,batch_size=args.batch,shuffle=True,collate_fn=collate)
 model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=3,id2label=ID2LABEL,label2id=LABELS)
 model.to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
 total=args.epochs*len(train_loader);sched=get_linear_schedule_with_warmup(opt,int(.08*total),total)
 gold=goldset(dev,dvann);best_f=0
 for epoch in range(args.epochs):
  model.train();loss_sum=0;step=0
  for batch in train_loader:
   batch.pop('offset_mapping');batch={k:v.to(DEVICE) for k,v in batch.items()}
   opt.zero_grad(set_to_none=True)
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if DEVICE.type=='cuda' else nullcontext()
   with ctx:loss=model(**batch).loss
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();loss_sum+=float(loss);step+=1
   if step%50==0:print('epoch',epoch+1,'step',step,'loss',round(loss_sum/step,4),flush=True)
  print('epoch',epoch+1,'trainloss',loss_sum/max(1,step),flush=True)
  if args.final:
   os.makedirs(args.out,exist_ok=True);model.save_pretrained(args.out);tok.save_pretrained(args.out)
   with open(os.path.join(args.out,'best.json'),'w') as out:json.dump({'bias':args.final_bias},out)
   persist_dir(args.out,'checkpoints/final')
   print('saved final epoch',epoch+1,flush=True)
   continue
  f,bias,pred,merged=eval_model(model,dv_items,meta,collate,args.batch,gold)
  if f>best_f:
   best_f=f;os.makedirs(args.out,exist_ok=True);model.save_pretrained(args.out);tok.save_pretrained(args.out)
   with open(os.path.join(args.out,'best.json'),'w') as out:json.dump({'f1':f,'bias':bias},out)
   with open(os.path.join(args.out,'dev_predictions.tsv'),'w') as out:
    for p,sec,s,e in sorted(pred):out.write(f'{p}\t{sec}:{s}:{e}\n')
   persist_dir(args.out,'checkpoints/best')
   print('saved best',best_f,bias,flush=True)
 print('DONE',best_f,flush=True)

if __name__=='__main__':main()
