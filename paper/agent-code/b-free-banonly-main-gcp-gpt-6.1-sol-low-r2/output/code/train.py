import os,json,time,random,argparse
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForTokenClassification,get_linear_schedule_with_warmup
P=argparse.ArgumentParser(); P.add_argument('--all',action='store_true'); P.add_argument('--seed',type=int,default=42); P.add_argument('--epochs',type=int,default=7); P.add_argument('--out',default='run1'); P.add_argument('--predict',action='store_true'); args=P.parse_args()
torch.set_num_threads(6); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
tok=AutoTokenizer.from_pretrained(MODEL)
def read(split):
 ann={}
 if split!='test':
  for line in open('data/'+split+'.annotations.txt'):
   pm,sec,a,b,*_=line.rstrip('\n').split('\t'); ann.setdefault((pm,sec),[]).append((int(a),int(b)))
 docs=[]
 for line in open('data/'+split+'.abstracts.txt'):
  pm,t,a=line.rstrip('\n').split('\t')
  for sec,text in [('T',t),('A',a)]:
   enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); offsets=enc['offset_mapping']; ids=enc['input_ids']; labs=[0]*len(ids)
   spans=sorted(ann.get((pm,sec),[])); k=0
   for i,(s,e) in enumerate(offsets):
    while k<len(spans) and spans[k][1]<=s:k+=1
    if k<len(spans) and s<spans[k][1] and e>spans[k][0]: labs[i]=1 if i==0 or offsets[i-1][1]<=spans[k][0] else 2
   docs.append(dict(pm=pm,sec=sec,text=text,ids=ids,offsets=offsets,labs=labs,gold=spans))
 return docs
def chunks(docs,train=False):
 items=[]
 for d,doc in enumerate(docs):
  n=len(doc['ids']); start=0
  while start<n:
   end=min(start+254,n)
   # all wordpieces receive supervision; overlaps provide more context
   items.append(dict(input_ids=[tok.cls_token_id]+doc['ids'][start:end]+[tok.sep_token_id],labels=[-100]+doc['labs'][start:end]+[-100],d=d,start=start,end=end))
   if end==n:break
   start+=206
 return items
def collate(items):
 n=max(len(x['input_ids']) for x in items)
 return {'input_ids':torch.tensor([x['input_ids']+[0]*(n-len(x['input_ids'])) for x in items]),'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in items]),'labels':torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in items])}
def infer(model,docs):
 items=chunks(docs); sums=[np.zeros((len(d['ids']),3),np.float32) for d in docs]; weights=[np.zeros(len(d['ids']),np.float32) for d in docs]
 model.eval()
 with torch.no_grad():
  for b in range(0,len(items),32):
   part=items[b:b+32]; batch=collate(part); batch.pop('labels')
   with torch.autocast('cuda',dtype=torch.float16): logits=model(**{k:v.cuda() for k,v in batch.items()}).logits.float().cpu().numpy()
   for x,l in zip(part,logits):
    n=x['end']-x['start']; w=np.minimum(np.minimum(np.arange(n)+1,np.arange(n)[::-1]+1),32)/32
    sums[x['d']][x['start']:x['end']]+=l[1:n+1]*w[:,None]; weights[x['d']][x['start']:x['end']]+=w
 return [s/np.maximum(w[:,None],1e-8) for s,w in zip(sums,weights)]
def decode(docs,scores):
 result=set()
 for doc,logit in zip(docs,scores):
  if not len(logit):continue
  # Viterbi: I cannot start a span or follow O
  dp=logit[0].copy(); dp[2]=-1e4; back=[]
  trans=np.zeros((3,3)); trans[0,2]=-1e4
  for l in logit[1:]:
   mat=dp[:,None]+trans; prev=mat.argmax(0); dp=mat.max(0)+l; back.append(prev)
  state=int(dp.argmax()); labels=[state]
  for prev in reversed(back):state=int(prev[state]); labels.append(state)
  labels=labels[::-1]; begin=None; end=None
  for tag,(s,e) in zip(labels,doc['offsets']):
   if tag!=2 and begin is not None:result.add((doc['pm'],doc['sec'],begin,end)); begin=None
   if tag==1 or (tag==2 and begin is None):begin=s
   if tag: end=e
  if begin is not None:result.add((doc['pm'],doc['sec'],begin,end))
 return result
def evaluate(docs,scores):
 pred=decode(docs,scores); gold={(d['pm'],d['sec'],s,e) for d in docs for s,e in d['gold']}; tp=len(pred&gold); f=2*tp/(len(pred)+len(gold));print('EVAL',dict(f1=f,precision=tp/len(pred),recall=tp/len(gold),pred=len(pred)),flush=True);return f
os.makedirs(args.out,exist_ok=True)
if not args.predict:
 train=read('training'); dev=read('development'); train+=dev if args.all else []
 items=chunks(train,True); print('train docs/chunks',len(train),len(items),flush=True)
 model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3).cuda()
 loader=DataLoader(items,batch_size=24,shuffle=True,collate_fn=collate,num_workers=2)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01); sched=get_linear_schedule_with_warmup(opt,int(len(loader)*args.epochs*.08),len(loader)*args.epochs); scaler=torch.amp.GradScaler('cuda'); best=0
 for epoch in range(args.epochs):
  model.train(); loss_sum=0; t=time.time()
  for step,batch in enumerate(loader):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=model(**{k:v.cuda() for k,v in batch.items()}).loss
   scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(opt); scaler.update(); sched.step(); loss_sum+=loss.item()
   if step%100==0:print('epoch',epoch+1,'step',step,'loss',loss_sum/(step+1),'sec',int(time.time()-t),flush=True)
  print('epoch finished',epoch+1,time.time()-t,flush=True)
  if not args.all:
   scores=infer(model,dev); f=evaluate(dev,scores);np.savez_compressed(args.out+'/dev_epoch'+str(epoch+1)+'.npz',*scores)
   if f>best:best=f;model.save_pretrained(args.out+'/best');json.dump({'epoch':epoch+1,'f1':f},open(args.out+'/best.json','w'))
  else:model.save_pretrained(args.out+'/best')
else:model=AutoModelForTokenClassification.from_pretrained(args.out+'/best').cuda()
model=AutoModelForTokenClassification.from_pretrained(args.out+'/best').cuda(); test=read('test'); scores=infer(model,test);np.savez_compressed(args.out+'/test.npz',*scores); pred=decode(test,scores)
with open(args.out+'/predictions.tsv','w') as f:
 for pm,sec,s,e in sorted(pred):f.write(f'{pm}\t{sec}:{s}:{e}\n')
print('DONE',len(pred),flush=True)
