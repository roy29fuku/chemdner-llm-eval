import os,json,random,time,argparse
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForTokenClassification,get_linear_schedule_with_warmup
from collections import defaultdict
from constrained import viterbi
p=argparse.ArgumentParser();p.add_argument('--out',default='run1');p.add_argument('--epochs',type=int,default=8);p.add_argument('--full',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--batch',type=int,default=16);p.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');p.add_argument('--unsorted-inference',action='store_true');p.add_argument('--random-batches',action='store_true');p.add_argument('--unscaled-loss',action='store_true');p.add_argument('--raw-decoder',action='store_true');args=p.parse_args()
os.makedirs(args.out,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(6)
tok=AutoTokenizer.from_pretrained(args.model); tok.save_pretrained(args.out)
def read(split):
 anns=defaultdict(list)
 fn=f'data/{split}.annotations.txt'
 if os.path.exists(fn):
  for l in open(fn):
   x=l.rstrip('\n').split('\t');anns[(x[0],x[1])].append((int(x[2]),int(x[3])))
 docs=[];chunks=[]
 for l in open(f'data/{split}.abstracts.txt'):
  pm,title,abstract=l.rstrip('\n').split('\t')
  for sec,txt in [('T',title),('A',abstract)]:
   enc=tok(txt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offs=enc['offset_mapping'];labels=[0]*len(ids)
   for s,e in anns[(pm,sec)]:
    inds=[i for i,(a,b) in enumerate(offs) if a>=s and b<=e and b>a]
    for j,i in enumerate(inds):labels[i]=1 if j==0 else 2
   d=len(docs);docs.append((pm,sec,txt,offs,anns[(pm,sec)]))
   for start in range(0,len(ids),256):
    end=min(start+382,len(ids)); chunks.append((d,start,ids[start:end],labels[start:end]));
    if end==len(ids):break
 return docs,chunks
train,tc=read('training');dev,dc=read('development');test,xc=read('test')
if args.full:tc+= [(d+len(train),s,ids,lab) for d,s,ids,lab in dc]
def collate(rows):
 n=max(len(r[2]) for r in rows)+2;ids=[];labs=[];mask=[]
 for d,s,i,l in rows:
  pad=n-len(i)-2;ids.append([tok.cls_token_id]+i+[tok.sep_token_id]+[tok.pad_token_id]*pad);labs.append([-100]+l+[-100]*(pad+1));mask.append([1]*(len(i)+2)+[0]*pad)
 return {k:torch.tensor(v).cuda() for k,v in [('input_ids',ids),('attention_mask',mask),('labels',labs)]}
model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=3).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
class Bucket:
 def __init__(self,rows,batch):self.rows=rows;self.batch=batch
 def __len__(self):return (len(self.rows)+self.batch-1)//self.batch
 def __iter__(self):
  inds=list(range(len(self.rows)));random.shuffle(inds);batches=[]
  for i in range(0,len(inds),self.batch*32):
   bucket=sorted(inds[i:i+self.batch*32],key=lambda j:len(self.rows[j][2]))
   batches.extend(bucket[k:k+self.batch] for k in range(0,len(bucket),self.batch))
  random.shuffle(batches);yield from batches
loader=DataLoader(tc,batch_size=args.batch,shuffle=True,collate_fn=collate) if args.random_batches else DataLoader(tc,batch_sampler=Bucket(tc,args.batch),collate_fn=collate)
sched=get_linear_schedule_with_warmup(opt,int(len(loader)*args.epochs*.06),len(loader)*args.epochs)
expected_tokens=sum(len(r[2]) for r in tc)/len(tc)*args.batch
scaler=torch.amp.GradScaler('cuda');best=0

def infer(docs,chunks):
 chunks=chunks if args.unsorted_inference else sorted(chunks,key=lambda r:len(r[2]));model.eval();acc=[np.zeros((len(d[3]),3),np.float32) for d in docs];cnt=[np.zeros(len(d[3]),np.float32) for d in docs]
 with torch.no_grad():
  for st in range(0,len(chunks),32):
   rows=chunks[st:st+32];b=collate(rows);b.pop('labels')
   with torch.autocast('cuda',dtype=torch.float16): pred=model(**b).logits.float().cpu().numpy()
   for row,pr in zip(rows,pred):
    d,s,ids,l=row;n=len(ids);w=np.minimum(np.arange(n)+1,np.arange(n,0,-1)).clip(1,64)/64
    acc[d][s:s+n]+=pr[1:n+1]*w[:,None];cnt[d][s:s+n]+=w
 return [a/np.maximum(c[:,None],1e-8) for a,c in zip(acc,cnt)]
def decode(docs,logits,path=None):
 out=set()
 for doc,pr in zip(docs,logits):
  pm,sec,txt,offs,gold=doc;tags=pr.argmax(-1) if args.raw_decoder else viterbi(pr,offs,txt,word=True,valid=True);start=None;end=None
  for tag,(a,b) in zip(tags,offs):
   if tag==0 or tag==1:
    if start is not None:out.add((pm,sec,start,end));start=None
   if tag:
    if start is None:start=a
    end=b
  if start is not None:out.add((pm,sec,start,end))
 if path:
  with open(path,'w') as f:
   for pm,sec,s,e in sorted(out):f.write(f'{pm}\t{sec}:{s}:{e}\n')
 return out
for epoch in range(args.epochs):
 model.train();losses=[];t=time.time()
 for step,b in enumerate(loader):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):
   loss=model(**b).loss
   if not args.unscaled_loss:loss=loss*(b['labels'].ne(-100).sum()/expected_tokens)
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();sched.step();losses.append(loss.item())
  if (step+1)%100==0:print("progress",epoch+1,step+1,"/",len(loader),"seconds",round(time.time()-t),flush=True)
 if not args.full:
  logits=infer(dev,dc);pred=decode(dev,logits);gold={(pm,s,a,b) for pm,s,txt,off,ann in dev for a,b in ann};hit=len(pred&gold);f=2*hit/(len(pred)+len(gold));print(epoch+1,'loss',np.mean(losses),'P',hit/len(pred),'R',hit/len(gold),'F1',f,'seconds',time.time()-t,flush=True)
  if f>best:
   best=f;model.save_pretrained(args.out);np.savez_compressed(args.out+'/dev_logits.npz',*logits);decode(dev,logits,args.out+'/dev.tsv')
 else:
  print(epoch+1,'loss',np.mean(losses),'seconds',time.time()-t,flush=True);model.save_pretrained(args.out)
model=AutoModelForTokenClassification.from_pretrained(args.out).cuda();logits=infer(test,xc);np.savez_compressed(args.out+'/test_logits.npz',*logits);decode(test,logits,args.out+'/test.tsv')
print('FINISHED',args.out,flush=True)
