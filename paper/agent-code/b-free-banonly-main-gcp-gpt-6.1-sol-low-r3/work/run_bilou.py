import os, argparse, json, random, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from collections import defaultdict

MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
def read_data(root, split):
    docs=[]; gold=defaultdict(list)
    ann=os.path.join(root,split+'.annotations.txt')
    if os.path.exists(ann):
        for line in open(ann):
            p,s,a,b,*_=line.rstrip('\n').split('\t'); gold[(p,s)].append((int(a),int(b)))
    for line in open(os.path.join(root,split+'.abstracts.txt')):
        p,t,a=line.rstrip('\n').split('\t')
        for s,text in [('T',t),('A',a)]: docs.append((p,s,text,sorted(gold[(p,s)])))
    return docs

def encode(docs,tok):
    chunks=[]; offsets=[]
    for d,(p,s,text,gold) in enumerate(docs):
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=enc['input_ids']; off=enc['offset_mapping']; offsets.append(off)
        labels=[]; j=0
        for a,b in off:
            while j<len(gold) and gold[j][1]<=a:j+=1
            if j<len(gold) and a<gold[j][1] and b>gold[j][0]:
                labels.append(1 if not labels or a<=gold[j][0] or (len(labels)>0 and off[len(labels)-1][1]<=gold[j][0]) else 2)
            else:labels.append(0)
        labels=[(4 if i+1==len(labels) or labels[i+1]!=2 else 1) if lab==1 else (3 if i+1==len(labels) or labels[i+1]!=2 else 2) if lab==2 else 0 for i,lab in enumerate(labels)]
        for start in range(0,len(ids),256):
            end=min(start+382,len(ids))
            chunks.append(dict(input_ids=[tok.cls_token_id]+ids[start:end]+[tok.sep_token_id],labels=[-100]+labels[start:end]+[-100],doc=d,start=start))
            if end==len(ids):break
    return chunks,offsets

class BucketSampler:
    def __init__(self,chunks,batch=16):self.chunks=chunks;self.batch=batch
    def __len__(self):return math.ceil(len(self.chunks)/self.batch)
    def __iter__(self):
        indices=list(range(len(self.chunks)));random.shuffle(indices);batches=[]
        for start in range(0,len(indices),512):
            pool=sorted(indices[start:start+512],key=lambda i:len(self.chunks[i]['input_ids']))
            batches.extend([pool[j:j+self.batch] for j in range(0,len(pool),self.batch)])
        random.shuffle(batches);yield from batches

def collate(items):
    n=max(len(x['input_ids']) for x in items)
    return {'input_ids':torch.tensor([x['input_ids']+[0]*(n-len(x['input_ids'])) for x in items]),'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in items]),'labels':torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in items])}

def infer(model,chunks,offsets,batch=32):
    model.eval(); sums=[np.zeros((len(o),3),np.float32) for o in offsets]; weights=[np.zeros(len(o),np.float32) for o in offsets]
    with torch.no_grad():
        for start in range(0,len(chunks),batch):
            items=chunks[start:start+batch]; b=collate(items)
            with torch.autocast('cuda',dtype=torch.float16):out=model(input_ids=b['input_ids'].cuda(),attention_mask=b['attention_mask'].cuda()).logits.float().softmax(-1).cpu().numpy()
            out=np.stack([out[:,:,0],out[:,:,1]+out[:,:,4],out[:,:,2]+out[:,:,3]],axis=-1)
            for i,c in enumerate(items):
                n=len(c['input_ids'])-2; pos=c['start']; w=np.minimum(np.arange(n)+1,np.arange(n,0,-1)); w=np.minimum(w,64).astype(np.float32)
                sums[c['doc']][pos:pos+n]+=out[i,1:n+1]*w[:,None]; weights[c['doc']][pos:pos+n]+=w
    return [x/np.maximum(w[:,None],1) for x,w in zip(sums,weights)]

def decode(probs,docs,offsets):
    pred=set()
    for pr,doc,off in zip(probs,docs,offsets):
        p,s,text,_=doc; active=None
        for label,(a,b) in zip(pr.argmax(-1),off):
            if label==0 or label==1:
                if active is not None:pred.add((p,s,active[0],active[1]));active=None
            if label in (1,2):
                if active is None:active=[a,b]
                else:active[1]=b
        if active is not None:pred.add((p,s,*active))
    return pred

def score(pred,docs):
    gold={(p,s,a,b) for p,s,t,spans in docs for a,b in spans}; correct=len(pred & gold); precision=correct/max(len(pred),1);recall=correct/max(len(gold),1)
    return dict(f1=2*precision*recall/max(precision+recall,1e-9),precision=precision,recall=recall,pred=len(pred),gold=len(gold))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--full',action='store_true');ap.add_argument('--predict',nargs='*');ap.add_argument('--model',default=MODEL);ap.add_argument('--lr',type=float,default=3e-5);args=ap.parse_args()
    torch.set_num_threads(6);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);os.makedirs(args.out,exist_ok=True)
    json.dump(vars(args),open(args.out+'/run_config.json','w'),indent=2)
    tok=AutoTokenizer.from_pretrained(args.model)
    if args.predict:
        docs=read_data(args.data,'test');chunks,off=encode(docs,tok);total=None
        for path in args.predict:
            model=AutoModelForTokenClassification.from_pretrained(path).cuda();pr=infer(model,chunks,off)
            if total is None:total=pr
            else:
                for a,b in zip(total,pr):a+=b
            del model;torch.cuda.empty_cache()
        pred=decode(total,docs,off)
        with open(args.out+'/predictions.tsv','w') as f:
            for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')
        np.savez_compressed(args.out+'/test_probs.tmp.npz',*total);os.replace(args.out+'/test_probs.tmp.npz',args.out+'/test_probs.npz');print('Predictions',len(pred),flush=True);return
    train=read_data(args.data,'training');dev=read_data(args.data,'development')
    if args.full:train+=dev
    chunks,_=encode(train,tok);dc,do=encode(dev,tok)
    model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=5).cuda()
    loader=DataLoader(chunks,batch_sampler=BucketSampler(chunks),collate_fn=collate)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    sched=get_linear_schedule_with_warmup(optim,int(len(loader)*args.epochs*.06),len(loader)*args.epochs)
    scaler=torch.amp.GradScaler('cuda');best=0
    for ep in range(args.epochs):
        model.train();loss_sum=0;t=time.time()
        for step,b in enumerate(loader):
            b={k:v.cuda() for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.float16):loss=model(**b).loss
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optim);scaler.update();sched.step();optim.zero_grad(set_to_none=True);loss_sum+=loss.item()
            if step%100==0:print('epoch',ep+1,'step',step,'/',len(loader),'loss',loss_sum/(step+1),'seconds',time.time()-t,flush=True)
        if not args.full:
            pr=infer(model,dc,do);res=score(decode(pr,dev,do),dev);print('EVAL',ep+1,json.dumps(res),flush=True)
            np.savez_compressed(args.out+f'/dev_ep{ep+1}.npz',*pr)
            if res['f1']>best:
                best=res['f1'];model.save_pretrained(args.out+'/best');np.savez_compressed(args.out+'/dev_probs.npz',*pr)
        else:
            model.save_pretrained(args.out+'/best')
            if ep+1>=args.epochs-2:model.save_pretrained(args.out+f'/ep{ep+1}')
        print('epoch complete',ep+1,'seconds',time.time()-t,flush=True)
if __name__=='__main__':main()
