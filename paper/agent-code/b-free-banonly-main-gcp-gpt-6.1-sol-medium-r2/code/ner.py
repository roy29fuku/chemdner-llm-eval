import os, json, time, random, argparse, pickle
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from numba import njit
MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

def read_data(root, tokenizer):
    import hashlib
    fingerprint=hashlib.md5(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()[:10]
    cache=os.path.join(root,'cache-'+fingerprint+'.pkl')
    if len(tokenizer)==30522 and not os.path.exists(cache) and os.path.exists(os.path.join(root,'cache.pkl')):
        return pickle.load(open(os.path.join(root,'cache.pkl'),'rb'))
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    data={}
    for split in ['training','development','test']:
        anns={}
        if split!='test':
            for line in open(os.path.join(root,split+'.annotations.txt')):
                p,s,a,b,*_=line.rstrip('\n').split('\t'); anns.setdefault((p,s),[]).append((int(a),int(b)))
        docs=[]; imperfect=0
        for line in open(os.path.join(root,split+'.abstracts.txt')):
            p,title,abstract=line.rstrip('\n').split('\t')
            ids=[]; offsets=[]; labels=[]; gold=[]
            for sec,text in [('T',title),('A',abstract)]:
                tok=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                off=tok['offset_mapping']; lab=[0]*len(off)
                for a,b in anns.get((p,sec),[]):
                    indices=[i for i,(c,d) in enumerate(off) if d>a and c<b]
                    gold.append((sec,a,b))
                    if not indices: continue
                    if off[indices[0]][0]!=a or off[indices[-1]][1]!=b: imperfect+=1
                    if len(indices)==1:lab[indices[0]]=4
                    else:
                        lab[indices[0]]=1;lab[indices[-1]]=3
                        for i in indices[1:-1]:lab[i]=2
                ids+=tok['input_ids'];offsets += [(sec,a,b) for a,b in off];labels+=lab
                if sec=='T': ids.append(tokenizer.sep_token_id);offsets.append(('X',0,0));labels.append(-100)
            docs.append(dict(pmid=p,ids=ids,offsets=offsets,labels=labels,gold=set(gold),texts={'T':title,'A':abstract}))
        data[split]=docs
        print('Loaded',split,len(docs),'imperfect boundaries',imperfect,flush=True)
    pickle.dump(data,open(cache,'wb'))
    return data

def add_types(data,root):
    classes={'TRIVIAL':0,'SYSTEMATIC':1,'ABBREVIATION':2,'FORMULA':3,'FAMILY':4,'IDENTIFIER':5,'MULTIPLE':6,'NO CLASS':7}
    for split in ('training','development'):
        annotations={}
        for line in open(os.path.join(root,split+'.annotations.txt')):
            p,sec,a,b,text,cl=line.rstrip('\n').split('\t')
            annotations.setdefault(p,[]).append((sec,int(a),int(b),classes[cl]))
        for doc in data[split]:
            labs=doc['labels'].copy()
            for sec,a,b,cl in annotations.get(doc['pmid'],[]):
                for i,(s,x,y) in enumerate(doc['offsets']):
                    if s==sec and y>a and x<b and labs[i]>0:
                        labs[i]=1+4*cl+(doc['labels'][i]-1)
            doc['labels']=labs

class Chunks(Dataset):
    def __init__(self,docs,size=256,core=192):
        self.docs=docs;self.chunks=[];self.size=size;self.core=core
        for d,doc in enumerate(docs):
            n=len(doc['ids']);ctx=(size-2-core)//2
            for start in range(0,n,core):
                lo=max(0,start-ctx);hi=min(n,start+core+ctx)
                self.chunks.append((d,lo,hi,start,min(start+core,n)))
    def __len__(self):return len(self.chunks)
    def __getitem__(self,i):
        d,lo,hi,a,b=self.chunks[i];doc=self.docs[d]
        ids=[2]+doc['ids'][lo:hi]+[3]
        labels=[-100]+[x if a<=j<b else -100 for j,x in enumerate(doc['labels'][lo:hi],lo)]+[-100]
        return ids,labels,(d,lo,hi)

def collate(batch):
    n=((max(len(x[0]) for x in batch)+7)//8)*8
    ids=torch.zeros((len(batch),n),dtype=torch.long);labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    meta=[]
    for i,(a,b,c) in enumerate(batch):
        ids[i,:len(a)]=torch.tensor(a);labels[i,:len(a)]=torch.tensor(b);mask[i,:len(a)]=1;meta.append(c)
    return {'input_ids':ids,'attention_mask':mask,'labels':labels},meta

@njit
def path_decode(scores,bias):
    n=scores.shape[0]; back=np.zeros((n,5),np.int32);dp=np.zeros(5,np.float32)
    for j in range(5):dp[j]=scores[0,j]+(bias if j==0 else 0)+(0 if j in (0,1,4) else -10000)
    for i in range(1,n):
        new=np.empty(5,np.float32)
        for j in range(5):
            best=-1e20;arg=0
            for k in range(5):
                allowed=((k==0 or k==3 or k==4) and (j==0 or j==1 or j==4)) or ((k==1 or k==2) and (j==2 or j==3))
                v=dp[k]+(0 if allowed else -10000)
                if v>best:best=v;arg=k
            new[j]=best+scores[i,j]+(bias if j==0 else 0);back[i,j]=arg
        dp=new
    best=-1e20;last=0
    for j in (0,3,4):
        if dp[j]>best:best=dp[j];last=j
    out=np.zeros(n,np.int32);out[n-1]=last
    for i in range(n-1,0,-1):out[i-1]=back[i,out[i]]
    return out

def decode(doc,scores,bias=0):
    result=set()
    for sec in ('T','A'):
        inds=[i for i,x in enumerate(doc['offsets']) if x[0]==sec]
        if not inds:continue
        tags=path_decode(scores[inds].astype(np.float32),bias);start=None
        for i,tag in zip(inds,tags):
            _,a,b=doc['offsets'][i]
            if tag in (1,4):start=a
            if tag in (3,4) and start is not None:
                result.add((sec,start,b));start=None
    return result

def evaluate(docs,logits,bias=0):
    tp=fp=fn=0
    for d,s in zip(docs,logits):
        pred=decode(d,s,bias);gold=d['gold'];tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return dict(f1=2*p*r/max(p+r,1e-10),precision=p,recall=r,tp=tp,fp=fp,fn=fn)

@torch.no_grad()
def infer(model,docs,batch=32,size=256):
    model.eval();ds=Chunks(docs,size=size,core=size-66)
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    scores=[np.zeros((len(d['ids']),5),np.float32) for d in docs];weights=[np.zeros(len(d['ids']),np.float32) for d in docs]
    for inputs,meta in loader:
        inputs={k:v.cuda(non_blocking=True) for k,v in inputs.items() if k!='labels'}
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs).logits
        if out.shape[-1]>5:
            out=torch.cat([out[...,0:1],torch.logsumexp(out[...,1:].float().reshape(*out.shape[:-1],-1,4),dim=-2)],dim=-1)
        out=out.float().cpu().numpy()
        for i,(d,lo,hi) in enumerate(meta):
            n=hi-lo;w=np.minimum(np.minimum(np.arange(n)+1,n-np.arange(n)),32).astype(np.float32)
            scores[d][lo:hi]+=out[i,1:n+1]*w[:,None];weights[d][lo:hi]+=w
    for s,w in zip(scores,weights):s/=np.maximum(w[:,None],1)
    return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['train','predict'],default='train');ap.add_argument('--checkpoint');ap.add_argument('--predict-test-only',action='store_true');ap.add_argument('--model',default=MODEL);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--save-epoch',type=int,default=0);ap.add_argument('--stop-after',type=int,default=0);ap.add_argument('--batch',type=int,default=24);ap.add_argument('--lr',type=float,default=3e-5);ap.add_argument('--size',type=int,default=256);ap.add_argument('--full',action='store_true');ap.add_argument('--typed',action='store_true');ap.add_argument('--holdout',type=int,default=500);ap.add_argument('--bias',type=float,default=0);args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True);json.dump(vars(args),open(os.path.join(args.out,'arguments.json'),'w'),indent=2);torch.set_num_threads(6)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model);data=read_data(args.data,tokenizer)
    if args.mode=='predict':
        model=AutoModelForTokenClassification.from_pretrained(args.checkpoint).cuda()
        for split in (['test'] if args.predict_test_only else ['development','test']):
            logits=infer(model,data[split],args.batch,args.size)
            pickle.dump(logits,open(os.path.join(args.out,split+'.logits.pkl'),'wb'))
            with open(os.path.join(args.out,split+'.predictions.tsv'),'w') as f:
                for d,s in zip(data[split],logits):
                    for sec,a,b in sorted(decode(d,s,args.bias)):f.write(f"{d['pmid']}\t{sec}:{a}:{b}\n")
            if split!='test':print(split,evaluate(data[split],logits,args.bias),flush=True)
        return
    if args.typed:add_types(data,args.data)
    train=data['training'];dev=data['development']
    if args.full:
        rng=random.Random(1234);dev=dev.copy();rng.shuffle(dev)
        train=train+dev[args.holdout:];dev=dev[:args.holdout] if args.holdout else dev[:500]
    ds=Chunks(train,args.size,args.size-66)
    loader=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=2,collate_fn=collate,pin_memory=True)
    model=AutoModelForTokenClassification.from_pretrained(args.checkpoint or args.model,num_labels=33 if args.typed else 5,attn_implementation='sdpa').cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01,fused=True)
    total=len(loader)*args.epochs;scheduler=get_linear_schedule_with_warmup(opt,int(total*.1),total)
    print('Train docs/chunks/steps',len(train),len(ds),total,flush=True);best=-1;t0=time.time();averaged=None;average_count=0
    for epoch in range(args.epochs):
        model.train();losses=[]
        for step,(inputs,_) in enumerate(loader):
            inputs={k:v.cuda(non_blocking=True) for k,v in inputs.items()};opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();scheduler.step();losses.append(loss.item())
            if step%100==0:print('epoch',epoch+1,'step',step,'loss',round(np.mean(losses[-100:]),4),'elapsed',round(time.time()-t0),flush=True)
        if epoch>=args.epochs-3:
            state={k:v.detach().float().cpu() for k,v in model.state_dict().items()}
            if averaged is None:averaged=state
            else:
                for k in averaged:averaged[k]+=state[k]
            average_count+=1
        logits=infer(model,dev,args.batch,args.size)
        results=[(evaluate(dev,logits,b),b) for b in [-1,-.5,0,.5,1,1.5]];result,bias=max(results,key=lambda x:x[0]['f1'])
        print('EVAL',epoch+1,result,'bias',bias,'elapsed',round(time.time()-t0),flush=True)
        with open(os.path.join(args.out,'metrics.jsonl'),'a') as f:f.write(json.dumps(dict(epoch=epoch+1,**result,bias=bias))+'\n')
        if epoch+1==args.save_epoch:
            model.save_pretrained(os.path.join(args.out,'chosen'));tokenizer.save_pretrained(os.path.join(args.out,'chosen'))
        if result['f1']>best:
            best=result['f1'];model.save_pretrained(os.path.join(args.out,'best'));tokenizer.save_pretrained(os.path.join(args.out,'best'))
            pickle.dump(logits,open(os.path.join(args.out,'dev.logits.pkl'),'wb'));pickle.dump(dev,open(os.path.join(args.out,'dev.docs.pkl'),'wb'))
            json.dump(dict(epoch=epoch+1,**result,bias=bias),open(os.path.join(args.out,'best.json'),'w'))
        if args.stop_after and epoch+1>=args.stop_after:break
    model.save_pretrained(os.path.join(args.out,'last'));tokenizer.save_pretrained(os.path.join(args.out,'last'))
    if averaged is not None:
        for k in averaged:averaged[k]/=average_count
        model.load_state_dict(averaged)
        model.save_pretrained(os.path.join(args.out,'average'));tokenizer.save_pretrained(os.path.join(args.out,'average'))
        logits=infer(model,dev,args.batch,args.size)
        results=[(evaluate(dev,logits,b),b) for b in [-1,-.5,0,.5,1,1.5]];result,bias=max(results,key=lambda x:x[0]['f1'])
        print('AVERAGE_EVAL',result,'bias',bias,flush=True)
        pickle.dump(logits,open(os.path.join(args.out,'average.logits.pkl'),'wb'))
        json.dump(dict(**result,bias=bias),open(os.path.join(args.out,'average.json'),'w'))
    print('DONE',best,flush=True)
if __name__=='__main__':main()
