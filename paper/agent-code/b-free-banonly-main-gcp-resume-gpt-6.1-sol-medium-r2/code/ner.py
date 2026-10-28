import os, json, random, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from viterbi import viterbi
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

AMP_DTYPE=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

def read_data(root, splits, labeled=True):
    docs=[]; gold=set(); annotations=defaultdict(list)
    for split in splits.split(','):
        if labeled:
            for line in open(f'{root}/{split}.annotations.txt'):
                p,s,a,b,txt,typ=line.rstrip('\n').split('\t'); a=int(a); b=int(b)
                annotations[p,s].append((a,b)); gold.add((p,s,a,b))
        for line in open(f'{root}/{split}.abstracts.txt'):
            p,t,a=line.rstrip('\n').split('\t')
            docs.extend([(p,'T',t),(p,'A',a)])
    return docs, annotations, gold

class Corpus(Dataset):
    def __init__(self,docs,annotations,tok,maxlen=384,stride=96):
        self.docs=docs; self.tokens=[]; self.offsets=[]; self.items=[]
        unaligned=0
        for di,(p,s,text) in enumerate(docs):
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc['input_ids']; offs=enc['offset_mapping']; labs=[0]*len(ids)
            anns=sorted(annotations[p,s]); j=0
            for k,(a,b) in enumerate(offs):
                while j<len(anns) and anns[j][1]<=a: j+=1
                if j<len(anns) and a<anns[j][1] and b>anns[j][0]:
                    labs[k]=1 if k==0 or offs[k-1][1]<=anns[j][0] else 2
            self.tokens.append(ids); self.offsets.append(offs)
            width=maxlen-2
            for start in range(0,max(1,len(ids)),width-stride):
                end=min(start+width,len(ids))
                self.items.append((di,start,end,[tok.cls_token_id]+ids[start:end]+[tok.sep_token_id],[-100]+labs[start:end]+[-100]))
                if end==len(ids): break
        print('Corpus',len(docs),'sections',len(self.items),'windows',flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return (i,self.items[i][3],self.items[i][4])

class LengthBatchSampler:
    def __init__(self,corpus,batch): self.corpus=corpus; self.batch=batch
    def __len__(self): return (len(self.corpus)+self.batch-1)//self.batch
    def __iter__(self):
        order=list(range(len(self.corpus))); random.shuffle(order); batches=[]
        for st in range(0,len(order),self.batch*50):
            block=sorted(order[st:st+self.batch*50],key=lambda i:len(self.corpus.items[i][3]))
            batches.extend([block[j:j+self.batch] for j in range(0,len(block),self.batch)])
        random.shuffle(batches)
        yield from batches

def collate(batch):
    n=max(len(x[1]) for x in batch); n=(n+7)//8*8
    ids=torch.zeros((len(batch),n),dtype=torch.long); masks=torch.zeros_like(ids); labs=torch.full_like(ids,-100)
    for j,(_,x,y) in enumerate(batch):
        ids[j,:len(x)]=torch.tensor(x); masks[j,:len(x)]=1; labs[j,:len(y)]=torch.tensor(y)
    return [x[0] for x in batch],ids,masks,labs

def decode(scores,corpus,bias=0):
    pred=set()
    for di,raw in enumerate(scores):
        if len(raw)==0: continue
        v=raw.astype(np.float32).copy(); v[:,1:]+=bias
        # Constrained BIO Viterbi. I may not follow O.
        tags=viterbi(v)
        p,s,text=corpus.docs[di]; start=None; end=None
        for k,tag in enumerate(tags):
            a,b=corpus.offsets[di][k]
            if tag!=2 and start is not None:
                if start<end: pred.add((p,s,start,end))
                start=None
            if tag==1: start=a; end=b
            elif tag==2: end=b
        if start is not None and start<end: pred.add((p,s,start,end))
    return pred

def metric(pred,gold):
    hit=len(pred&gold); p=hit/max(1,len(pred)); r=hit/max(1,len(gold)); f=2*p*r/max(1e-10,p+r)
    return {'f1':f,'precision':p,'recall':r,'pred':len(pred),'gold':len(gold),'correct':hit}

@torch.inference_mode()
def infer(model,corpus,batch):
    model.eval(); sums=[np.zeros((len(x),3),np.float32) for x in corpus.tokens]; weights=[np.zeros(len(x),np.float32) for x in corpus.tokens]
    dl=DataLoader(corpus,batch_size=batch,sampler=sorted(range(len(corpus)),key=lambda i:len(corpus.items[i][3])),collate_fn=collate,num_workers=2)
    for ii,ids,mask,labs in dl:
        with torch.autocast('cuda',dtype=AMP_DTYPE): out=model(input_ids=ids.cuda(),attention_mask=mask.cuda()).logits
        out=out.float().cpu().numpy()
        for j,i in enumerate(ii):
            di,a,b,_,_=corpus.items[i]; n=b-a
            # Central context receives more weight than window boundaries.
            w=np.minimum(np.minimum(np.arange(n)+1,np.arange(n)[::-1]+1),64).astype(np.float32)
            sums[di][a:b]+=out[j,1:n+1]*w[:,None]; weights[di][a:b]+=w
    return [x/np.maximum(w[:,None],1) for x,w in zip(sums,weights)]

def save_scores(path,scores,corpus):
    lengths=np.array([len(s) for s in scores]); flat=np.concatenate(scores).astype(np.float32)
    np.savez_compressed(path,logits=flat,lengths=lengths,offsets=np.array([pair for offs in corpus.offsets for pair in offs],dtype=np.int32))

def write_preds(path,pred):
    with open(path,'w') as f:
        for p,s,a,b in sorted(pred): f.write(f'{p}\t{s}:{a}:{b}\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['train','predict'],default='train'); ap.add_argument('--data',default='/workspace/data'); ap.add_argument('--train',default='training'); ap.add_argument('--dev',default='development'); ap.add_argument('--out',required=True); ap.add_argument('--model',default=MODEL); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch',type=int,default=24); ap.add_argument('--lr',type=float,default=3e-5); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--maxlen',type=int,default=384); ap.add_argument('--stride',type=int,default=96); ap.add_argument('--split',default='test'); ap.add_argument('--bias',type=float,default=0); args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32=True
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    if args.mode=='predict':
        docs,ann,gold=read_data(args.data,args.split,labeled=args.split!='test'); corpus=Corpus(docs,ann,tok,args.maxlen,args.stride)
        model=AutoModelForTokenClassification.from_pretrained(args.model).cuda(); scores=infer(model,corpus,args.batch*2); save_scores(f'{args.out}/{args.split}_scores.npz',scores,corpus); pred=decode(scores,corpus,args.bias); write_preds(f'{args.out}/{args.split}_predictions.tsv',pred)
        if gold: print(metric(pred,gold),flush=True)
        return
    docs,ann,gold=read_data(args.data,args.train); train=Corpus(docs,ann,tok,args.maxlen,args.stride)
    if args.dev!='none':
        vd,va,vg=read_data(args.data,args.dev); dev=Corpus(vd,va,tok,args.maxlen,args.stride)
    model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=3,id2label={0:'O',1:'B-CHEM',2:'I-CHEM'},label2id={'O':0,'B-CHEM':1,'I-CHEM':2}).cuda()
    groups=[{'params':[p for n,p in model.named_parameters() if not ('bias' in n or 'LayerNorm.weight' in n)],'weight_decay':.01},{'params':[p for n,p in model.named_parameters() if ('bias' in n or 'LayerNorm.weight' in n)],'weight_decay':0}]
    opt=torch.optim.AdamW(groups,lr=args.lr,fused=True)
    dl=DataLoader(train,batch_sampler=LengthBatchSampler(train,args.batch),collate_fn=collate,num_workers=2,pin_memory=True)
    scaler=torch.amp.GradScaler('cuda',enabled=AMP_DTYPE==torch.float16)
    sched=get_linear_schedule_with_warmup(opt,int(.08*len(dl)*args.epochs),len(dl)*args.epochs); best=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]; ts=time.time()
        for step,(ii,ids,mask,labs) in enumerate(dl):
            with torch.autocast('cuda',dtype=AMP_DTYPE): loss=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),labels=labs.cuda()).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); losses.append(loss.item())
            if (step+1)%100==0: print('epoch',epoch,'step',step+1,'/',len(dl),'loss',round(np.mean(losses[-100:]),5),'seconds',round(time.time()-ts),flush=True)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'train_seconds':time.time()-ts}
        if args.dev!='none':
            scores=infer(model,dev,args.batch*2); row.update(metric(decode(scores,dev),vg)); print(row,flush=True)
            save_scores(f'{args.out}/dev_epoch{epoch}.npz',scores,dev)
            if row['f1']>best:
                best=row['f1']; model.save_pretrained(f'{args.out}/best'); tok.save_pretrained(f'{args.out}/best'); write_preds(f'{args.out}/dev_predictions.tsv',decode(scores,dev))
        else:
            print(row,flush=True)
        history.append(row); Path(f'{args.out}/history.json').write_text(json.dumps(history,indent=2))
        # Store final checkpoints, enabling a continuation or ensemble.
        if epoch>=3:
            model.save_pretrained(f'{args.out}/epoch{epoch}'); tok.save_pretrained(f'{args.out}/epoch{epoch}')
    print('DONE',flush=True)

if __name__=='__main__': main()
