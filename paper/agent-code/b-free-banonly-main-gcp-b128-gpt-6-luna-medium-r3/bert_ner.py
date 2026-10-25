import os, sys, json, math, random, argparse, collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

MODEL='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LABELS={'O':0,'B':1,'I':2}

def read_abstracts(path):
    out={}
    for line in open(path,encoding='utf8'):
        if line.strip():
            pmid,t,a=line.rstrip('\n').split('\t',2); out[pmid]={'T':t,'A':a}
    return out

def read_annotations(path):
    out=collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path,encoding='utf8'):
        if line.strip():
            pmid,s,b,e,txt,cl=line.rstrip('\n').split('\t')
            out[pmid][s].append((int(b),int(e),txt,cl))
    return out

class Chunks(Dataset):
    def __init__(self, articles, annotations, tokenizer, labeled=True):
        self.items=[]; self.refs=[]
        for pmid,sections in articles.items():
            for sec,text in sections.items():
                spans=annotations.get(pmid,{}).get(sec,[]) if labeled else []
                enc=tokenizer(text,return_offsets_mapping=True,truncation=True,max_length=512,stride=96,
                              return_overflowing_tokens=True,padding=False)
                for ci,ids in enumerate(enc['input_ids']):
                    offsets=enc['offset_mapping'][ci]
                    labels=[-100 if b==e else 0 for b,e in offsets]
                    active={}
                    for i,(tb,te) in enumerate(offsets):
                        if tb==te: continue
                        for si,(b,e,*_) in enumerate(spans):
                            if tb<e and te>b:
                                labels[i]=1 if si not in active else 2
                                active[si]=True
                    d={'input_ids':ids,'attention_mask':enc['attention_mask'][ci],
                       'offset_mapping':offsets,'labels':labels}
                    if 'token_type_ids' in enc: d['token_type_ids']=enc['token_type_ids'][ci]
                    self.items.append(d); self.refs.append((pmid,sec,text))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(rows):
    n=max(len(x['input_ids']) for x in rows)
    batch={}
    for key in ('input_ids','attention_mask','token_type_ids'):
        if key in rows[0]:
            vals=[]
            for x in rows:
                v=x[key]; vals.append(v+[0]*(n-len(v)))
            batch[key]=torch.tensor(vals,dtype=torch.long)
    batch['labels']=torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in rows],dtype=torch.long)
    return batch

def spans_from_probs(text,tokens,threshold=.5,b_start=None):
    # each token tuple is (char_start, char_end, pB, pI, pO)
    out=[]; cur=None; end=None
    for b,e,pb,pi,po in tokens:
        chem=pb+pi
        if chem>=threshold:
            if cur is None:
                cur=b
            end=e
        else:
            if cur is not None: out.append((cur,end));cur=end=None
    if cur is not None:out.append((cur,end))
    return out

@torch.no_grad()
def predict(model, tokenizer, articles, batch_size=24, threshold=.5, device=None):
    model.eval(); device=device or next(model.parameters()).device
    data=Chunks(articles,{},tokenizer,labeled=False)
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,collate_fn=collate)
    sums=collections.defaultdict(lambda: collections.defaultdict(lambda:[np.zeros(3,dtype=float),0]))
    k=0
    for batch in loader:
        offsets_batch=[data.items[j]['offset_mapping'] for j in range(k,k+len(batch['input_ids']))]
        refs=data.refs[k:k+len(batch['input_ids'])];k+=len(batch['input_ids'])
        model_inputs={a:b.to(device) for a,b in batch.items() if a in ('input_ids','attention_mask','token_type_ids')}
        probs=torch.softmax(model(**model_inputs).logits,dim=-1).cpu().numpy()
        for (pmid,sec,text),offsets,pr in zip(refs,offsets_batch,probs):
            for (b,e),q in zip(offsets,pr):
                if b==e:continue
                slot=sums[(pmid,sec)][(b,e)];slot[0]+=q;slot[1]+=1
    preds={}
    for (pmid,sec),entries in sums.items():
        toks=[]
        for (b,e),(s,n) in sorted(entries.items()):
            q=s/n;toks.append((b,e,float(q[1]),float(q[2]),float(q[0])))
        preds[(pmid,sec)]=spans_from_probs(articles[pmid][sec],toks,threshold)
    return preds

def goldset(articles,annotations):
    return {(pmid,sec,b,e) for pmid,ss in annotations.items() for sec,spans in ss.items() for b,e,*_ in spans}

def exact_score(gold,pred):
    tp=len(gold&pred);p=tp/len(pred) if pred else 0;r=tp/len(gold) if gold else 0
    return p,r,2*p*r/(p+r) if p+r else 0,tp,len(pred),len(gold)

def predset(pdict):return {(pmid,sec,b,e) for (pmid,sec),spans in pdict.items() for b,e in spans}

def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    tr=read_abstracts(args.train);ta=read_annotations(args.train_ann)
    articles=tr; annotations=ta
    dev=None;da=None
    if args.dev:
        dev=read_abstracts(args.dev);da=read_annotations(args.dev_ann)
    ds=Chunks(articles,annotations,tok,labeled=True)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    id2label={v:k for k,v in LABELS.items()}
    model=AutoModelForTokenClassification.from_pretrained(args.model,num_labels=3,id2label=id2label,label2id=LABELS)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps=len(dl)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=max(1,int(steps*.08)),num_training_steps=steps)
    weights=torch.tensor([1.,args.pos_weight,args.pos_weight],device=device)
    best=0; best_epoch=0
    os.makedirs(args.out,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total=0
        for batch in dl:
            batch={k:v.to(device) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            logits=model(**{k:v for k,v in batch.items() if k!='labels'}).logits
            loss=torch.nn.functional.cross_entropy(logits.view(-1,3),batch['labels'].view(-1),weight=weights,ignore_index=-100)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();total+=float(loss)
        print('epoch',epoch+1,'loss',round(total/len(dl),4),flush=True)
        if dev:
            pr=predict(model,tok,dev,args.batch_size,threshold=.5,device=device)
            gold=goldset(dev,da);pred=predset(pr)
            print('dev@.5',exact_score(gold,pred),flush=True)
            dev_f1=exact_score(gold,pred)[2]
            if dev_f1>best:
                best=dev_f1;best_epoch=epoch+1
                model.save_pretrained(args.out);tok.save_pretrained(args.out)
        else:
            best_epoch=epoch+1
    if not dev:
        model.save_pretrained(args.out);tok.save_pretrained(args.out)
    print('saved',args.out,'best_dev_f1_at_.5',best,'epoch',best_epoch,flush=True)

def infer(args):
    articles=read_abstracts(args.test)
    tok=AutoTokenizer.from_pretrained(args.modeldir,use_fast=True)
    model=AutoModelForTokenClassification.from_pretrained(args.modeldir)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    pd=predict(model,tok,articles,args.batch_size,args.threshold,device)
    with open(args.output,'w',encoding='utf8') as f:
        for pmid in articles:
            for sec in ('T','A'):
                for b,e in sorted(pd.get((pmid,sec),[])):
                    f.write(f'{pmid}\t{sec}:{b}:{e}\n')
    print('wrote',args.output,'mentions',sum(map(len,pd.values())),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train');t.add_argument('--train',required=True);t.add_argument('--train-ann',required=True);t.add_argument('--dev');t.add_argument('--dev-ann');t.add_argument('--model',default=MODEL);t.add_argument('--out',required=True);t.add_argument('--epochs',type=int,default=8);t.add_argument('--batch-size',type=int,default=8);t.add_argument('--lr',type=float,default=2e-5);t.add_argument('--pos-weight',type=float,default=5);t.add_argument('--seed',type=int,default=42);t.set_defaults(fn=train)
    q=sub.add_parser('infer');q.add_argument('--test',required=True);q.add_argument('--modeldir',required=True);q.add_argument('--output',required=True);q.add_argument('--batch-size',type=int,default=24);q.add_argument('--threshold',type=float,default=.5);q.set_defaults(fn=infer)
    a=p.parse_args();a.fn(a)
