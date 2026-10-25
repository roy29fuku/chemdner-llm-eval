"""PubMed-domain supervised token classifier for exact chemical mention spans.

Backbone is an unlabeled masked-language-model checkpoint; only the supplied
training/dev annotations supervise chemical recognition.
"""
import os, re, json, math, random, argparse, collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
DATA=os.environ.get('CHEM_DATA','/workspace/data')
MAXLEN=256
STRIDE=64
LABELS={'O':0,'B':1,'I':2}

def load(split):
    texts=[]; keys=[]
    with open(f'{DATA}/{split}.abstracts.txt',encoding='utf8') as f:
        for line in f:
            pmid,title,abstract=line.rstrip('\n').split('\t',2)
            texts.extend([title,abstract]); keys.extend([(pmid,'T'),(pmid,'A')])
    anns=collections.defaultdict(list)
    path=f'{DATA}/{split}.annotations.txt'
    if os.path.exists(path):
        with open(path,encoding='utf8') as f:
            for line in f:
                p,s,a,b,txt,cl=line.rstrip('\n').split('\t')
                anns[(p,s)].append((int(a),int(b)))
    return texts,keys,anns

def encode(texts,keys,anns,tokenizer):
    enc=tokenizer(texts,truncation=True,max_length=MAXLEN,stride=STRIDE,
                  return_overflowing_tokens=True,return_offsets_mapping=True,
                  padding=False)
    examples=[]
    for j,off in enumerate(enc['offset_mapping']):
        key=keys[enc['overflow_to_sample_mapping'][j]]
        labels=[]; last=None
        for a,b in off:
            if a==b:
                labels.append(-100); continue
            found=None
            for idx,(x,y) in enumerate(anns.get(key,())):
                if a<y and b>x:
                    found=idx; break
            if found is None: labels.append(0);last=None
            else:
                labels.append(2 if found==last else 1);last=found
        examples.append({'input_ids':enc['input_ids'][j],
                         'attention_mask':enc['attention_mask'][j],
                         'labels':labels,'key':key,'offsets':off})
    return examples

class Collator:
    def __init__(self,pad):self.pad=pad
    def __call__(self,rows):
        n=max(len(r['input_ids']) for r in rows)
        def pad(seq,value):return seq+[value]*(n-len(seq))
        return {k:torch.tensor([pad(r[k],v) for r in rows],dtype=torch.long)
                for k,v in [('input_ids',self.pad),('attention_mask',0),('labels',-100)]}

def chunks(xs,n):
    for i in range(0,len(xs),n):yield xs[i:i+n]

def exact_from_labels(examples,logits,bias=0):
    probs=collections.defaultdict(lambda:[np.zeros(3,dtype=np.float64),0])
    for ex,lg in zip(examples,logits):
        key=ex['key']
        for (a,b),z in zip(ex['offsets'],lg):
            if a==b:continue
            val=probs[(key,a,b)];val[0]+=z;val[1]+=1
    by_section=collections.defaultdict(list)
    for (key,a,b),(z,n) in probs.items():by_section[key].append((a,b,z/n))
    pred=set()
    for key,toks in by_section.items():
        toks.sort(key=lambda x:(x[0],x[1]))
        start=end=None
        for a,b,z in toks:
            z=z.copy();z[1:]+=bias
            tag=int(np.argmax(z))
            if tag==1:
                if start is not None:pred.add((*key,start,end))
                start,end=a,b
            elif tag==2:
                if start is None:start=a
                end=b
            else:
                if start is not None:pred.add((*key,start,end));start=end=None
        if start is not None:pred.add((*key,start,end))
    return pred

def score(pred,keys,anns):
    gold={(p,s,a,b) for (p,s),items in anns.items() for a,b in items}
    tp=len(pred&gold);pr=tp/max(1,len(pred));rc=tp/max(1,len(gold))
    return 2*pr*rc/max(1e-9,pr+rc),pr,rc,len(pred),len(gold)

@torch.no_grad()
def predict_logits(model,examples,device,batch=32):
    model.eval(); outputs=[];coll=Collator(0)
    for batch_ex in chunks(examples,batch):
        ins=coll(batch_ex);ids=ins.pop('labels').to(device);ins.pop('attention_mask') if False else None
        inputs={k:v.to(device) for k,v in ins.items()}
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(**inputs).logits.float().cpu().numpy()
        outputs.extend(logits[:,:])
    return outputs

def evaluate(model,examples,keys,anns,device):
    logits=predict_logits(model,examples,device)
    best=None
    for bias in np.arange(-1.01,1.011,.1):
        pred=exact_from_labels(examples,logits,float(bias));sc=score(pred,keys,anns)
        if best is None or sc[0]>best[0][0]:best=(sc,float(bias),pred)
    print('DEV exact best F1/P/R',*[round(x,4) if isinstance(x,float) else x for x in best[0]],'bias',best[1],flush=True)
    return best

def train(args):
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    tr=[];dv=[];dkeys=[];dans={}
    for split in (['training','development'] if args.combine else ['training']):
        texts,keys,anns=load(split);ex=encode(texts,keys,anns,tok)
        print(split,'sections',len(texts),'chunks',len(ex),flush=True)
        if split=='development' and not args.combine:dv=ex;dkeys=keys;dans=anns
        else:tr.extend(ex)
    model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3,
        id2label={0:'O',1:'B',2:'I'},label2id=LABELS)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    loader=DataLoader(tr,batch_size=args.batch,shuffle=True,collate_fn=Collator(tok.pad_token_id),num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    steps=len(loader)*args.epochs;warm=int(0.08*steps)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    step=0
    for ep in range(args.epochs):
        model.train();running=0
        for bi,batch in enumerate(loader):
            step+=1
            rate=args.lr*min(step/max(1,warm),1.0)*max(0.0,(steps-step)/max(1,steps-warm))
            for g in opt.param_groups:g['lr']=rate
            batch={k:v.to(device) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                loss=model(**batch).loss
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();running+=loss.item()
            if bi%50==0:print('epoch',ep+1,'step',bi,'loss',round(loss.item(),4),'lr',rate,flush=True)
        print('epoch',ep+1,'avg_loss',running/len(loader),flush=True)
        if dv:evaluate(model,dv,dkeys,dans,device)
        model.save_pretrained(args.out);tok.save_pretrained(args.out)
    if not dv:print('trained model saved',args.out,flush=True)

@torch.no_grad()
def run_predict(args):
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    model=AutoModelForTokenClassification.from_pretrained(args.model)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device).eval()
    texts,keys,_=load(args.split);ex=encode(texts,keys,{},tok)
    logits=predict_logits(model,ex,device)
    pred=exact_from_labels(ex,logits,args.bias)
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,'w') as f:
        for p,s,a,b in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])):
            f.write(f'{p}\t{s}:{a}:{b}\n')
    print('wrote',len(pred),'mentions to',args.output,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train');t.add_argument('--epochs',type=int,default=3);t.add_argument('--batch',type=int,default=12);t.add_argument('--lr',type=float,default=2e-5);t.add_argument('--out',default='/workspace/pubmedbert');t.add_argument('--combine',action='store_true');t.set_defaults(fn=train)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--split',default='test');p.add_argument('--output',default='/workspace/output/predictions.tsv');p.add_argument('--bias',type=float,default=0.0);p.set_defaults(fn=run_predict)
    args=ap.parse_args();args.fn(args)
