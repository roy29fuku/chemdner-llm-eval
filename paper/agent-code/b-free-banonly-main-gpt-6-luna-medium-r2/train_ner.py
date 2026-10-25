import argparse, json, os, random, re, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, get_linear_schedule_with_warmup

MODEL = "google/bert_uncased_L-2_H-128_A-2"
ID2LABEL = {0: "O", 1: "B-CHEM", 2: "I-CHEM"}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}
MAX_CONTENT = 382
STRIDE = 332
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(split, base="/workspace/data"):
    docs = {}
    with open(f"{base}/{split}.abstracts.txt", encoding="utf8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            docs[pmid] = {"T":title, "A":abstract}
    anns = {p:{"T":[],"A":[]} for p in docs}
    ap = Path(base) / f"{split}.annotations.txt"
    if ap.exists():
        with ap.open(encoding="utf8") as f:
            for line in f:
                p, sec, s, e, txt, cls = line.rstrip("\n").split("\t")
                s,e=int(s),int(e)
                if docs[p][sec][s:e] != txt:
                    raise ValueError((p,sec,s,e,txt,docs[p][sec][s:e]))
                anns[p][sec].append((s,e))
    return docs, anns

def token_labeling(text, spans, tokenizer):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    labels = [0] * len(offsets)
    # Spans are exact gold intervals. Label every token intersecting a mention, preserving boundaries.
    for s,e in spans:
        inds = [i for i,(a,b) in enumerate(offsets) if a < e and b > s]
        if not inds: continue
        labels[inds[0]] = 1
        for i in inds[1:]: labels[i] = 2
    return enc["input_ids"], offsets, labels

def windows(ids, offsets, labels=None):
    out=[]
    starts=list(range(0, max(1,len(ids)), STRIDE))
    for st in starts:
        en=min(st+MAX_CONTENT,len(ids))
        if st >= len(ids): break
        assert en-st <= MAX_CONTENT
        out.append((st,en))
        if en==len(ids): break
    return out

class WindowDataset(Dataset):
    def __init__(self, docs, anns, tokenizer):
        self.data=[]
        self.tokenizer=tokenizer
        self.examples=[]
        for p in docs:
            for sec in ("T","A"):
                text=docs[p][sec]
                ids, offsets, toklabels=token_labeling(text, anns[p][sec], tokenizer)
                for st,en in windows(ids,offsets,toklabels):
                    self.data.append((ids[st:en],toklabels[st:en]))
                self.examples.append((p,sec,text,ids,offsets))
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        ids,labs=self.data[i]
        x=self.tokenizer.prepare_for_model(ids, add_special_tokens=True, return_attention_mask=True, truncation=False)
        labels=[-100]+labs+[-100]
        return {"input_ids":x["input_ids"],"attention_mask":x["attention_mask"],"labels":labels}

def collate(batch, pad_id):
    n=max(len(x["input_ids"]) for x in batch)
    result={"input_ids":[],"attention_mask":[],"labels":[]}
    for x in batch:
        pad=n-len(x["input_ids"])
        result["input_ids"].append(x["input_ids"]+[pad_id]*pad)
        result["attention_mask"].append(x["attention_mask"]+[0]*pad)
        result["labels"].append(x["labels"]+[-100]*pad)
    return {k:torch.tensor(v,dtype=torch.long) for k,v in result.items()}

@torch.no_grad()
def predict(model, dataset, batch_size=16):
    model.eval(); pred=set()
    sums=[np.zeros((len(ex[3]),3),dtype=np.float32) for ex in dataset.examples]
    counts=[np.zeros(len(ex[3]),dtype=np.float32) for ex in dataset.examples]
    jobs=[]
    for ei,ex in enumerate(dataset.examples):
        ids,offsets=ex[3],ex[4]
        for st,en in windows(ids,offsets):
            x=dataset.tokenizer.prepare_for_model(ids[st:en],add_special_tokens=True,return_attention_mask=True)
            jobs.append((ei,st,en,x["input_ids"],x["attention_mask"]))
    for j in range(0,len(jobs),batch_size):
        part=jobs[j:j+batch_size]; n=max(len(x[3]) for x in part)
        ti=torch.tensor([x[3]+[dataset.tokenizer.pad_token_id]*(n-len(x[3])) for x in part],device=DEVICE)
        am=torch.tensor([x[4]+[0]*(n-len(x[4])) for x in part],device=DEVICE)
        if DEVICE=="cuda":
            with torch.autocast("cuda",dtype=torch.float16): logits=model(input_ids=ti,attention_mask=am).logits.float().cpu().numpy()
        else: logits=model(input_ids=ti,attention_mask=am).logits.numpy()
        for k,(ei,st,en,_,_) in enumerate(part):
            vals=logits[k,1:1+(en-st)]
            sums[ei][st:en]+=vals; counts[ei][st:en]+=1
    for ei,(p,sec,text,ids,offsets) in enumerate(dataset.examples):
        if not len(ids): continue
        best=(sums[ei] / np.maximum(counts[ei][:,None],1)).argmax(1)
        start=None; last_end=None
        for i,lab in enumerate(best):
            a,b=offsets[i]
            if b<=a: continue
            if lab==1 or (lab==2 and start is None):
                if start is not None: pred.add((p,sec,start,last_end))
                start=a; last_end=b
            elif lab==2:
                last_end=b
            elif start is not None:
                pred.add((p,sec,start,last_end)); start=None; last_end=None
        if start is not None: pred.add((p,sec,start,last_end))
    return pred

def score(pred, gold):
    g={(p,s,a,b) for p in gold for s in ("T","A") for a,b in gold[p][s]}
    tp=len(g&pred); pr=len(pred); gr=len(g)
    return (2*tp/(pr+gr) if pr+gr else 0, tp,pr,gr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="training"); ap.add_argument("--dev",default="development")
    ap.add_argument("--epochs",type=int,default=4); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--accum",type=int,default=1)
    ap.add_argument("--lr",type=float,default=3e-5); ap.add_argument("--out",default="/workspace/output/biobert")
    ap.add_argument("--predict",default=None); ap.add_argument("--checkpoint",default=None)
    args=ap.parse_args()
    random.seed(2307); np.random.seed(2307); torch.manual_seed(2307); torch.cuda.manual_seed_all(2307)
    torch.backends.cuda.matmul.allow_tf32=True
    if DEVICE=="cpu": torch.set_num_threads(20)
    tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    config=AutoConfig.from_pretrained(MODEL,num_labels=3,id2label=ID2LABEL,label2id=LABEL2ID)
    if args.checkpoint:
        model=AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    else:
        model=AutoModelForTokenClassification.from_pretrained(MODEL,config=config)
    model.to(DEVICE)
    if args.predict:
        docs,anns=load_data(args.predict)
        ds=WindowDataset(docs,anns,tok)
        pred=predict(model,ds)
        out=args.out
        with open(out,"w",encoding="utf8") as f:
            for p,s,a,b in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])): f.write(f"{p}\t{s}:{a}:{b}\n")
        if any(anns.values()): print("F1,TP,pred,gold",score(pred,anns))
        print("wrote",len(pred),out)
        return
    trdocs,trans=load_data(args.train); dvdocs,dvanns=load_data(args.dev)
    train=WindowDataset(trdocs,trans,tok); dev=WindowDataset(dvdocs,dvanns,tok)
    dl=DataLoader(train,batch_size=args.batch,shuffle=True,collate_fn=lambda x:collate(x,tok.pad_token_id),num_workers=2,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    total=math.ceil(len(dl)/args.accum)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total*.08),total)
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE=="cuda")
    best=0
    os.makedirs(args.out,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); losses=[]; optimizer.zero_grad(set_to_none=True)
        for step,batch in enumerate(dl):
            batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
            if DEVICE=="cuda":
                with torch.autocast("cuda",dtype=torch.float16): loss=model(**batch).loss
            else: loss=model(**batch).loss
            scaler.scale(loss/args.accum).backward()
            if (step+1)%args.accum==0 or step+1==len(dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if step%100==0: print(f"epoch {epoch+1} step {step}/{len(dl)} loss {np.mean(losses[-100:]):.4f}",flush=True)
        pred=predict(model,dev); f1,tp,pr,gr=score(pred,dvanns)
        print(f"EPOCH {epoch+1} loss {np.mean(losses):.4f} dev f1={f1:.5f} tp={tp} pred={pr} gold={gr}",flush=True)
        if f1>best:
            best=f1; model.save_pretrained(args.out); tok.save_pretrained(args.out)
            with open(os.path.join(args.out,"dev_predictions.json"),"w") as f: json.dump(sorted(list(pred)),f)
    print("BEST",best)

if __name__=="__main__": main()
