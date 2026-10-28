import os, json, random, math, time
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

ROOT='/workspace/data'
MODEL='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
MAX_LEN=384
WINDOW=300
LABELS={'O':0,'B-CHEM':1,'I-CHEM':2}

random.seed(41); torch.manual_seed(41)
torch.set_num_threads(8)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load(split):
    sections={}
    for line in open(f'{ROOT}/{split}.abstracts.txt',encoding='utf8'):
        pmid,title,abstract=line.rstrip('\n').split('\t')
        sections[(pmid,'T')]=title
        sections[(pmid,'A')]=abstract
    spans=defaultdict(list)
    if split!='test':
        for line in open(f'{ROOT}/{split}.annotations.txt',encoding='utf8'):
            pmid,sec,start,end,text,cl=line.rstrip('\n').split('\t')
            spans[(pmid,sec)].append((int(start),int(end)))
    return sections,spans

class ChunkDataset(Dataset):
    def __init__(self, sections, spans, tokenizer, labeled):
        self.items=[]
        self.tokenizer=tokenizer
        self.sections=sections
        self.keys=[]
        for key,text in sections.items():
            self.keys.append(key)
            enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc['input_ids']; offsets=enc['offset_mapping']
            if not ids:
                self.items.append((key,0,[tokenizer.cls_token_id,tokenizer.sep_token_id],[-100,-100],[],[])); continue
            step=WINDOW
            starts=list(range(0,len(ids),step))
            for ci,st in enumerate(starts):
                en=min(st+MAX_LEN-2,len(ids))
                # Avoid a tiny final window when the prior window already covers it.
                if en==len(ids) and ci>0 and len(ids)-st<80:
                    continue
                subids=ids[st:en]; suboffs=offsets[st:en]
                labs=[]
                gold=spans.get(key,[])
                for a,b in suboffs:
                    label=0
                    if labeled and b>a:
                        hit=None
                        for ga,gb in gold:
                            if a<gb and b>ga:
                                hit=(ga,gb); break
                        if hit:
                            label=1 if a<=hit[0] and (a==hit[0] or st==0) else 2
                            # If a mention starts before this crop, its first seen token is B.
                            if a<=hit[0]: label=1
                    labs.append(label)
                input_ids=[tokenizer.cls_token_id]+subids+[tokenizer.sep_token_id]
                full_labs=[-100]+labs+[-100]
                full_offsets=[(0,0)]+suboffs+[(0,0)]
                self.items.append((key,st,input_ids,full_labs,full_offsets,list(range(st,en))))
        self.labeled=labeled
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(batch):
    n=max(len(x[2]) for x in batch)
    ids=torch.full((len(batch),n),tokenizer.pad_token_id,dtype=torch.long)
    masks=torch.zeros((len(batch),n),dtype=torch.long)
    labs=torch.full((len(batch),n),-100,dtype=torch.long)
    for i,x in enumerate(batch):
        l=len(x[2]); ids[i,:l]=torch.tensor(x[2]); masks[i,:l]=1; labs[i,:l]=torch.tensor(x[3])
    return ids,masks,labs,batch

def decoded_spans(logits, chunk_info, bias=0.):
    # Recombine overlapping windows at the original subword-token level.
    sums=defaultdict(lambda: torch.zeros(3)); counts=defaultdict(int)
    for pred,item in zip(logits,chunk_info):
        pred=torch.as_tensor(pred)
        key,st,input_ids,labs,offs,global_indices=item
        for j,gi in enumerate(global_indices):
            sums[(key,gi)]+=pred[j+1]; counts[(key,gi)]+=1
    result=defaultdict(set)
    for key,text in section_data.items():
        enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        row=[]
        for gi,(a,b) in enumerate(enc['offset_mapping']):
            v=sums[(key,gi)]/max(counts[(key,gi)],1)
            v=v.clone(); v[1:]+=bias
            row.append((int(v.argmax()),a,b))
        cur=None
        for lab,a,b in row:
            if lab==1:
                if cur: result[key].add(cur)
                cur=(a,b)
            elif lab==2:
                if cur: cur=(cur[0],b)
                else: cur=(a,b)
            else:
                if cur: result[key].add(cur); cur=None
        if cur: result[key].add(cur)
    return result

def scores(pred,gold):
    ps={(k,a,b) for k,v in pred.items() for a,b in v}
    gs={(k,a,b) for k,v in gold.items() for a,b in v}
    tp=len(ps&gs); p=tp/max(len(ps),1); r=tp/max(len(gs),1)
    return p,r,2*p*r/max(p+r,1e-9),len(ps),len(gs)

tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
train_sections,train_gold=load('training')
dev_sections,dev_gold=load('development')
section_data=dev_sections
train_ds=ChunkDataset(train_sections,train_gold,tokenizer,True)
dev_ds=ChunkDataset(dev_sections,dev_gold,tokenizer,False)
print('chunks',len(train_ds),len(dev_ds),'mentions',sum(map(len,train_gold.values())),sum(map(len,dev_gold.values())),flush=True)

model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=3,id2label={v:k for k,v in LABELS.items()},label2id=LABELS)
model.to(device)
batch_size=16
loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0)
vloader=DataLoader(dev_ds,batch_size=16,shuffle=False,collate_fn=collate,num_workers=0)
epochs=3
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
total=epochs*len(loader)
sched=get_linear_schedule_with_warmup(optimizer,int(.06*total),total)
class_weights=torch.tensor([1.,2.5,2.5],device=device)
scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
best_f1=0.; best_bias=0.
for epoch in range(epochs):
    model.train(); running=0.; tic=time.time()
    for step,(ids,mask,labs,_) in enumerate(loader):
        ids,mask,labs=ids.to(device),mask.to(device),labs.to(device)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            out=model(input_ids=ids,attention_mask=mask).logits
        loss=torch.nn.functional.cross_entropy(out.float().reshape(-1,3),labs.reshape(-1),weight=class_weights,ignore_index=-100)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer); scaler.update(); sched.step(); optimizer.zero_grad(set_to_none=True)
        running+=loss.item()
        if (step+1)%100==0: print('epoch',epoch+1,'step',step+1,'loss',round(running/(step+1),4),flush=True)
    model.eval(); logits=[]; info=[]
    with torch.no_grad():
        for ids,mask,labs,batch in vloader:
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                out=model(input_ids=ids.to(device),attention_mask=mask.to(device)).logits
            out=out.float().cpu()
            for i,item in enumerate(batch):
                info.append(item); logits.append(out[i,:len(item[2])].tolist())
    results=[]
    for bias in [-1.5,-1.0,-.5,0,.25,.5,.75,1.0,1.5]:
        pred=decoded_spans(logits,info,bias)
        p,r,f1,npred,ngold=scores(pred,dev_gold)
        results.append((f1,bias,p,r,npred,ngold))
    results.sort(reverse=True)
    print('epoch_done',epoch+1,'sec',round(time.time()-tic,1),'loss',round(running/max(len(loader),1),4),'best',results[:5],flush=True)
    if results[0][0]>best_f1:
        best_f1,best_bias=results[0][0],results[0][1]
        torch.save(model.state_dict(),'/workspace/output/chemical_ner.pt')
        tokenizer.save_pretrained('/workspace/output/tokenizer')
        with open('/workspace/output/model_meta.json','w') as f: json.dump({'model':MODEL,'bias':best_bias,'f1':best_f1},f)
        print('saved best model',best_f1,best_bias,flush=True)

print('final_best',best_f1,best_bias)
