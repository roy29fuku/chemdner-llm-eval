import os, json, argparse, random, time, math, bisect, contextlib
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, get_linear_schedule_with_warmup

MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
CHEM_TYPES=['TRIVIAL','SYSTEMATIC','ABBREVIATION','FORMULA','FAMILY','IDENTIFIER','MULTIPLE']

class CaseBertForTokenClassification(BertForTokenClassification):
    def __init__(self,config):
        super().__init__(config)
        self.case_embeddings=torch.nn.Embedding(9,config.hidden_size,padding_idx=0)
        self.gap_embeddings=torch.nn.Embedding(3,config.hidden_size,padding_idx=0)
        self.type_classifier=torch.nn.Linear(config.hidden_size,7) if getattr(config,'aux_types',False) else None
        torch.nn.init.zeros_(self.case_embeddings.weight)
        torch.nn.init.zeros_(self.gap_embeddings.weight)
    def forward(self,input_ids=None,case_ids=None,gap_ids=None,inputs_embeds=None,type_labels=None,**kwargs):
        if inputs_embeds is None:inputs_embeds=self.bert.embeddings.word_embeddings(input_ids)
        if case_ids is not None:inputs_embeds=inputs_embeds+self.case_embeddings(case_ids)
        if gap_ids is not None:inputs_embeds=inputs_embeds+self.gap_embeddings(gap_ids)
        if type_labels is not None and self.type_classifier is not None:kwargs['output_hidden_states']=True
        out=super().forward(inputs_embeds=inputs_embeds,**kwargs)
        if type_labels is not None and self.type_classifier is not None:
            type_scores=self.type_classifier(self.dropout(out.hidden_states[-1]))
            extra=torch.nn.functional.cross_entropy(type_scores.reshape(-1,7),type_labels.reshape(-1),ignore_index=-100,reduction='sum')
            extra=extra/(type_labels!=-100).sum().clamp(min=1)
            out.loss=out.loss+.2*extra
        return out

def load_model(path,case_features=False,aux_types=False,**kwargs):
    config=AutoConfig.from_pretrained(path,**kwargs)
    had_case=getattr(config,'case_features',False)
    config.aux_types=bool(aux_types or getattr(config,'aux_types',False))
    config.case_features=bool(case_features or had_case or config.aux_types)
    if config.case_features:
        model=CaseBertForTokenClassification.from_pretrained(path,config=config)
        if not had_case:
            torch.nn.init.zeros_(model.case_embeddings.weight);torch.nn.init.zeros_(model.gap_embeddings.weight)
    else:model=AutoModelForTokenClassification.from_pretrained(path,config=config)
    return model

def case_id(text):
    letters=''.join(c for c in text if c.isalpha());digits=any(c.isdigit() for c in text)
    if not letters:return 7 if digits else 8
    if digits:return 5 if letters.isupper() else 6
    if letters.islower():return 1
    if letters.isupper():return 3
    if letters[0].isupper() and letters[1:].islower():return 2
    return 4

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda fn:fn

@njit(cache=True)
def viterbi_path(scores,continuation_bonus=0.):
    n=len(scores);k=scores.shape[1];back=np.empty((n,k),dtype=np.int64)
    previous=scores[0].astype(np.float64)
    for state in range(2,k,2):previous[state]=-1e9
    for i in range(1,n):
        best=0;val=previous[0]
        for state in range(1,k):
            if previous[state]>val:best=state;val=previous[state]
        current=np.empty(k,dtype=np.float64);current[0]=val+scores[i,0];back[i,0]=best
        for state in range(1,k,2):
            current[state]=val+scores[i,state];back[i,state]=best
            if previous[state]>=previous[state+1]+continuation_bonus:prev=state;inside=previous[state]
            else:prev=state+1;inside=previous[state+1]+continuation_bonus
            back[i,state+1]=prev;current[state+1]=inside+scores[i,state+1]
        previous=current
    state=0;val=previous[0]
    for j in range(1,k):
        if previous[j]>val:state=j;val=previous[j]
    labels=np.empty(n,dtype=np.int64);labels[-1]=state
    for i in range(n-1,0,-1):state=back[i,state];labels[i-1]=state
    return labels

def collapse_typed(scores):
    if scores.shape[1]==3:return scores
    # The target metric ignores mention type. Retain the strongest B and I
    # evidence without imposing type consistency on a predicted mention.
    return np.column_stack([scores[:,0],scores[:,1::2].max(-1),scores[:,2::2].max(-1)]).astype(np.float32)

def read_docs(path):
    docs=[]
    for line in open(path,encoding='utf8'):
        p,t,a=line.rstrip('\n').split('\t')
        docs.append(dict(pmid=p,title=t,text=t+'\n'+a,split=len(t)+1))
    return docs

def read_gold(path,docs):
    spans=defaultdict(list); gold=set(); byid={d['pmid']:d for d in docs}
    for line in open(path,encoding='utf8'):
        p,s,b,e,t,c=line.rstrip('\n').split('\t'); b=int(b);e=int(e)
        if p not in byid:continue
        gold.add((p,s,b,e)); shift=0 if s=='T' else byid[p]['split']
        spans[p].append((b+shift,e+shift))
    return spans,gold

def read_types(path,docs):
    byid={d['pmid']:d for d in docs};out={};indices={name:i for i,name in enumerate(CHEM_TYPES)}
    for line in open(path,encoding='utf8'):
        p,s,b,e,t,c=line.rstrip('\n').split('\t')
        if p not in byid:continue
        shift=0 if s=='T' else byid[p]['split']
        out[p,int(b)+shift,int(e)+shift]=indices.get(c,0)
    return out

class Corpus(Dataset):
    def __init__(self,docs,tokenizer,spans=None,length=384,overlap=96,types=None,case_features=False,aux_types=False):
        self.docs=docs; self.items=[];self.offsets=[];self.labels=[]
        case_features=case_features or aux_types
        self.case_features=case_features;self.cases=[];self.gaps=[];self.aux_types=aux_types;self.aux_labels=[]
        for di,d in enumerate(docs):
            tok=tokenizer(d['text'],add_special_tokens=False,return_offsets_mapping=True)
            ids=tok['input_ids']; offs=tok['offset_mapping'];self.offsets.append(offs)
            if case_features:
                self.cases.append([case_id(d['text'][b:e]) for b,e in offs]);gaps=[];previous_end=0
                for b,e in offs:
                    gap=d['text'][previous_end:b];gaps.append(2 if '\n' in gap else (1 if gap else 0));previous_end=e
                self.gaps.append(gaps)
            labs=np.zeros(len(ids),dtype=np.int64)
            auxiliary=np.full(len(ids),-100,dtype=np.int64) if aux_types else None
            ends=[e for b,e in offs]
            for b,e in sorted((spans or {}).get(d['pmid'],[])):
                j=bisect.bisect_right(ends,b); first=True
                chemical_type=(types or {}).get((d['pmid'],b,e),0)
                while j<len(offs) and offs[j][0]<e:
                    # Include intersecting tokens; decoder can use learned boundary corrections later.
                    if offs[j][1]>b:
                        labs[j]=(1 if first else 2)+(0 if aux_types else 2*chemical_type)
                        if aux_types:auxiliary[j]=chemical_type
                        first=False
                    j+=1
            self.labels.append(labs)
            if aux_types:self.aux_labels.append(auxiliary)
            cap=length-2
            starts=list(range(0,max(1,len(ids)-cap+1),cap-overlap))
            if starts[-1]+cap<len(ids):starts.append(max(0,len(ids)-cap))
            for start in starts:
                stop=min(start+cap,len(ids))
                self.items.append((di,start,stop,[tokenizer.cls_token_id]+ids[start:stop]+[tokenizer.sep_token_id]))
        print('corpus',len(docs),'documents',len(self.items),'windows',flush=True)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        di,b,e,ids=self.items[i]
        labs=[-100]+self.labels[di][b:e].tolist()+[-100]
        # A window beginning inside an entity is allowed to start with I.
        if self.case_features:
            out=(ids,labs,i,[0]+self.cases[di][b:e]+[0],[0]+self.gaps[di][b:e]+[0])
            if self.aux_types:out+=([-100]+self.aux_labels[di][b:e].tolist()+[-100],)
            return out
        return ids,labs,i

def collate(batch):
    n=(max(len(x[0]) for x in batch)+7)//8*8
    ids=torch.zeros((len(batch),n),dtype=torch.long)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    indices=[]
    cases=torch.zeros_like(ids) if len(batch[0])>=5 else None
    gaps=torch.zeros_like(ids) if cases is not None else None
    auxiliary=torch.full_like(ids,-100) if len(batch[0])==6 else None
    for j,item in enumerate(batch):
        tok,lab,i=item[:3]
        ids[j,:len(tok)]=torch.tensor(tok); mask[j,:len(tok)]=1
        labels[j,:len(tok)]=torch.tensor(lab);indices.append(i)
        if cases is not None:
            cases[j,:len(tok)]=torch.tensor(item[3]);gaps[j,:len(tok)]=torch.tensor(item[4])
        if auxiliary is not None:auxiliary[j,:len(tok)]=torch.tensor(item[5])
    data={'input_ids':ids,'attention_mask':mask,'labels':labels}
    if cases is not None:data['case_ids']=cases;data['gap_ids']=gaps
    if auxiliary is not None:data['type_labels']=auxiliary
    return data,indices

@torch.no_grad()
def infer(model,corpus,batch=24):
    model.eval(); sums=[np.zeros((len(o),model.config.num_labels),dtype=np.float32) for o in corpus.offsets]
    device=next(model.parameters()).device
    weights=[np.zeros(len(o),dtype=np.float32) for o in corpus.offsets]
    loader=DataLoader(corpus,batch_size=batch,collate_fn=collate,num_workers=2)
    for data,indices in loader:
        data={k:v.to(device) for k,v in data.items() if k not in ['labels','type_labels']}
        context=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()
        with context:out=model(**data).logits.float().cpu().numpy()
        for j,i in enumerate(indices):
            di,b,e,_=corpus.items[i];n=e-b
            w=np.minimum(np.minimum(np.arange(n)+1,np.arange(n)[::-1]+1),64).astype(np.float32)
            sums[di][b:e]+=out[j,1:n+1]*w[:,None];weights[di][b:e]+=w
    return [s/np.maximum(w[:,None],1) for s,w in zip(sums,weights)]

def decode(logits,corpus,bias=0,method='greedy',begin_penalty=0,continuation_bonus=0):
    result=set()
    for d,offs,scores in zip(corpus.docs,corpus.offsets,logits):
        if not len(scores):continue
        scores=collapse_typed(scores).copy(); scores[:,0]+=bias
        scores[:,1::2]-=begin_penalty
        # Enforce BIO transitions, preserving likelihood information at ambiguous boundaries.
        if method=='viterbi':
            states=viterbi_path(scores,continuation_bonus).tolist()
        else:states=scores.argmax(-1).tolist()
        lab=[];previous=0
        for state in states:
            if state==0:lab.append(0)
            elif state%2==1 or previous==0 or (state-1)//2!=(previous-1)//2:lab.append(1)
            else:lab.append(2)
            previous=state
        active=None
        def emit(span):
            if span is None:return
            b,e=span
            while b<e and d['text'][b].isspace():b+=1
            while e>b and d['text'][e-1].isspace():e-=1
            if e<=b:return
            if e<=d['split']-1:result.add((d['pmid'],'T',b,e))
            elif b>=d['split']:result.add((d['pmid'],'A',b-d['split'],e-d['split']))
        for (b,e),tag in zip(offs,lab):
            if active is not None and active[0]<d['split'] and b>=d['split']:
                emit(active);active=None
            if tag==0:emit(active);active=None
            elif tag==1 or active is None:
                emit(active);active=[b,e]
            else:active[1]=e
        emit(active)
    return result

def metrics(pred,gold):
    tp=len(pred&gold);p=tp/max(1,len(pred));r=tp/max(1,len(gold));f=2*p*r/max(1e-12,p+r)
    return dict(f1=f,precision=p,recall=r,tp=tp,pred=len(pred),gold=len(gold))

def write_predictions(pred,path):
    with open(path,'w') as f:
        for p,s,b,e in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])):f.write(f'{p}\t{s}:{b}:{e}\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--out',required=True)
    ap.add_argument('--model',default=MODEL);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--length',type=int,default=384);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=3e-5)
    ap.add_argument('--all',action='store_true');ap.add_argument('--predict',action='store_true');ap.add_argument('--bias',type=float,default=0)
    ap.add_argument('--accum',type=int,default=1);ap.add_argument('--checkpoint',action='store_true');ap.add_argument('--snapshots',action='store_true')
    ap.add_argument('--dev-fraction',type=float,default=0);ap.add_argument('--splits',nargs='+',default=['development','test'])
    ap.add_argument('--typed',action='store_true')
    ap.add_argument('--case-features',action='store_true')
    ap.add_argument('--aux-types',action='store_true')
    args=ap.parse_args();os.makedirs(args.out,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(6)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    if args.predict:
        model=load_model(args.model).to('cuda' if torch.cuda.is_available() else 'cpu')
        for split in args.splits:
            docs=read_docs(f'{args.data}/{split}.abstracts.txt');corpus=Corpus(docs,tokenizer,length=args.length,case_features=model.config.case_features)
            logits=infer(model,corpus);np.savez_compressed(f'{args.out}/{split}.logits.npz',*logits)
            write_predictions(decode(logits,corpus,args.bias),f'{args.out}/{split}.tsv')
        return
    typed=args.typed or AutoConfig.from_pretrained(args.model).num_labels>3
    use_aux=args.aux_types or getattr(AutoConfig.from_pretrained(args.model),'aux_types',False)
    use_case=args.case_features or getattr(AutoConfig.from_pretrained(args.model),'case_features',False) or use_aux
    docs=read_docs(f'{args.data}/training.abstracts.txt');spans,gold=read_gold(f'{args.data}/training.annotations.txt',docs)
    devdocs=read_docs(f'{args.data}/development.abstracts.txt');devspans,devgold=read_gold(f'{args.data}/development.annotations.txt',devdocs)
    types=None
    if typed or use_aux:
        types=read_types(f'{args.data}/training.annotations.txt',docs);types.update(read_types(f'{args.data}/development.annotations.txt',devdocs))
    if args.all:docs+=devdocs;spans.update(devspans)
    elif args.dev_fraction:
        indices=list(range(len(devdocs)));random.Random(1729).shuffle(indices)
        cut=int(len(indices)*args.dev_fraction);train_ids=set(indices[:cut]);held_ids=set(indices[cut:])
        docs+=[d for i,d in enumerate(devdocs) if i in train_ids];spans.update(devspans)
        devdocs=[d for i,d in enumerate(devdocs) if i in held_ids]
        held={d['pmid'] for d in devdocs};devgold={g for g in devgold if g[0] in held}
        json.dump(sorted(held),open(args.out+'/heldout_pmids.json','w'))
    train=Corpus(docs,tokenizer,spans,args.length,types=types,case_features=use_case,aux_types=use_aux);dev=Corpus(devdocs,tokenizer,devspans,args.length,types=types,case_features=use_case,aux_types=use_aux)
    labels=['O']+[prefix+'-'+name for name in (CHEM_TYPES if typed else ['CHEM']) for prefix in ['B','I']]
    model=load_model(args.model,case_features=use_case,aux_types=use_aux,num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={s:i for i,s in enumerate(labels)}).cuda()
    if args.checkpoint:model.gradient_checkpointing_enable()
    params=[{'params':[p for n,p in model.named_parameters() if not any(x in n for x in ['bias','LayerNorm.weight'])],'weight_decay':.01},
            {'params':[p for n,p in model.named_parameters() if any(x in n for x in ['bias','LayerNorm.weight'])],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(params,lr=args.lr)
    loader=DataLoader(train,batch_size=args.batch,shuffle=True,collate_fn=collate,num_workers=2)
    total=args.epochs*math.ceil(len(loader)/args.accum);scheduler=get_linear_schedule_with_warmup(optimizer,int(total*.06),total)
    scaler=torch.amp.GradScaler('cuda');best=0;step=0;start=time.time()
    for ep in range(args.epochs):
        model.train();loss_sum=0
        optimizer.zero_grad(set_to_none=True)
        for micro,(data,_) in enumerate(loader):
            data={k:v.cuda() for k,v in data.items()}
            with torch.autocast('cuda',dtype=torch.float16):loss=model(**data).loss
            group_size=min(args.accum,len(loader)-(micro//args.accum)*args.accum)
            scaler.scale(loss/group_size).backward()
            if (micro+1)%args.accum==0 or micro+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            loss_sum+=loss.item();step+=1
            if step%100==0:print(json.dumps(dict(epoch=ep+1,step=step,loss=loss_sum/(step%len(loader) or len(loader)),elapsed=time.time()-start)),flush=True)
        if args.all:
            score=ep+1
        else:
            logits=infer(model,dev);res=metrics(decode(logits,dev),devgold);score=res['f1']
            print('DEV',ep+1,json.dumps(res),'seconds',time.time()-start,flush=True)
        if score>best:
            best=score;model.save_pretrained(args.out+'/best');tokenizer.save_pretrained(args.out+'/best')
            if not args.all:
                np.savez_compressed(args.out+'/development.logits.npz',*logits)
                write_predictions(decode(logits,dev),args.out+'/development.tsv')
            with open(args.out+'/best.json','w') as f:json.dump(dict(epoch=ep+1,score=score,args=vars(args)),f)
        if args.snapshots and ep>=2:
            model.save_pretrained(f'{args.out}/epoch{ep+1}');tokenizer.save_pretrained(f'{args.out}/epoch{ep+1}')
        print('EPOCH',ep+1,'loss',loss_sum/len(loader),'best',best,'elapsed',time.time()-start,flush=True)

if __name__=='__main__':main()
