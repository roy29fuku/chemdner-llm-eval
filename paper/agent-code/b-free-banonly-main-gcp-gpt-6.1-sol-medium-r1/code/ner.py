import os, json, time, math, random, argparse, pickle, bisect
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(**kwargs):return lambda f:f

LABELS = ['O','B','I','L','U']

def read_documents(root, split):
    annotations = defaultdict(list)
    ap = Path(root) / f'{split}.annotations.txt'
    if ap.exists():
        for line in ap.read_text().splitlines():
            p, sec, a, b, text, cls = line.split('\t')
            annotations[p,sec].append((int(a),int(b)))
    documents = []
    for line in (Path(root) / f'{split}.abstracts.txt').read_text().splitlines():
        p,t,a = line.split('\t')
        documents.append(dict(pmid=p, title=t, abstract=a,
                              annotations=[annotations[p,'T'],annotations[p,'A']]))
    return documents

def tokenize_documents(documents, tokenizer):
    result = []
    mismatches = 0
    for d in documents:
        ids, offsets, sections, labels = [], [], [], []
        for sec, text, anns in zip(['T','A'], [d['title'],d['abstract']],d['annotations']):
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            si, so = enc['input_ids'], enc['offset_mapping']
            sl = [0]*len(si)
            starts=[a for a,b in so]; ends=[b for a,b in so]
            for a,b in sorted(anns):
                i=bisect.bisect_right(ends,a)
                j=bisect.bisect_left(starts,b)
                if i>=j: continue
                mismatches += int(so[i][0]!=a or so[j-1][1]!=b)
                if j-i==1: sl[i]=4
                else: sl[i]=1; sl[i+1:j-1]=[2]*(j-i-2); sl[j-1]=3
            if ids:
                ids.append(tokenizer.sep_token_id); offsets.append((0,0)); sections.append('S'); labels.append(0)
            ids.extend(si); offsets.extend(so); sections.extend([sec]*len(si)); labels.extend(sl)
        result.append(dict(**d, ids=ids, offsets=offsets, sections=sections, labels=labels))
    print(f'tokenized {len(result)} documents; {mismatches} nonaligned mentions', flush=True)
    return result

def make_windows(docs, tokenizer, max_length=384, overlap=96):
    windows=[]
    width=max_length-2
    for di,d in enumerate(docs):
        n=len(d['ids']); start=0
        while start<n:
            end=min(start+width,n)
            lab=list(d['labels'][start:end])
            # Interior windows contribute labels only where they have enough context.
            if start: lab[:24]=[-100]*min(24,len(lab))
            if end<n: lab[-24:]=[-100]*min(24,len(lab))
            windows.append(dict(input_ids=[tokenizer.cls_token_id]+d['ids'][start:end]+[tokenizer.sep_token_id],
                                labels=[-100]+lab+[-100], doc=di, start=start, end=end))
            if end==n: break
            start=end-overlap
    return windows

@njit(cache=True)
def viterbi(e):
    trans=np.full((5,5),-10000.,dtype=np.float32)
    for a in (0,3,4):
        for b in (0,1,4):trans[a,b]=0
    for a in (1,2):
        for b in (2,3):trans[a,b]=0
    score=e[0].copy(); score[2]=-10000;score[3]=-10000
    back=np.empty((len(e),5),dtype=np.int8)
    for i in range(1,len(e)):
        updated=np.empty(5,dtype=np.float32)
        for b in range(5):
            value=-1e20; previous=0
            for a in range(5):
                candidate=score[a]+trans[a,b]
                if candidate>value:value=candidate;previous=a
            back[i,b]=previous;updated[b]=value+e[i,b]
        score=updated
    score[1]=-10000;score[2]=-10000
    state=int(score.argmax()); states=np.empty(len(e),dtype=np.int8);states[-1]=state
    for i in range(len(e)-1,0,-1):
        state=int(back[i,state]); states[i-1]=state
    return states

def decode(emissions, sections, bias=0):
    # Constrained BILOU Viterbi decoding.
    e=np.asarray(emissions,dtype=np.float32).copy()
    e[:,0]+=bias
    for i,s in enumerate(sections):
        if s=='S': e[i,1:]=-10000
    return viterbi(e)

def spans_from_scores(docs, scores, bias=0, boundary_map=None):
    predictions=set()
    for d,score in zip(docs,scores):
        states=decode(score,d['sections'],bias)
        start=None
        for i,y in enumerate(states):
            if y in [1,4]: start=i
            if y in [3,4] and start is not None:
                sec=d['sections'][start]
                if sec!='S' and sec==d['sections'][i]:
                    a,b=d['offsets'][start][0],d['offsets'][i][1]
                    text=d['title'] if sec=='T' else d['abstract']
                    if boundary_map:
                        delta=boundary_map.get(text[a:b].lower())
                        if delta: a+=delta[0]; b-=delta[1]
                    if a<b: predictions.add((d['pmid'],sec,a,b))
                start=None
    return predictions

def gold_spans(docs):
    return {(d['pmid'],s,a,b) for d in docs for s,anns in zip(['T','A'],d['annotations']) for a,b in anns}

def metric(pred,gold):
    tp=len(pred&gold); p=tp/max(1,len(pred)); r=tp/max(1,len(gold)); f=2*p*r/max(1e-12,p+r)
    return dict(f1=f,precision=p,recall=r,tp=tp,predicted=len(pred),gold=len(gold))

def boundary_corrections(docs,min_count=2):
    counts=defaultdict(Counter)
    for d in docs:
        for sec,text,anns in zip(['T','A'],[d['title'],d['abstract']],d['annotations']):
            offsets=[o for o,s in zip(d['offsets'],d['sections']) if s==sec]
            starts=[a for a,b in offsets]; ends=[b for a,b in offsets]
            for a,b in anns:
                i=bisect.bisect_right(ends,a);j=bisect.bisect_left(starts,b)
                if i>=j:continue
                aa,bb=offsets[i][0],offsets[j-1][1]
                counts[text[aa:bb].lower()][a-aa,bb-b]+=1
    return {k:cnt.most_common(1)[0][0] for k,cnt in counts.items()
            if cnt.most_common(1)[0][0]!=(0,0) and cnt.most_common(1)[0][1]>=min_count}

def save_predictions(pred,path):
    with open(path,'w') as f:
        for p,s,a,b in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])):f.write(f'{p}\t{s}:{a}:{b}\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train','predict'])
    parser.add_argument('--model',default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--data',default='/workspace/data')
    parser.add_argument('--out',required=True)
    parser.add_argument('--splits',default='training')
    parser.add_argument('--eval',default='development')
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch',type=int,default=24)
    parser.add_argument('--length',type=int,default=384)
    parser.add_argument('--overlap',type=int,default=96)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--eval-every',type=int,default=1)
    parser.add_argument('--save-epochs',default='')
    parser.add_argument('--checkpoint',default='')
    parser.add_argument('--grad-checkpoint',action='store_true')
    parser.add_argument('--accumulation',type=int,default=1)
    parser.add_argument('--bucket',action='store_true')
    parser.add_argument('--split-punct',action='store_true')
    parser.add_argument('--case-sensitive',action='store_true')
    args=parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device='cuda'
    args.case_sensitive=args.case_sensitive or ('cased' in args.model.lower() and 'uncased' not in args.model.lower())
    tokenizer=AutoTokenizer.from_pretrained(args.model,**({'do_lower_case':False} if args.case_sensitive else {}))
    if args.split_punct:
        from tokenizers import Regex, pre_tokenizers
        tokenizer.backend_tokenizer.pre_tokenizer=pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r'(?<=\S)[^\w\s]'),behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False,trim_offsets=True)])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tokenizer.save_pretrained(out/'tokenizer')
    cache=Path(args.data)/('cache_'+args.model.replace('/','_')+('_splitpunct' if args.split_punct else '')+('_cased' if args.case_sensitive else ''))
    cache.mkdir(exist_ok=True)
    def get_docs(split):
        p=cache/f'{split}.pkl'
        if p.exists():return pickle.load(p.open('rb'))
        docs=tokenize_documents(read_documents(args.data,split),tokenizer)
        pickle.dump(docs,p.open('wb'));return docs
    eval_docs=get_docs(args.eval)
    eval_windows=make_windows(eval_docs,tokenizer,args.length,args.overlap)
    # Sort inference windows to reduce padding, retaining their document coordinates.
    eval_windows.sort(key=lambda w:len(w['input_ids']))
    def collate(windows):
        n=((max(len(w['input_ids']) for w in windows)+7)//8)*8
        ids=np.full((len(windows),n),tokenizer.pad_token_id,dtype=np.int64)
        mask=np.zeros_like(ids); labels=np.full_like(ids,-100)
        for i,w in enumerate(windows):
            l=len(w['input_ids']); ids[i,:l]=w['input_ids']; mask[i,:l]=1;labels[i,:l]=w['labels']
        return dict(input_ids=torch.from_numpy(ids),attention_mask=torch.from_numpy(mask),labels=torch.from_numpy(labels)),windows
    def infer(model):
        model.eval()
        accum=[np.zeros((len(d['ids']),5),np.float32) for d in eval_docs]
        weights=[np.zeros(len(d['ids']),np.float32) for d in eval_docs]
        loader=DataLoader(eval_windows,batch_size=args.batch*2,collate_fn=collate,num_workers=2,pin_memory=True)
        with torch.inference_mode():
            for inputs, windows in loader:
                inputs={k:v.to(device,non_blocking=True) for k,v in inputs.items() if k!='labels'}
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(**inputs).logits.float().cpu().numpy()
                for i,w in enumerate(windows):
                    a,b=w['start'],w['end'];di=w['doc'];n=b-a
                    weight=np.minimum(np.arange(1,n+1),np.arange(n,0,-1)).clip(1,48).astype(np.float32)
                    accum[di][a:b]+=logits[i,1:n+1]*weight[:,None];weights[di][a:b]+=weight
        return [a/w[:,None] for a,w in zip(accum,weights)]
    model=AutoModelForTokenClassification.from_pretrained(args.checkpoint or args.model,num_labels=5,
                                                         id2label=dict(enumerate(LABELS)),label2id={x:i for i,x in enumerate(LABELS)},
                                                         attn_implementation='sdpa').to(device)
    if args.grad_checkpoint:model.gradient_checkpointing_enable()
    if args.mode=='predict':
        scores=infer(model)
        pickle.dump(dict(docs=eval_docs,scores=scores),open(out/f'{args.eval}_scores.pkl','wb'))
        pred=spans_from_scores(eval_docs,scores)
        save_predictions(pred,out/f'{args.eval}_predictions.tsv')
        if args.eval!='test':print(metric(pred,gold_spans(eval_docs)),flush=True)
        return
    train_docs=sum([get_docs(s) for s in args.splits.split(',')],[])
    train_windows=make_windows(train_docs,tokenizer,args.length,args.overlap)
    boundary_map=boundary_corrections(train_docs)
    json.dump(boundary_map,open(out/'boundary_map.json','w'))
    json.dump(vars(args),open(out/'config.json','w'),indent=2)
    print(f'TRAIN {len(train_docs)} docs, {len(train_windows)} windows; eval {len(eval_windows)} windows',flush=True)
    groups=[{'params':[p for n,p in model.named_parameters() if not any(k in n for k in ['bias','LayerNorm.weight'])], 'weight_decay':0.01},
            {'params':[p for n,p in model.named_parameters() if any(k in n for k in ['bias','LayerNorm.weight'])], 'weight_decay':0.0}]
    optimizer=torch.optim.AdamW(groups,lr=args.lr,eps=1e-8)
    steps=math.ceil(math.ceil(len(train_windows)/args.batch)/args.accumulation)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*0.08),steps)
    scaler=torch.amp.GradScaler('cuda')
    gold=gold_spans(eval_docs)
    best=-1;global_step=0;began=time.time()
    requested={int(x) for x in args.save_epochs.split(',') if x}
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0
        if args.bucket:
            order=list(range(len(train_windows)));random.shuffle(order);batches=[]
            for a in range(0,len(order),args.batch*64):
                pool=order[a:a+args.batch*64]
                pool.sort(key=lambda i:len(train_windows[i]['input_ids']))
                batches.extend(pool[b:b+args.batch] for b in range(0,len(pool),args.batch))
            random.shuffle(batches)
            loader=DataLoader(train_windows,batch_sampler=batches,collate_fn=collate,num_workers=2,pin_memory=True)
        else:
            loader=DataLoader(train_windows,batch_size=args.batch,shuffle=True,collate_fn=collate,num_workers=2,pin_memory=True)
        for step,(inputs,windows) in enumerate(loader):
            inputs={k:v.to(device,non_blocking=True) for k,v in inputs.items()}
            with torch.autocast('cuda',dtype=torch.float16):loss=model(**inputs).loss
            scaler.scale(loss/args.accumulation).backward()
            if (step+1)%args.accumulation==0 or step+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            loss_sum+=loss.item();global_step+=1
            if (step+1)%100==0:
                print(f'epoch={epoch} step={step+1}/{len(loader)} loss={loss_sum/(step+1):.5f} elapsed={time.time()-began:.1f}s',flush=True)
        print(f'EPOCH {epoch} train_loss={loss_sum/len(loader):.5f} elapsed={time.time()-began:.1f}s',flush=True)
        if epoch%args.eval_every==0:
            scores=infer(model)
            result=metric(spans_from_scores(eval_docs,scores,boundary_map=boundary_map),gold)
            print('EVAL',epoch,json.dumps(result),flush=True)
            with open(out/'metrics.jsonl','a') as f:f.write(json.dumps(dict(epoch=epoch,**result))+'\n')
            if result['f1']>best:
                best=result['f1'];model.save_pretrained(out/'best')
                pickle.dump(dict(docs=eval_docs,scores=scores),open(out/'best_scores.pkl','wb'))
        if epoch in requested:
            model.save_pretrained(out/f'epoch{epoch}')
            if epoch%args.eval_every==0:
                pickle.dump(dict(docs=eval_docs,scores=scores),open(out/f'epoch{epoch}_scores.pkl','wb'))
        if epoch==args.epochs:model.save_pretrained(out/'last')
    print('DONE',best,time.time()-began,flush=True)

if __name__=='__main__':main()
