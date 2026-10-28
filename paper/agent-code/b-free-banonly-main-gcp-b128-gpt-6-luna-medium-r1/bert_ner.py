import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

from collections import defaultdict
import random
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

import ner

MODEL_ID = 'distilbert/distilbert-base-cased'
MAX_LEN = 256
STRIDE = 64
LABELS = {'O':0,'B':1,'I':2}

def make_items(articles, annotations):
    items=[]
    for pmid, sections in articles.items():
        for sec in ('T','A'):
            items.append((pmid,sec,sections[sec],annotations.get(pmid,{}).get(sec,[])))
    return items

def encode_items(items, tokenizer):
    data=[]
    special=2
    width=MAX_LEN-special
    step=width-STRIDE
    for docid,(pmid,sec,text,anns) in enumerate(items):
        enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=enc['input_ids']; offsets=enc['offset_mapping']
        y=[0]*len(ids)
        for s,e,_,_ in anns:
            hits=[i for i,(a,b) in enumerate(offsets) if a<e and b>s]
            if not hits: continue
            y[hits[0]]=1
            for i in hits[1:]:y[i]=2
        chunks=[]
        start=0
        while start<len(ids) or (len(ids)==0 and start==0):
            end=min(start+width,len(ids)); sub=ids[start:end]; att=[tokenizer.cls_token_id]+sub+[tokenizer.sep_token_id]
            labels=[-100]+y[start:end]+[-100]
            token_indices=[None]+list(range(start,end))+[None]
            pad=MAX_LEN-len(att)
            att=att+[tokenizer.pad_token_id]*pad
            labels=labels+[-100]*pad
            token_indices=token_indices+[None]*pad
            mask=[1]*(MAX_LEN-pad)+[0]*pad
            data.append((att,mask,labels,docid,token_indices))
            if end>=len(ids):break
            start+=step
    return data

def train_model(items, tokenizer, epochs=4, seed=13, class_weights=(1.0,3.0,2.0), batch_size=8, lr=3e-5, base_model=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data=encode_items(items,tokenizer)
    model=AutoModelForTokenClassification.from_pretrained(base_model or MODEL_ID,num_labels=3,ignore_mismatched_sizes=True)
    model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    weights=torch.tensor(class_weights,dtype=torch.float32)
    for epoch in range(epochs):
        order=list(range(len(data)));random.shuffle(order); losses=[]
        for ix in range(0,len(order),batch_size):
            batch=[data[j] for j in order[ix:ix+batch_size]]
            ids=torch.tensor([x[0] for x in batch],dtype=torch.long)
            mask=torch.tensor([x[1] for x in batch],dtype=torch.long)
            labels=torch.tensor([x[2] for x in batch],dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits=model(input_ids=ids,attention_mask=mask).logits
            loss=torch.nn.functional.cross_entropy(logits.view(-1,3),labels.view(-1),weight=weights,ignore_index=-100)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
        print('epoch',epoch+1,'loss',round(sum(losses)/max(1,len(losses)),4),'chunks',len(data),flush=True)
    model.eval()
    return model,data

@torch.no_grad()
def predict_model(model, items, tokenizer, data=None, batch_size=8):
    if data is None:data=encode_items(items,tokenizer)
    aggregate=[]
    for _,_,text,_ in items:
        n=len(tokenizer(text,add_special_tokens=False)['input_ids'])
        aggregate.append([np.zeros(3,dtype=np.float64) for _ in range(n)])
    counts=[np.zeros(len(a),dtype=np.float64) for a in aggregate]
    for ix in range(0,len(data),batch_size):
        batch=data[ix:ix+batch_size]
        ids=torch.tensor([x[0] for x in batch],dtype=torch.long)
        mask=torch.tensor([x[1] for x in batch],dtype=torch.long)
        logits=model(input_ids=ids,attention_mask=mask).logits.detach().cpu().numpy()
        for row,(_,_,_,docid,token_indices) in enumerate(batch):
            for pos,tok in enumerate(token_indices):
                if tok is None:continue
                aggregate[docid][tok]+=logits[row,pos]
                counts[docid][tok]+=1
    probs=[]
    for dd,cc in zip(aggregate,counts):
        arr=np.stack([x/max(1,c) for x,c in zip(dd,cc)]) if dd else np.zeros((0,3))
        # Stable softmax
        arr=arr-arr.max(axis=1,keepdims=True); ex=np.exp(arr); probs.append(ex/ex.sum(axis=1,keepdims=True))
    return probs

def decode_items(items, probs, tokenizer, bth=.5, ith=.5):
    pred=set()
    for (pmid,sec,text,anns),pp in zip(items,probs):
        enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        offsets=enc['offset_mapping']; labels=[]
        for q in pp:
            b,i,o=q[1],q[2],q[0]
            if b>o*bth and b>=i:labels.append('B')
            elif i>o*ith:labels.append('I')
            else:labels.append('O')
        spans=[];start=None;end=None
        for lab,(a,b) in zip(labels,offsets):
            if a==b:continue
            if lab=='B':
                if start is not None:spans.append((start,end))
                start=a;end=b
            elif lab=='I':
                if start is None:start=a
                end=b
            else:
                if start is not None:spans.append((start,end));start=None;end=None
        if start is not None:spans.append((start,end))
        pred.update((pmid,sec,a,b)for a,b in spans if a<b)
    return pred

def score_params(items, probs, tokenizer, gold, pairs=None):
    if pairs is None:pairs=[(1.0,1.0),(.75,.75),(.6,.6),(.5,.5),(.4,.4),(.3,.3),(.2,.2),(.1,.1),(.5,.25),(.25,.5),(.25,.1)]
    for b,i in pairs:
        pred=decode_items(items,probs,tokenizer,b,i)
        print('thresholds',b,i,ner.score(pred,gold),flush=True)

def main():
    torch.set_num_threads(min(8,torch.get_num_threads()))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    tr=ner.load_articles('training');ta=ner.load_annotations('training')
    dv=ner.load_articles('development');da=ner.load_annotations('development')
    items=make_items(tr,ta); vitems=make_items(dv,da)
    model,data=train_model(items,tokenizer,epochs=4)
    probs=predict_model(model,vitems,tokenizer)
    gold={(p,s,a,b)for p,ss in da.items()for s,aa in ss.items()for a,b,_,_ in aa}
    score_params(vitems,probs,tokenizer,gold)
    with open('/workspace/dev_bert_probs.pkl','wb') as f:
        pickle.dump(([(p,s,t) for p,s,t,_ in vitems],probs),f,protocol=4)

if __name__=='__main__':main()
