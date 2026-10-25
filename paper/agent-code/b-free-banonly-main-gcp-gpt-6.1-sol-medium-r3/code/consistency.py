import argparse, bisect, json, os
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
from ner import read_docs,read_gold,Corpus,decode,metrics,write_predictions

def smooth(logits,corpus,blend=.5):
    initial=decode(logits,corpus)
    candidates=defaultdict(set)
    for p,s,b,e in initial:
        candidates[p].add((s,b,e))
    out=[]
    for d,offs,score in zip(corpus.docs,corpus.offsets,logits):
        scores=score.copy();text=d['text'];starts={b:i for i,(b,e) in enumerate(offs)};ends={e:i+1 for i,(b,e) in enumerate(offs)}
        surfaces=set();occupied=[]
        for s,b,e in candidates[d['pmid']]:
            shift=0 if s=='T' else d['split'];b+=shift;e+=shift;occupied.append((b,e))
            surface=text[b:e]
            if len(surface)>=3:surfaces.add(surface)
        for surface in surfaces:
            spans=[];pos=0
            while True:
                b=text.find(surface,pos)
                if b<0:break
                e=b+len(surface);pos=e
                if b not in starts or e not in ends:continue
                if (b>0 and text[b-1].isalnum()) or (e<len(text) and text[e].isalnum()):continue
                if any(x<e and y>b and (x,y)!=(b,e) for x,y in occupied):continue
                spans.append((starts[b],ends[e]))
            if len(spans)<2:continue
            lens={e-b for b,e in spans}
            if len(lens)!=1:continue
            avg=np.mean([score[b:e] for b,e in spans],axis=0)
            for b,e in spans:scores[b:e]=(1-blend)*score[b:e]+blend*avg
        out.append(scores)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--logits',required=True);ap.add_argument('--tokenizer',default='/workspace/code/tokenizer');ap.add_argument('--data',default='/workspace/data');ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);ap.add_argument('--blend',type=float,default=.5);ap.add_argument('--bias',type=float,default=0);ap.add_argument('--method',default='greedy');ap.add_argument('--tune',action='store_true')
    a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');corpus=Corpus(docs,AutoTokenizer.from_pretrained(a.tokenizer));f=np.load(a.logits);logits=[f[f'arr_{i}'] for i in range(len(docs))]
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs);best=(0,None,None,None,None)
        for blend in [0,.25,.5,.75,1.]:
            scores=smooth(logits,corpus,blend) if blend else logits
            for method in ['greedy','viterbi']:
                for bias in [0,.25,.5]:
                    pred=decode(scores,corpus,bias,method);m=metrics(pred,gold)
                    print(json.dumps(dict(blend=blend,bias=bias,method=method,**m)),flush=True)
                    if m['f1']>best[0]:best=(m['f1'],blend,bias,method,pred)
        _,blend,bias,method,pred=best
        json.dump(dict(blend=blend,bias=bias,method=method,metrics=metrics(pred,gold)),open(a.out+'/consistency.json','w'),indent=2)
    else:pred=decode(smooth(logits,corpus,a.blend),corpus,a.bias,a.method)
    write_predictions(pred,a.out+f'/{a.split}.tsv')

if __name__=='__main__':main()
