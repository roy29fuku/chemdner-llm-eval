import argparse, json, os
import numpy as np
from transformers import AutoTokenizer
from ner import read_docs, read_gold, Corpus, decode, metrics, write_predictions,collapse_typed
from filter_rules import filter_mentions

def probabilities(scores):
    s=scores-scores.max(-1,keepdims=True);p=np.exp(s);return p/p.sum(-1,keepdims=True)

def project(source_probs,source_offsets,target_offsets,text_length):
    if source_offsets==target_offsets:return source_probs
    entity=np.zeros(text_length,dtype=np.float32);begin=np.zeros(text_length,dtype=np.float32)
    for (b,e),p in zip(source_offsets,source_probs):
        entity[b:e]=p[1]+p[2];begin[b]=p[1]
    sums=np.concatenate([[0],np.cumsum(entity,dtype=np.float64)])
    starts=np.array([b for b,e in target_offsets]);ends=np.array([e for b,e in target_offsets])
    ent=(sums[ends]-sums[starts])/np.maximum(1,ends-starts)
    b=np.minimum(ent,begin[starts]);return np.stack([1-ent,b,ent-b],axis=-1).astype(np.float32)

def load_aligned(paths,tokenizers,docs,canonical,indices=None):
    aligned=[]
    for path,tokpath in zip(paths,tokenizers):
        source=Corpus(docs,AutoTokenizer.from_pretrained(tokpath));f=np.load(path)
        aligned.append([project(probabilities(collapse_typed(f[f'arr_{indices[i] if indices is not None else i}'])),source.offsets[i],canonical.offsets[i],len(d['text'])) for i,d in enumerate(docs)])
    return aligned

def combine(arrays,weights):
    weights=np.array(weights)/sum(weights)
    return [np.log(np.maximum(sum(w*arrays[j][i] for j,w in enumerate(weights)),1e-8)) for i in range(len(arrays[0]))]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--split',default='development')
    ap.add_argument('--logits',nargs='+',required=True);ap.add_argument('--tokenizers',nargs='+',required=True);ap.add_argument('--canonical',default='/workspace/code/tokenizer')
    ap.add_argument('--pmids')
    ap.add_argument('--weights',nargs='+',type=float);ap.add_argument('--bias',type=float,default=0);ap.add_argument('--begin',type=float,default=0);ap.add_argument('--method',default='viterbi');ap.add_argument('--tune',action='store_true');ap.add_argument('--save-logits',action='store_true');ap.add_argument('--out',required=True)
    args=ap.parse_args();os.makedirs(args.out,exist_ok=True)
    all_docs=read_docs(f'{args.data}/{args.split}.abstracts.txt');indices=list(range(len(all_docs)))
    if args.pmids:
        allowed=set(json.load(open(args.pmids)));indices=[i for i,d in enumerate(all_docs) if d['pmid'] in allowed]
    docs=[all_docs[i] for i in indices];canon=Corpus(docs,AutoTokenizer.from_pretrained(args.canonical))
    arrays=load_aligned(args.logits,args.tokenizers,docs,canon,indices)
    weights=args.weights or [1.]*len(arrays)
    if args.tune:
        _,gold=read_gold(f'{args.data}/{args.split}.annotations.txt',docs)
        if len(arrays)==2:choices=[[w,1-w] for w in [0,.2,.35,.5,.65,.8,1.]]
        elif len(arrays)==3:choices=[[i/5,j/5,(5-i-j)/5] for i in range(6) for j in range(6-i)]
        elif len(arrays)==4:
            base=np.array(weights[:3])/sum(weights[:3]);choices=[list((1-w)*base)+[w] for w in [0,.2,.25,.35,.5,.65]]
        elif len(arrays)==5:
            base=np.array(weights[:3])/sum(weights[:3])
            choices=[list((1-u-v)*base)+[u,v] for u in [0,.2,.35,.5,.65] for v in [0,.2,.35,.5,.65] if u+v<=.85]
        else:choices=[weights]
        best=(0,None,None,None,None,None)
        for w in choices:
            logits=combine(arrays,w)
            method='viterbi'
            for bias in [-1.,-.5,0,.5]:
                for begin in [0,.5,1.,1.5]:
                    pred=filter_mentions(decode(logits,canon,bias,method,begin),docs);m=metrics(pred,gold)
                    print(json.dumps(dict(weights=w,method=method,bias=bias,begin=begin,**m)),flush=True)
                    if m['f1']>best[0]:best=(m['f1'],w,method,bias,begin,pred)
        _,weights,method,bias,begin,pred=best
        json.dump(dict(weights=weights,method=method,bias=bias,begin=begin,metrics=metrics(pred,gold)),open(args.out+'/ensemble.json','w'),indent=2)
    else:
        pred=filter_mentions(decode(combine(arrays,weights),canon,args.bias,args.method,args.begin),docs)
    if args.save_logits:np.savez_compressed(args.out+f'/{args.split}.logits.npz',*combine(arrays,weights))
    write_predictions(pred,args.out+f'/{args.split}.tsv')

if __name__=='__main__':main()
