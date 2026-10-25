import argparse,json,os
import numpy as np
from transformers import AutoTokenizer
from ner import read_docs,read_gold,Corpus,decode,metrics
from filter_rules import filter_mentions

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--logits',nargs='+',required=True);p.add_argument('--data',default='/workspace/data');p.add_argument('--pmids');p.add_argument('--out',required=True);a=p.parse_args()
    all_docs=read_docs(a.data+'/development.abstracts.txt');indices=list(range(len(all_docs)))
    if a.pmids:
        held=set(json.load(open(a.pmids)));indices=[i for i,d in enumerate(all_docs) if d['pmid'] in held]
    docs=[all_docs[i] for i in indices];_,gold=read_gold(a.data+'/development.annotations.txt',docs)
    best=(0,None,None,None)
    for path,logitpath in zip(a.models,a.logits):
        corpus=Corpus(docs,AutoTokenizer.from_pretrained(path));f=np.load(logitpath);scores=[f[f'arr_{i}'] for i in indices]
        for bias in [-1.,-.5,0,.5]:
            for begin in [0,.5,1.,1.5]:
                pred=filter_mentions(decode(scores,corpus,bias,'viterbi',begin),docs);m=metrics(pred,gold)
                print(json.dumps(dict(model=path,bias=bias,begin=begin,**m)),flush=True)
                if m['f1']>best[0]:best=(m['f1'],path,bias,begin)
    f1,model,bias,begin=best
    json.dump(dict(f1=f1,model=model,bias=bias,begin=begin,documents=len(docs)),open(a.out,'w'),indent=2)
    print('SELECTED',model,f1,flush=True)

if __name__=='__main__':main()
