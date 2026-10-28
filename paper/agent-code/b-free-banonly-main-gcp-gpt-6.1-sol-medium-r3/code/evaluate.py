import argparse, json, os
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from ner import read_docs,read_gold,Corpus,decode,metrics,write_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--logits',nargs='+',required=True);p.add_argument('--data',default='/workspace/data');p.add_argument('--tokenizer',default='/workspace/code/tokenizer');p.add_argument('--out',required=True)
    a=p.parse_args();os.makedirs(a.out,exist_ok=True)
    docs=read_docs(a.data+'/development.abstracts.txt');_,gold=read_gold(a.data+'/development.annotations.txt',docs)
    corpus=Corpus(docs,AutoTokenizer.from_pretrained(a.tokenizer))
    arrays=[]
    for path in a.logits:
        f=np.load(path);arrays.append([f[f'arr_{i}'] for i in range(len(docs))])
    logits=[sum(arr[i] for arr in arrays)/len(arrays) for i in range(len(docs))]
    best=(0,None,None,None)
    for method in ['greedy','viterbi']:
        for bias in [-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.]:
            pred=decode(logits,corpus,bias,method);m=metrics(pred,gold)
            print(method,bias,json.dumps(m),flush=True)
            if m['f1']>best[0]:best=(m['f1'],method,bias,pred)
    _,method,bias,pred=best
    write_predictions(pred,a.out+'/development.tsv')
    json.dump(dict(method=method,bias=bias,metrics=metrics(pred,gold)),open(a.out+'/decoder.json','w'),indent=2)
    byid={d['pmid']:d for d in docs}
    def txt(x):
        p,s,b,e=x;d=byid[p];shift=0 if s=='T' else d['split'];return d['text'][b+shift:e+shift]
    report=[]
    for typ,errors in [('FP',pred-gold),('FN',gold-pred)]:
        report.append((typ,Counter(txt(x) for x in errors).most_common(100)))
    json.dump(report,open(a.out+'/errors.json','w'),indent=2)
    print('BEST',method,bias,json.dumps(metrics(pred,gold)))

if __name__=='__main__':main()
