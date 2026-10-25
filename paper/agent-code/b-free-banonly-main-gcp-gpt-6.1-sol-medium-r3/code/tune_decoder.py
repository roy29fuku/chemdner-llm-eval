import argparse,json,os
import numpy as np
from transformers import AutoTokenizer
from ner import read_docs,read_gold,Corpus,decode,metrics,write_predictions
from filter_rules import filter_mentions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--logits',required=True);ap.add_argument('--tokenizer',default='/workspace/code/tokenizer');ap.add_argument('--data',default='/workspace/data');ap.add_argument('--out',required=True);ap.add_argument('--split',default='development');ap.add_argument('--bias',type=float,default=0);ap.add_argument('--begin',type=float,default=0);ap.add_argument('--continuation',type=float,default=0);ap.add_argument('--tune',action='store_true');ap.add_argument('--bias-grid',nargs='+',type=float,default=[-.5,0,.5,1.]);ap.add_argument('--begin-grid',nargs='+',type=float,default=[0,.5,1.,1.5,2.]);a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');corpus=Corpus(docs,AutoTokenizer.from_pretrained(a.tokenizer));f=np.load(a.logits);logits=[f[f'arr_{i}'] for i in range(len(docs))]
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs);best=(0,None,None,None)
        for bias in a.bias_grid:
            for begin in a.begin_grid:
                pred=filter_mentions(decode(logits,corpus,bias,'viterbi',begin),docs);m=metrics(pred,gold)
                print(json.dumps(dict(bias=bias,begin=begin,**m)),flush=True)
                if m['f1']>best[0]:best=(m['f1'],bias,begin,pred)
        _,bias,begin,pred=best
        json.dump(dict(bias=bias,begin=begin,metrics=metrics(pred,gold)),open(a.out+'/decoder.json','w'),indent=2)
    else:pred=filter_mentions(decode(logits,corpus,a.bias,'viterbi',a.begin,a.continuation),docs)
    write_predictions(pred,a.out+f'/{a.split}.tsv')

if __name__=='__main__':main()
