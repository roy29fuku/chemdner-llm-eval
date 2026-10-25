import argparse, json, os, re
from collections import Counter
from ner import read_docs,read_gold,metrics,write_predictions
from filter_rules import load_predictions,filter_mentions

def build(data,splits):
    counts=Counter();positive=Counter()
    for split in splits:
        docs=read_docs(f'{data}/{split}.abstracts.txt')
        for line in open(f'{data}/{split}.annotations.txt'):
            positive[line.rstrip('\n').split('\t')[4].casefold()]+=1
        for d in docs:
            # Full whitespace-delimited lexical tokens. Keep chemical punctuation inside tokens.
            words=list(re.finditer(r'[^\s]+',d['text']))
            for i,w in enumerate(words):
                for j in range(i,min(i+3,len(words))):
                    name=d['text'][w.start():words[j].end()].strip('.,;:()[]{}"\'')
                    if '\n' in name:continue
                    counts[name.casefold()]+=1
    return {s:n for s,n in counts.items() if positive[s]==0 and n>=3}

def apply(pred,docs,lexicon,minimum):
    byid={d['pmid']:d for d in docs};out=set()
    for p,s,b,e in pred:
        d=byid[p];shift=0 if s=='T' else d['split'];text=d['text'][b+shift:e+shift].casefold()
        if lexicon.get(text,0)<minimum:out.add((p,s,b,e))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--splits',nargs='+',default=['training']);ap.add_argument('--predictions',required=True);ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);ap.add_argument('--minimum',type=int,default=10);ap.add_argument('--tune',action='store_true');a=ap.parse_args()
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');pred=filter_mentions(load_predictions(a.predictions),docs);lexicon=build(a.data,a.splits)
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs)
        best=(0,None,None)
        for minimum in [3,5,10,15,20,30,50,100,10000000]:
            filtered=apply(pred,docs,lexicon,minimum);m=metrics(filtered,gold)
            print(json.dumps(dict(minimum=minimum,**m)),flush=True)
            if m['f1']>best[0]:best=(m['f1'],minimum,filtered)
        _,minimum,filtered=best
        json.dump(dict(minimum=minimum,metrics=metrics(filtered,gold)),open(a.out+'.json','w'),indent=2)
    else:filtered=apply(pred,docs,lexicon,a.minimum)
    write_predictions(filtered,a.out)

if __name__=='__main__':main()
