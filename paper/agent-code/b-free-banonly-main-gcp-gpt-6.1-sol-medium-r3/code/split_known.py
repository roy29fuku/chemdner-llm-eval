import argparse,json,re
from collections import Counter,defaultdict
from functools import lru_cache
from ner import read_docs,read_gold,metrics,write_predictions
from filter_rules import load_predictions,filter_mentions

def build(data,splits):
    names=defaultdict(Counter)
    for split in splits:
        for line in open(f'{data}/{split}.annotations.txt'):
            p,s,b,e,text,cls=line.rstrip('\n').split('\t');names[text.casefold()][cls]+=1
    return names

def apply(pred,docs,names,minimum=1,families=False,enzyme_only=False):
    byid={d['pmid']:d for d in docs};out=set()
    for p,s,b,e in pred:
        d=byid[p];shift=0 if s=='T' else d['split'];text=d['text'][b+shift:e+shift]
        if enzyme_only:
            following=d['text'][e+shift:e+shift+60]
            if not re.match(r'^[\s-]+[a-zA-Z]*(?:transferase|transaminase|cyclase|oxygenase|oxidase|kinase|hydrolase|dehydrogenase|reductase|lyase|synthase|synthetase|phosphatase|deaminase|mutase|isomerase|carboxylase)\b',following):
                out.add((p,s,b,e));continue
        if text.casefold() in names or (not enzyme_only and not any(c in text for c in '/:-')) or len(text)>180:
            out.add((p,s,b,e));continue
        separators=[i for i,c in enumerate(text) if c in '/:-' or (enzyme_only and c.isspace())]
        @lru_cache(None)
        def parse(start):
            best=None
            for end in separators+[len(text)]:
                if end<=start:continue
                piece=text[start:end];lb=len(piece)-len(piece.lstrip());re_=len(piece.rstrip());name=piece[lb:re_].casefold()
                counts=names.get(name)
                if not counts or sum(counts.values())<minimum:continue
                if enzyme_only and (len(name)<3 or name[0].isdigit()):continue
                if not families and set(counts)=={'FAMILY'} and '/' not in text:continue
                if end==len(text):tail=[]
                else:tail=parse(end+1)
                if tail is None:continue
                candidate=[(start+lb,start+re_)]+tail
                # Single-letter locants preceding named molecules belong to the name.
                if len(candidate)>1 and len(name)==1 and text[end]=='-':
                    next_name=text[candidate[1][0]:candidate[1][1]].casefold()
                    if 'FORMULA' not in names.get(next_name,{}):continue
                if best is None or len(candidate)<len(best):best=candidate
            return best
        pieces=parse(0)
        if pieces is None or len(pieces)<2:out.add((p,s,b,e))
        else:
            for lb,re_ in pieces:out.add((p,s,b+lb,b+re_))
    return filter_mentions(out,docs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--splits',nargs='+',default=['training']);ap.add_argument('--predictions',required=True);ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);ap.add_argument('--minimum',type=int,default=1);ap.add_argument('--families',action='store_true');ap.add_argument('--enzyme-only',action='store_true');ap.add_argument('--tune',action='store_true');a=ap.parse_args()
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');pred=load_predictions(a.predictions);names=build(a.data,a.splits)
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs);best=(0,None,None,None)
        print('before',metrics(pred,gold),flush=True)
        for minimum in [1,2,3,5]:
            for families in [False,True]:
                adjusted=apply(pred,docs,names,minimum,families,a.enzyme_only);m=metrics(adjusted,gold)
                print(json.dumps(dict(minimum=minimum,families=families,**m)),flush=True)
                if m['f1']>best[0]:best=(m['f1'],minimum,families,adjusted)
        _,minimum,families,adjusted=best
        json.dump(dict(minimum=minimum,families=families,metrics=metrics(adjusted,gold)),open(a.out+'.json','w'),indent=2)
    else:adjusted=apply(pred,docs,names,a.minimum,a.families,a.enzyme_only)
    write_predictions(adjusted,a.out)

if __name__=='__main__':main()
