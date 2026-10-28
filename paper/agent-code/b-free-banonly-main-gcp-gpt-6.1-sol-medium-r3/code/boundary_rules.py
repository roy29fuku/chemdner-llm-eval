import argparse,json,os,re
from collections import Counter,defaultdict
from ner import read_docs,read_gold,metrics,write_predictions
from filter_rules import load_predictions,filter_mentions

def build(data,splits):
    whole=Counter();patterns=defaultdict(Counter)
    for split in splits:
        docs=read_docs(f'{data}/{split}.abstracts.txt');_,gold=read_gold(f'{data}/{split}.annotations.txt',docs)
        bysec=defaultdict(list)
        for p,s,b,e in gold:bysec[p,s].append((b,e))
        for d in docs:
            for section in ['T','A']:
                text=d['title'] if section=='T' else d['text'][d['split']:]
                spans=sorted(bysec[d['pmid'],section])
                for b,e in spans:whole[text[b:e].casefold()]+=1
                for i,(b,e) in enumerate(spans):
                    pieces=[(0,e-b)];last=e
                    for nb,ne in spans[i+1:i+5]:
                        gap=text[last:nb]
                        if len(gap)>15 or not re.fullmatch(r'[\s,/:+\-]*(?:(?:and|or|of|with|to|plus|protected)[\s,/:+\-]*)?',gap):break
                        pieces.append((nb-b,ne-b));last=ne
                        key=text[b:ne].casefold()
                        if len(key)>180:break
                        patterns[key][tuple(pieces)]+=1
    return {key:dict(count=counts.most_common(1)[0][1],pieces=counts.most_common(1)[0][0],whole=whole[key]) for key,counts in patterns.items()}

def trim(name):
    b=0;e=len(name)
    while b<e and name[b] in ' \t\n,;:':b+=1
    while e>b and name[e-1] in ' \t\n,;:.®™':e-=1
    if name[b:e].endswith("'s"):e-=2
    # Remove an outer pair of round brackets only if it encloses the entire name.
    while e-b>2 and name[b]=='(' and name[e-1]==')':
        depth=0;outer=True
        for i in range(b,e):
            if name[i]=='(':depth+=1
            elif name[i]==')':depth-=1
            if depth==0 and i<e-1:outer=False;break
        if not outer or depth!=0:break
        b+=1;e-=1
    while e>b and name[e-1]==')' and name[b:e].count(')')>name[b:e].count('('):e-=1
    return b,e

def apply(pred,docs,patterns,minimum=1,ratio=.8,do_trim=True):
    byid={d['pmid']:d for d in docs};out=set()
    for p,s,b,e in pred:
        d=byid[p];shift=0 if s=='T' else d['split'];name=d['text'][b+shift:e+shift]
        if do_trim:
            nb,ne=trim(name);b+=nb;e=b+ne-nb;name=name[nb:ne]
        if not name:continue
        rule=patterns.get(name.casefold())
        if rule and rule['count']>=minimum and rule['count']/(rule['count']+rule['whole'])>=ratio:
            for nb,ne in rule['pieces']:out.add((p,s,b+nb,b+ne))
        else:out.add((p,s,b,e))
    return filter_mentions(out,docs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--splits',nargs='+',default=['training']);ap.add_argument('--predictions',required=True);ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--minimum',type=int,default=1);ap.add_argument('--ratio',type=float,default=.8);ap.add_argument('--tune',action='store_true');a=ap.parse_args()
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');pred=load_predictions(a.predictions)
    if os.path.exists(a.cache):patterns=json.load(open(a.cache))
    else:patterns=build(a.data,a.splits);json.dump(patterns,open(a.cache,'w'))
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs);best=(0,None,None,None)
        print('before',json.dumps(metrics(pred,gold)),flush=True)
        for minimum in [1,2,3,5,1000000]:
            for ratio in [.5,.8,1.]:
                adjusted=apply(pred,docs,patterns,minimum,ratio);m=metrics(adjusted,gold)
                print(json.dumps(dict(minimum=minimum,ratio=ratio,**m)),flush=True)
                if m['f1']>best[0]:best=(m['f1'],minimum,ratio,adjusted)
        _,minimum,ratio,adjusted=best
        json.dump(dict(minimum=minimum,ratio=ratio,metrics=metrics(adjusted,gold)),open(a.out+'.json','w'),indent=2)
    else:adjusted=apply(pred,docs,patterns,a.minimum,a.ratio)
    write_predictions(adjusted,a.out)

if __name__=='__main__':main()
