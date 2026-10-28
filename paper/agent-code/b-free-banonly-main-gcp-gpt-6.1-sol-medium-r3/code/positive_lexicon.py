import argparse,json,os
from collections import Counter,defaultdict
import ahocorasick
from ner import read_docs,read_gold,write_predictions,metrics
from filter_rules import load_predictions,filter_mentions

def case_sensitive(name):return len(name)<=5 or any(c.isupper() for c in name[1:])
def make_automata(names):
    exact=ahocorasick.Automaton();fold=ahocorasick.Automaton()
    for name in names:
        a=exact if case_sensitive(name) else fold;key=name if case_sensitive(name) else name.casefold()
        a.add_word(key,key)
    exact.make_automaton();fold.make_automaton();return exact,fold

def matches(text,automata):
    for automaton,source in zip(automata,[text,text.casefold()]):
        for end,name in automaton.iter(source):
            b=end-len(name)+1;e=end+1
            if (b>0 and text[b-1].isalnum()) or (e<len(text) and text[e].isalnum()):continue
            yield b,e,name

def build(data,splits):
    names=set();documents=[];gold_by_id=defaultdict(set)
    for split in splits:
        docs=read_docs(f'{data}/{split}.abstracts.txt');documents+=docs
        spans,gold=read_gold(f'{data}/{split}.annotations.txt',docs)
        for p,s,b,e in gold:gold_by_id[p].add((s,b,e))
        for l in open(f'{data}/{split}.annotations.txt'):names.add(l.rstrip('\n').split('\t')[4])
    automata=make_automata(names);positive=Counter();total=Counter();papers=defaultdict(set)
    for d in documents:
        for b,e,name in matches(d['text'],automata):
            if e<d['split']:span=('T',b,e)
            elif b>=d['split']:span=('A',b-d['split'],e-d['split'])
            else:continue
            total[name]+=1
            if span in gold_by_id[d['pmid']]:positive[name]+=1;papers[name].add(d['pmid'])
    return {n:dict(positive=positive[n],total=total[n],papers=len(papers[n])) for n in total if positive[n]}

def add(pred,docs,stats,minimum,ratio,papers=1):
    accepted={n for n,s in stats.items() if s['positive']>=minimum and s['positive']/s['total']>=ratio and s['papers']>=papers}
    automata=make_automata(accepted);out=set(pred);existing=defaultdict(list)
    for p,s,b,e in pred:existing[p,s].append((b,e))
    for d in docs:
        for b,e,name in matches(d['text'],automata):
            if e<d['split']:s='T'
            elif b>=d['split']:s='A';b-=d['split'];e-=d['split']
            else:continue
            if any(x<e and y>b for x,y in existing[d['pmid'],s]):continue
            out.add((d['pmid'],s,b,e))
    return filter_mentions(out,docs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/workspace/data');ap.add_argument('--splits',nargs='+',default=['training']);ap.add_argument('--predictions',required=True);ap.add_argument('--split',default='development');ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--minimum',type=int,default=5);ap.add_argument('--ratio',type=float,default=.99);ap.add_argument('--papers',type=int,default=1);ap.add_argument('--tune',action='store_true');a=ap.parse_args()
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');pred=load_predictions(a.predictions)
    if os.path.exists(a.cache):stats=json.load(open(a.cache))
    else:stats=build(a.data,a.splits);json.dump(stats,open(a.cache,'w'))
    if a.tune:
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs);best=(0,None,None,None,None)
        for minimum in [2,3,5,10]:
            for ratio in [.95,.98,1.]:
                for papers in [1,2]:
                    augmented=add(pred,docs,stats,minimum,ratio,papers);m=metrics(augmented,gold)
                    print(json.dumps(dict(minimum=minimum,ratio=ratio,papers=papers,**m)),flush=True)
                    if m['f1']>best[0]:best=(m['f1'],minimum,ratio,papers,augmented)
        _,minimum,ratio,papers,augmented=best
        json.dump(dict(minimum=minimum,ratio=ratio,papers=papers,metrics=metrics(augmented,gold)),open(a.out+'.json','w'),indent=2)
    else:augmented=add(pred,docs,stats,a.minimum,a.ratio,a.papers)
    write_predictions(augmented,a.out)

if __name__=='__main__':main()
