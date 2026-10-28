from ensemble import *
from postprocess import *
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('scores',nargs='+')
    parser.add_argument('--out',required=True)
    parser.add_argument('--lexicon',default='/workspace/output/training_lexicon.json')
    parser.add_argument('--boundary-map',default='')
    args=parser.parse_args()
    records=[pickle.load(open(p,'rb')) for p in args.scores]
    docs=records[0]['docs'];gold=gold_spans(docs)
    scores=[project_scores(r['docs'],r['scores'],docs) for r in records]
    lex=json.load(open(args.lexicon))
    boundary_map=json.load(open(args.boundary_map)) if args.boundary_map else None
    # First select model combination and entity threshold, then secondary corrections.
    best=(-1,None,None)
    combinations=[]
    n=len(scores)
    for i in range(n):
        w=[0.]*n;w[i]=1.;combinations.append(w)
    if n>1:
        combinations.append([1/n]*n)
        for i in range(n):
            w=[.5/(n-1)]*n;w[i]=.5;combinations.append(w)
        if n==2:combinations += [[.25,.75],[.75,.25]]
        if n>=3:
            for excluded in range(n):
                w=[1/(n-1)]*n;w[excluded]=0.;combinations.append(w)
            for i in range(n):
                for j in range(i+1,n):
                    w=[0.]*n;w[i]=.5;w[j]=.5;combinations.append(w)
    combinations=list({tuple(w):w for w in combinations}.values())
    results=[];mode_best={}
    for arithmetic in [False,True] if n>1 else [False]:
        for weights in combinations:
            merged=combine(scores,weights,arithmetic)
            for bias in [-.75,-.5,-.25,0,.25,.5,.75]:
                pred=filter_predictions(spans_from_scores(docs,merged,bias,boundary_map),docs,short=True)
                m=metric(pred,gold)
                setting=dict(weights=weights,arithmetic=arithmetic,bias=bias,lexicon_strength=0.,role_strength=0.,propagate_min_length=0,short_filter=True)
                results.append(dict(**setting,**m))
                if arithmetic not in mode_best or m['f1']>mode_best[arithmetic][0]:
                    mode_best[arithmetic]=(m['f1'],setting,merged)
                if m['f1']>best[0]:
                    best=(m['f1'],setting,merged)
                    print('BEST',json.dumps(dict(**setting,**m)),flush=True)
    if n>1:
        for arithmetic in mode_best:
            local_best=mode_best[arithmetic]
            for _ in range(2):
                base=local_best[1].copy();candidates=[]
                for i in range(n):
                    for alpha in [-.2,-.1,.1,.2]:
                        weights=[(1-alpha)*w for w in base['weights']];weights[i]+=alpha
                        if min(weights)<-1e-8:continue
                        weights=[max(0.,w) for w in weights]
                        norm=sum(weights);weights=[w/norm for w in weights]
                        candidates.append(weights)
                for weights in list({tuple(w):w for w in candidates}.values()):
                    merged=combine(scores,weights,base['arithmetic'])
                    for bias in [base['bias']-.25,base['bias'],base['bias']+.25]:
                        pred=filter_predictions(spans_from_scores(docs,merged,bias,boundary_map),docs,short=True)
                        m=metric(pred,gold)
                        setting=dict(base,weights=weights,bias=bias)
                        results.append(dict(**setting,**m))
                        if m['f1']>local_best[0]:local_best=(m['f1'],setting,merged)
                        if m['f1']>best[0]:
                            best=(m['f1'],setting,merged)
                            print('BEST',json.dumps(dict(**setting,**m)),flush=True)
    merged=best[2];base_setting=best[1].copy()
    for strength in [0,.5,1.,1.5,2.]:
        boosted=lexicon_boost(docs,merged,lex,strength) if strength else merged
        for bias in sorted(set([base_setting['bias']-.25,base_setting['bias'],base_setting['bias']+.25])):
            pred=filter_predictions(spans_from_scores(docs,boosted,bias,boundary_map),docs,short=True)
            for minlen in [0,3,4,6,8]:
                p=propagate(pred,docs,minlen) if minlen else pred
                m=metric(p,gold)
                setting=dict(base_setting,bias=bias,lexicon_strength=strength,propagate_min_length=minlen)
                results.append(dict(**setting,**m))
                if m['f1']>best[0]:
                    best=(m['f1'],setting,merged)
                    print('BEST',json.dumps(dict(**setting,**m)),flush=True)
    base_setting=best[1].copy()
    merged=best[2]
    if base_setting['lexicon_strength']:
        merged=lexicon_boost(docs,merged,lex,base_setting['lexicon_strength'])
    for strength in [.5,1.,2.,4.,8.]:
        boosted=lexicon_boost(docs,merged,lex,strength,min_count=1,min_precision=0,role_context=True)
        for bias in sorted(set([base_setting['bias']-.25,base_setting['bias'],base_setting['bias']+.25])):
            pred=filter_predictions(spans_from_scores(docs,boosted,bias,boundary_map),docs,short=True)
            for minlen in sorted(set([0,base_setting['propagate_min_length'],6])):
                p=propagate(pred,docs,minlen) if minlen else pred
                m=metric(p,gold)
                setting=dict(base_setting,bias=bias,role_strength=strength,propagate_min_length=minlen)
                results.append(dict(**setting,**m))
                if m['f1']>best[0]:
                    best=(m['f1'],setting,best[2])
                    print('BEST',json.dumps(dict(**setting,**m)),flush=True)
    json.dump(dict(scores=args.scores,settings=best[1],development_f1=best[0],results=results),open(args.out,'w'),indent=2)

if __name__=='__main__':main()
