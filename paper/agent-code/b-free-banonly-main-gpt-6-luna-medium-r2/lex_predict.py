import json, pickle, sys
from lex_eval import load, make_automaton
from lex_clf import surfaces, augment_aliases, candidate_docs, build_features

def main():
    trdocs,tgold=load('training')
    devdocs,dgold=load('development')
    testdocs,_=load('test')
    all_docs={**trdocs,**devdocs}; all_gold=tgold|dgold
    stats=augment_aliases(surfaces(all_docs,all_gold)); pats=list(stats); go,fail,out=make_automaton(pats)
    cand=candidate_docs(testdocs,pats,go,fail,out)
    features,meta=build_features(testdocs,cand,stats)
    vec,clf,_,threshold=pickle.load(open('/workspace/lexclf_full.pkl','rb'))
    scores=clf.predict_proba(vec.transform(features))[:,1]
    pred={x for x,s in zip(meta,scores) if s>=threshold}
    with open('/workspace/output/predictions.tsv','w',encoding='utf8') as f:
        for p,sec,a,b in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])):
            f.write(f'{p}\t{sec}:{a}:{b}\n')
    print(f'wrote {len(pred)} predictions for {len(testdocs)} articles at threshold {threshold:.6f}')

if __name__=='__main__': main()
