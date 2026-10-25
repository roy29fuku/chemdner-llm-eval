import bisect,collections,json,pickle,re
from lex_eval import load,make_automaton
from lex_clf import surfaces,augment_aliases,candidate_docs,build_features,WORD_RE,featurize

def main():
    trdocs,tgold=load('training'); devdocs,dgold=load('development'); testdocs,_=load('test')
    # Final lexical model uses every provided labeled article.
    alldocs={**trdocs,**devdocs}; allgold=tgold|dgold
    fullstats=augment_aliases(surfaces(alldocs,allgold)); pats=list(fullstats); go,fail,out=make_automaton(pats)
    fullcand=candidate_docs(testdocs,pats,go,fail,out); fx,fm=build_features(testdocs,fullcand,fullstats)
    fvec,fclf,_,fthreshold=pickle.load(open('/workspace/lexclf_full.pkl','rb'))
    probs=fclf.predict_proba(fvec.transform(fx))[:,1]
    base={x for x,z in zip(fm,probs) if z>=fthreshold}

    # The sequence model contributes only spans absent from the training lexicon.
    bert=set()
    for line in open('/workspace/output/bert_test.tsv',encoding='utf8'):
        p,spec=line.rstrip('\n').split('\t'); sec,a,b=spec.split(':'); bert.add((p,sec,int(a),int(b)))
    trstats=augment_aliases(surfaces(trdocs,tgold)); tpats=list(trstats); tgo,tfail,tout=make_automaton(tpats)
    traincand=candidate_docs(testdocs,tpats,tgo,tfail,tout)
    extra=[x for x in bert if x not in traincand and x not in base]
    avec,aclf,_,_=pickle.load(open('/workspace/lexclf_alias.pkl','rb'))
    token_cache={}; rows=[]
    for p,sec,a,b in extra:
        text=testdocs[p][sec]
        if (p,sec) not in token_cache:
            toks=[(m.start(),m.end(),m.group().lower()) for m in WORD_RE.finditer(text)]
            token_cache[(p,sec)]=([z[0] for z in toks],[z[1] for z in toks],toks)
        starts,ends,toks=token_cache[(p,sec)]; q=text[a:b].lower()
        st=trstats.get(q,(0,set()))
        rows.append(featurize(q,text[a:b],a,b,starts,ends,toks,(st[0],len(st[1]))))
    if rows:
        scores=aclf.predict_proba(avec.transform(rows))[:,1]
        add={x for x,z in zip(extra,scores) if z>=0.9}
    else: add=set()
    pred=base|add
    with open('/workspace/output/predictions.tsv','w',encoding='utf8') as f:
        for p,sec,a,b in sorted(pred,key=lambda x:(int(x[0]),x[1],x[2],x[3])):
            f.write(f'{p}\t{sec}:{a}:{b}\n')
    print(f'base={len(base)} bert_extra={len(extra)} accepted={len(add)} total={len(pred)} threshold={fthreshold:.6f}/0.9')

if __name__=='__main__': main()
