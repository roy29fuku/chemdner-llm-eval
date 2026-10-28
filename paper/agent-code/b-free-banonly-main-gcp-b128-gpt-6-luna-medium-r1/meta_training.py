"""Out-of-fold predictions and span-level model for final CEM selection."""
import gc,random,sys
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/workspace')
import ner,bert_ner,candidate_model as cm
from transformers import AutoTokenizer

MODEL_ID='distilbert/distilbert-base-cased'
BERT_PAIRS=[(1.,1.),(.75,.75),(.5,.5),(.3,.3),(.2,.2),(.1,.1),(.5,.25),(.25,.5),(.25,.1),(.4,.1),(.5,.1)]

def merge(files):
    docs={}; anns=defaultdict(lambda:defaultdict(list))
    for f in files:
        docs.update(ner.load_articles(f)); aa=ner.load_annotations(f)
        for p,ss in aa.items():
            for sec,ls in ss.items(): anns[p][sec].extend(ls)
    return docs,anns

def known_strings(anns):
    return {a[2].lower() for ss in anns.values() for ls in ss.values() for a in ls}

def make_candidate_rows(docs,anns,knownanns):
    known=known_strings(knownanns)
    knownseq={tuple(t[0].lower() for t in ner.tokenize(s)) for s in known if ner.tokenize(s)}
    X=[]; y=[]; meta=[]
    for p,secs in docs.items():
        for sec,text in secs.items():
            toks,cs=cm.cands_for(text,knownseq,False)
            gold={(a,b) for a,b,_,_ in anns.get(p,{}).get(sec,[])}
            for c in cs:
                X.append(cm.featurize(text,toks,c,sec=='T',known))
                y.append(int((c[2],c[3]) in gold)); meta.append((p,sec,text,toks,c))
    return X,np.asarray(y),meta

def direct_matches(docs, knownanns):
    seqs={tuple(t[0].lower() for t in ner.tokenize(a[2])) for ss in knownanns.values() for ls in ss.values() for a in ls if ner.tokenize(a[2])}
    out=set()
    for p,secs in docs.items():
        for sec,text in secs.items():
            toks=ner.tokenize(text); lo=[t[0].lower() for t in toks]
            for i in range(len(lo)):
                for n in range(1,min(12,len(lo)-i)+1):
                    seq=tuple(lo[i:i+n]); base=seq
                    if seq not in seqs and len(seq[-1])>4 and seq[-1].endswith('s'):
                        base=seq[:-1]+(seq[-1][:-1],)
                    if base in seqs and toks[i+n-1][2]-toks[i][1]>=3:
                        out.add((p,sec,toks[i][1],toks[i+n-1][2]))
    return out

def bertsources(items, probs, tokenizer):
    return {pair:bert_ner.decode_items(items,probs,tokenizer,*pair) for pair in BERT_PAIRS}

def span_features(p,sec,text,start,end, bert_probs, tokenizer, dict_prob, dict_feat, direct, crf, bertflags):
    phrase=text[start:end]; lower=phrase.lower(); chars=''.join(c for c in lower if c.isalpha())
    toks=ner.tokenize(text); ti=[i for i,(_,a,b) in enumerate(toks) if a<end and b>start]
    words=[t[0].lower() for t in toks]
    idx0=ti[0] if ti else 0; idx1=ti[-1] if ti else -1
    prev=words[idx0-1] if idx0>0 else '<BOS>'; nxt=words[idx1+1] if idx1+1<len(words) else '<EOS>'
    ptoks=ner.tokenize(phrase)
    f={'span':lower,'section_title':sec=='T','span_chars':min(len(phrase),100),'span_tokens':min(len(ptoks),12),
       'shape':ner.shape(phrase),'prev':prev,'next':nxt,'prev_next':prev+'|'+nxt,
       'dict.source':dict_feat is not None,'dict.prob':float(dict_prob or 0),'known.direct':(p,sec,start,end) in direct,
       'crf.source':(p,sec,start,end) in crf}
    for k,v in bertflags.items():f['bert.flag.'+str(k)]=(p,sec,start,end) in v
    for n in (2,3,4,5,6):
        if len(chars)>=n:f['prefix'+str(n)]=chars[:n];f['suffix'+str(n)]=chars[-n:]
    for marker in ('benz','phenyl','pyrid','thiaz','hydroxy','methyl','ethyl','chloro','fluoro','sulf','phosph','glycos','oxide','amine','amide','acid','ene','ine','ol','one','ide','ate','ose','yl'):
        f['morph.'+marker]=marker in chars
    if dict_feat:
        for k,v in dict_feat.items():
            f['dictfeat.'+str(k)]=v
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True); offsets=enc['offset_mapping']
    overlap=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if overlap and bert_probs is not None:
        pp=bert_probs[overlap]
        f.update({'bert.startB':float(bert_probs[overlap[0],1]),'bert.firstI':float(bert_probs[overlap[0],2]),
                  'bert.endI':float(bert_probs[overlap[-1],2]),'bert.meanEntity':float(pp[:,1:].sum(axis=1).mean()),
                  'bert.maxEntity':float(pp[:,1:].sum(axis=1).max()),'bert.minEntity':float(pp[:,1:].sum(axis=1).min()),
                  'bert.startVsO':float(bert_probs[overlap[0],1]-bert_probs[overlap[0],0]),
                  'bert.endVsO':float(bert_probs[overlap[-1],2]-bert_probs[overlap[-1],0])})
    return f

def fold_rows(docs,anns,valid_ids,tokenizer,bert_epochs=3,seed=13):
    train_docs={p:x for p,x in docs.items() if p not in valid_ids}; valid_docs={p:x for p,x in docs.items() if p in valid_ids}
    train_anns={p:anns[p] for p in train_docs}; valid_anns={p:anns[p] for p in valid_docs}
    # Candidate model, with term knowledge restricted to the other articles.
    CX,cy,cm_train=make_candidate_rows(train_docs,train_anns,train_anns)
    VX,vy,vm=make_candidate_rows(valid_docs,valid_anns,train_anns)
    vec=DictVectorizer(); xx=vec.fit_transform(CX); vv=vec.transform(VX)
    for mat in (xx,vv):mat.indices=mat.indices.astype(np.int32);mat.indptr=mat.indptr.astype(np.int32)
    clf=LogisticRegression(C=.15,max_iter=250,solver='lbfgs').fit(xx,cy)
    dprob=clf.predict_proba(vv)[:,list(clf.classes_).index(1)]
    dict_data={}; dict_feats={}
    for meta,prob,feat in zip(vm,dprob,VX):
        p,sec,text,toks,c=meta; key=(p,sec,c[2],c[3])
        if prob>dict_data.get(key,-1):dict_data[key]=float(prob);dict_feats[key]=feat
    direct=direct_matches(valid_docs,train_anns)
    # CRF model.
    kterms=ner.terms_from_annotations(train_anns)
    tx,ty,_=ner.build_data(train_docs,train_anns,kterms); vx,_,vmeta=ner.build_data(valid_docs,valid_anns,kterms)
    crf_model=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=.02,c2=.1,max_iterations=80,all_possible_transitions=True)
    crf_model.fit(tx,ty); pred=crf_model.predict(vx)
    crfpred={(p,s,a,b) for (p,s,t,toks,ann),labs in zip(vmeta,pred) for a,b in ner.spans_from_labels(toks,labs)}
    # Transformer model.
    train_items=bert_ner.make_items(train_docs,train_anns); valid_items=bert_ner.make_items(valid_docs,valid_anns)
    model,train_data=bert_ner.train_model(train_items,tokenizer,epochs=bert_epochs,seed=seed)
    probs=bert_ner.predict_model(model,valid_items,tokenizer)
    bflags=bertsources(valid_items,probs,tokenizer)
    candidate_set=set(dict_data)|direct|crfpred
    for s in bflags.values():candidate_set.update(s)
    lookup={(p,sec):i for i,(p,sec,text,a) in enumerate(valid_items)}
    rows=[]; labels=[]
    gold={(p,sec,a,b) for p,ss in valid_anns.items() for sec,ls in ss.items() for a,b,_,_ in ls}
    for p,sec,start,end in candidate_set:
        i=lookup[(p,sec)]; _,_,text,_=valid_items[i]
        feat=span_features(p,sec,text,start,end,probs[i],tokenizer,dict_data.get((p,sec,start,end)),dict_feats.get((p,sec,start,end)),direct,crfpred,bflags)
        rows.append((p,sec,start,end,feat));labels.append(int((p,sec,start,end) in gold))
    del model,train_data,tx,ty,vx,vec,clf,probs
    gc.collect()
    return rows,np.asarray(labels)

def main():
    torch.set_num_threads(min(8,torch.get_num_threads()))
    docs,anns=merge(['training','development']); ids=list(docs); random.Random(17).shuffle(ids)
    folds=[set(ids[i::4]) for i in range(4)]; tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    allrows=[]; alllabels=[]
    for k,valid in enumerate(folds):
        rows,labels=fold_rows(docs,anns,valid,tokenizer,bert_epochs=3,seed=23+k)
        allrows.extend(rows); alllabels.extend(labels.tolist())
        print('fold',k,'candidates',len(rows),'positive',int(labels.sum()),flush=True)
        with open('/workspace/oof_meta_partial.pkl','wb') as f:
            import pickle;pickle.dump((allrows,np.asarray(alllabels)),f,protocol=4)
    with open('/workspace/oof_meta.pkl','wb') as f:
        import pickle;pickle.dump((allrows,np.asarray(alllabels)),f,protocol=4)
    # Cross-validate the second-level model by held-out article.
    oofids=list({r[0] for r in allrows});random.Random(31).shuffle(oofids); mf=[set(oofids[i::4]) for i in range(4)]
    gold={(p,s,a,b) for p,ss in anns.items() for s,ls in ss.items() for a,b,_,_ in ls}
    meta_pred={th:set() for th in [.1,.2,.3,.4,.5,.6,.7,.8]}
    for k,vp in enumerate(mf):
        trix=[i for i,r in enumerate(allrows) if r[0] not in vp]; vaix=[i for i,r in enumerate(allrows) if r[0] in vp]
        dv=DictVectorizer(); tx=dv.fit_transform([allrows[i][4] for i in trix]); vx=dv.transform([allrows[i][4] for i in vaix])
        for mat in (tx,vx):mat.indices=mat.indices.astype(np.int32);mat.indptr=mat.indptr.astype(np.int32)
        model=LogisticRegression(C=.15,max_iter=300,solver='lbfgs').fit(tx,np.asarray(alllabels)[trix])
        probs=model.predict_proba(vx)[:,list(model.classes_).index(1)]
        for th in meta_pred:
            meta_pred[th].update((allrows[i][0],allrows[i][1],allrows[i][2],allrows[i][3]) for i,pr in zip(vaix,probs) if pr>=th)
        print('meta fold',k,'rows',len(trix),len(vaix),flush=True)
    for th,pred in meta_pred.items():print('META',th,ner.score(pred,gold),flush=True)

if __name__=='__main__':main()
