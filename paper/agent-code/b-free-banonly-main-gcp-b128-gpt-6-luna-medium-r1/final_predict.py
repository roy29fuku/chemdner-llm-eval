"""Train on all supplied annotations and write test CEM spans."""
import os,pickle,sys
from collections import defaultdict
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer
sys.path.insert(0,'/workspace')
import ner,bert_ner,meta_training as mt
from candidate_model import cands_for,featurize

META_THRESHOLD=.30  # selected by out-of-fold article-level validation

def main():
    torch.set_num_threads(min(8,torch.get_num_threads()))
    with open('/workspace/oof_meta.pkl','rb') as f:oof_rows,oof_y=pickle.load(f)
    mvec=DictVectorizer(); mx=mvec.fit_transform([r[4] for r in oof_rows])
    mx.indices=mx.indices.astype(np.int32);mx.indptr=mx.indptr.astype(np.int32)
    meta=LogisticRegression(C=.15,max_iter=300,solver='lbfgs').fit(mx,oof_y)

    train_docs,train_anns=mt.merge(['training','development'])
    test_docs=ner.load_articles('test')
    known=ner.terms_from_annotations(train_anns)
    # Fit dictionary candidate model.
    CX,cy,_=mt.make_candidate_rows(train_docs,train_anns,train_anns)
    tv=DictVectorizer();tx=tv.fit_transform(CX);tx.indices=tx.indices.astype(np.int32);tx.indptr=tx.indptr.astype(np.int32)
    dclf=LogisticRegression(C=.15,max_iter=300,solver='lbfgs').fit(tx,cy)
    DX,_,dmeta=mt.make_candidate_rows(test_docs,{},train_anns)
    dx=tv.transform(DX);dx.indices=dx.indices.astype(np.int32);dx.indptr=dx.indptr.astype(np.int32)
    dprob=dclf.predict_proba(dx)[:,list(dclf.classes_).index(1)]
    dict_data={};dict_feats={}
    for item,prob,feat in zip(dmeta,dprob,DX):
        p,sec,text,toks,c=item;key=(p,sec,c[2],c[3])
        if prob>dict_data.get(key,-1):dict_data[key]=float(prob);dict_feats[key]=feat
    direct=mt.direct_matches(test_docs,train_anns)

    # CRF sequence predictions.
    crf=ner.sklearn_crfsuite.CRF(algorithm='lbfgs',c1=.02,c2=.1,max_iterations=120,all_possible_transitions=True)
    train_X,train_Y,_=ner.build_data(train_docs,train_anns,known);test_X,_,test_meta=ner.build_data(test_docs,{},known)
    crf.fit(train_X,train_Y);crf_labels=crf.predict(test_X)
    crfpred={(p,s,a,b)for(p,s,t,toks,ann),labs in zip(test_meta,crf_labels)for a,b in ner.spans_from_labels(toks,labs)}

    # Fine tune the general language model on all provided article annotations.
    tokenizer=AutoTokenizer.from_pretrained(mt.MODEL_ID,use_fast=True)
    train_items=bert_ner.make_items(train_docs,train_anns);test_items=bert_ner.make_items(test_docs,{})
    bmodel,_=bert_ner.train_model(train_items,tokenizer,epochs=4,seed=13)
    probs=bert_ner.predict_model(bmodel,test_items,tokenizer)
    bflags=mt.bertsources(test_items,probs,tokenizer)

    candidate_set=set(dict_data)|direct|crfpred
    for spans in bflags.values():candidate_set.update(spans)
    item_index={(p,sec):i for i,(p,sec,text,anns) in enumerate(test_items)}
    feats=[];keys=[]
    for p,sec,start,end in candidate_set:
        i=item_index[(p,sec)];_,_,text,_=test_items[i];key=(p,sec,start,end)
        feats.append(mt.span_features(p,sec,text,start,end,probs[i],tokenizer,dict_data.get(key),dict_feats.get(key),direct,crfpred,bflags))
        keys.append(key)
    q=mvec.transform(feats);q.indices=q.indices.astype(np.int32);q.indptr=q.indptr.astype(np.int32)
    scores=meta.predict_proba(q)[:,list(meta.classes_).index(1)]
    predictions={k for k,s in zip(keys,scores) if s>=META_THRESHOLD}

    os.makedirs('/workspace/output',exist_ok=True)
    order={p:i for i,p in enumerate(test_docs)}
    secorder={'T':0,'A':1}
    with open('/workspace/output/predictions.tsv','w',encoding='utf-8') as f:
        for p,sec,s,e in sorted(predictions,key=lambda x:(order[x[0]],secorder[x[1]],x[2],x[3])):
            f.write(f'{p}\t{sec}:{s}:{e}\n')
    print('written',len(predictions),'test predictions',flush=True)

if __name__=='__main__':main()
