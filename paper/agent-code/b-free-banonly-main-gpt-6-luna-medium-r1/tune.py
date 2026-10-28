import collections, shutil
import pycrfsuite, ner

tr=ner.read_articles(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt')
dv=ner.read_articles(ner.DATA/'development.abstracts.txt'); da=ner.read_annotations(ner.DATA/'development.annotations.txt')
lex=ner.lexicon_from(tr,ta)
gold={(pmid,sec,s,e) for (pmid,sec),spans in da.items() for s,e,*_ in spans}
best=0
for c1,c2 in [(0.005,0.001),(0.01,0.01),(0.03,0.01)]:
    path='/workspace/output/tune_char_params.crfsuite'
    trainer=pycrfsuite.Trainer(verbose=False)
    n,nt,miss=ner.append_corpus(trainer,tr,ta,lex)
    print('training',c1,c2,'tokens',nt,'mentions',n,flush=True)
    trainer.set_params({'c1':c1,'c2':c2,'max_iterations':110,'feature.possible_transitions':True})
    trainer.train(path)
    tg=pycrfsuite.Tagger();tg.open(path);pred=set()
    for pmid,sections in dv.items():
      for sec,text in sections.items():
        toks=ner.tokenize(text)
        pred.update((pmid,sec,s,e) for s,e in ner.get_spans(toks,tg.tag(ner.features(toks,lex))))
    tp=len(pred&gold);p=tp/max(1,len(pred));r=tp/len(gold);f=2*p*r/(p+r)
    print('score %.5f P %.4f R %.4f tp/pred %d/%d'%(f,p,r,tp,len(pred)),flush=True)
    if f>best:
       best=f;shutil.copy(path,'/workspace/output/best_train_char_params.crfsuite')
print('best',best,flush=True)
