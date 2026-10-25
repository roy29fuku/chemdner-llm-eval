import collections
import pycrfsuite
import ner

OUT=ner.OUT
tr=ner.read_articles(ner.DATA/'training.abstracts.txt')
ta=ner.read_annotations(ner.DATA/'training.annotations.txt')
dv=ner.read_articles(ner.DATA/'development.abstracts.txt')
da=ner.read_annotations(ner.DATA/'development.annotations.txt')
all_articles=dict(tr); all_articles.update(dv)
all_ann=collections.defaultdict(list)
for dataset in (ta,da):
    for key,spans in dataset.items(): all_ann[key].extend(spans)
lex=ner.lexicon_from(all_articles,all_ann)
print('combined data',len(all_articles),'articles',sum(map(len,all_ann.values())),'mentions','lexicon',len(lex),flush=True)
model=OUT/'final.crfsuite'
ner.train(all_articles,all_ann,lex,model,iters=140,c1=0.001,c2=0.001)
test=ner.read_articles(ner.DATA/'test.abstracts.txt')
tagger=pycrfsuite.Tagger(); tagger.open(str(model))
with open(OUT/'predictions.tsv','w',encoding='utf-8') as f:
    for pmid,sections in test.items():
        for sec,text in sections.items():
            toks=ner.tokenize(text)
            for s,e in ner.get_spans(toks,tagger.tag(ner.features(toks,lex))):
                f.write(f'{pmid}\t{sec}:{s}:{e}\n')
print('predictions generated',flush=True)
