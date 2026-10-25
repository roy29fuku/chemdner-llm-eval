import collections, sys
import pycrfsuite
import ner

tr=ner.read_articles(ner.DATA/'training.abstracts.txt'); ta=ner.read_annotations(ner.DATA/'training.annotations.txt')
dv=ner.read_articles(ner.DATA/'development.abstracts.txt'); da=ner.read_annotations(ner.DATA/'development.annotations.txt')
counts=collections.Counter()
for (pmid,sec),spans in ta.items():
    text=tr[pmid][sec]
    for s,e,*_ in spans:
        alltok=ner.tokenize(text)
        ids=[i for i,(_,a,b) in enumerate(alltok) if a>=s and b<=e]
        if ids and alltok[ids[0]][1]==s and alltok[ids[-1]][2]==e:
            key=tuple(alltok[i][0].casefold() for i in ids)
            counts[key]+=1
lex=set(counts)
tagger=pycrfsuite.Tagger(); tagger.open('/workspace/output/best_train.crfsuite')
gold=set(); crf=set(); cats=collections.defaultdict(set)
for pmid,sections in dv.items():
 for sec,text in sections.items():
    toks=ner.tokenize(text); words=[w.casefold() for w,_,_ in toks]
    gold.update((pmid,sec,s,e) for s,e,*_ in da.get((pmid,sec),[]))
    crf.update((pmid,sec,s,e) for s,e in ner.get_spans(toks,tagger.tag(ner.features(toks,lex))))
    for i in range(len(toks)):
      for n in range(1,min(10,len(toks)-i)+1):
        key=tuple(words[i:i+n])
        if key in counts:
          cand=(pmid,sec,toks[i][1],toks[i+n-1][2])
          for minimum in (1,2,3,5,10):
            if counts[key]>=minimum and n>=2: cats[('len2freq',minimum)].add(cand)
            if counts[key]>=minimum and n>=3: cats[('len3freq',minimum)].add(cand)
            if counts[key]>=minimum and len(''.join(words[i:i+n]))>=12: cats[('chars12freq',minimum)].add(cand)
          for f in (1,2,3,5,10):
            if counts[key]>=f: cats[f].add(cand)
          if n>=2: cats['len2'].add(cand)
          if n>=3: cats['len3'].add(cand)

def score(pred):
 tp=len(pred&gold); p=tp/max(1,len(pred));r=tp/max(1,len(gold));return (round(2*p*r/(p+r),4),round(p,4),round(r,4),tp,len(pred))
print('gold',len(gold),'crf',score(crf))
for k,p in cats.items():print('lex',k,score(p),'unique',len(p))
for k,p in cats.items():
 if isinstance(k,tuple): print('union',k,score(crf|p),'unique',len(crf|p))
