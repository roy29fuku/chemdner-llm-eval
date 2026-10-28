import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import sklearn_crfsuite

DATA = Path('/workspace/data')

# Unicode alphabetic runs plus punctuation. Split internal case and digit
# boundaries (e.g. MetAP2), while preserving each token's original offsets.
BASE = re.compile(r'[^\W\d_]+|\d+|[^\w\s]', re.UNICODE)
CASE_SPLIT = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)')

def tokenize(text):
    out = []
    for m in BASE.finditer(text):
        s, e, val = m.start(), m.end(), m.group()
        splits = [0] + [x.start() for x in CASE_SPLIT.finditer(val)] + [len(val)]
        for a,b in zip(splits, splits[1:]):
            if a < b:
                out.append((val[a:b], s+a, s+b))
    return out

def load_articles(which):
    rows = {}
    with open(DATA / f'{which}.abstracts.txt', encoding='utf-8') as f:
        for line in f:
            pmid, title, abstract = line.rstrip('\n').split('\t')
            rows[pmid] = {'T': title, 'A': abstract}
    return rows

def load_annotations(which):
    anns = defaultdict(lambda: defaultdict(list))
    with open(DATA / f'{which}.annotations.txt', encoding='utf-8') as f:
        for line in f:
            pmid, sec, s, e, text, cls = line.rstrip('\n').split('\t')
            anns[pmid][sec].append((int(s), int(e), text, cls))
    return anns

def shape(w):
    out=[]; last=''
    for c in w:
        cat = 'd' if c.isdigit() else ('A' if c.isupper() else ('a' if c.islower() else ('g' if c.isalpha() else c)))
        if cat != last: out.append(cat); last=cat
    return ''.join(out[:12])

CHEM_PREFIX = ('acet','amino','benz','bromo','but','carbox','chloro','cyclo','deoxy','di','ethyl','fluoro','gluco','hydro','hydroxy','iodo',' keto','methyl','methoxy','nitro','oxo','phenyl','phosph','poly','prop','sulfo','thio','tri')
CHEM_SUFFIX = ('acetate','acetamide','acid','al','alkane','alkene','alkyl','amine','amide','amidine','ate','ene',' ester','ether','ide','ine','inol','ium','ketone','nitrile','nucleotide','ol','one','ose','oxin','oxide','oyl','sulfate','sulfonate','yl','yne')

def features(tokens, is_title, known_terms):
    words = [t[0] for t in tokens]
    low = [w.lower() for w in words]
    # Whole observed name and term-subsequence features from supplied labels.
    known_start = [False]*len(tokens); known_inside = [False]*len(tokens)
    for i in range(len(tokens)):
        for n in range(1, min(7, len(tokens)-i)+1):
            phrase = tuple(low[i:i+n])
            if phrase in known_terms:
                known_start[i] = True
                for k in range(i+1,i+n): known_inside[k] = True
    out=[]
    for i,(w,s,e) in enumerate(tokens):
        l=low[i]
        f={
            'bias':1.0, 'word.lower':l, 'word':w,
            'shape':shape(w), 'len':min(len(w),20), 'title':is_title,
            'known.start':known_start[i], 'known.inside':known_inside[i],
            'known.word': (l,) in known_terms,
            'has_digit':any(c.isdigit() for c in w),
            'has_upper':any(c.isupper() for c in w),
            'all_upper':w.isupper(), 'initial_upper':w[:1].isupper(),
            'has_hyphen': '-' in w or '−' in w,
            'has_greek':any('GREEK' in __import__('unicodedata').name(c,'') for c in w),
            'starts_digit':w[:1].isdigit(), 'ends_digit':w[-1:].isdigit(),
            'chem.prefix':any(l.startswith(x) for x in CHEM_PREFIX),
            'chem.suffix':any(l.endswith(x.strip()) for x in CHEM_SUFFIX),
            'charclass': ''.join('d' if c.isdigit() else ('u' if c.isupper() else ('l' if c.islower() else 'p')) for c in w)[:20],
        }
        for n in (1,2,3,4):
            if len(w)>=n:
                f[f'pre{n}']=l[:n]; f[f'suf{n}']=l[-n:]
        if i>0:
            f.update({'-1.lower':low[i-1], '-1.shape':shape(words[i-1]), '-1.title':words[i-1][:1].isupper()})
        else: f['BOS']=True
        if i>1: f['-2.lower']=low[i-2]
        if i+1<len(tokens):
            f.update({'+1.lower':low[i+1], '+1.shape':shape(words[i+1]), '+1.title':words[i+1][:1].isupper()})
        else: f['EOS']=True
        if i+2<len(tokens): f['+2.lower']=low[i+2]
        if i>0 and i+1<len(tokens): f['context_pair']=low[i-1]+'_'+low[i+1]
        out.append(f)
    return out

def labels_for(tokens, annotations):
    labels=['O']*len(tokens)
    for s,e,_,_ in annotations:
        hits=[i for i,(_,a,b) in enumerate(tokens) if a<e and b>s]
        if not hits: continue
        labels[hits[0]]='B'
        for i in hits[1:]: labels[i]='I'
    return labels

def terms_from_annotations(annotations):
    known=set()
    for bysec in annotations.values():
        for sec, anns in bysec.items():
            # Terms tokenized the same way as documents.
            for s,e,text,cls in anns:
                seq=tuple(x[0].lower() for x in tokenize(text))
                if seq: known.add(seq)
    return known

def build_data(articles, annotations, known):
    X=[]; Y=[]; meta=[]
    for pmid,sections in articles.items():
        for sec in ('T','A'):
            toks=tokenize(sections[sec])
            anns=annotations.get(pmid,{}).get(sec,[])
            X.append(features(toks,sec=='T',known)); Y.append(labels_for(toks,anns))
            meta.append((pmid,sec,sections[sec],toks,anns))
    return X,Y,meta

def spans_from_labels(tokens, labels):
    out=[]; start=None; last=None
    for i,(tok,s,e) in enumerate(tokens):
        label=labels[i]
        if label=='B':
            if start is not None: out.append((start,last))
            start=s; last=e
        elif label=='I' and start is not None:
            last=e
        else:
            if start is not None: out.append((start,last)); start=None; last=None
    if start is not None: out.append((start,last))
    return out

def gold_spans(meta):
    return {(p,s,a,b) for p,sec,text,toks,anns in meta for a,b,_,_ in anns for s in [sec]}

def score(pred, gold):
    tp=len(pred&gold); fp=len(pred-gold); fn=len(gold-pred)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return p,r,2*p*r/(p+r) if p+r else 0,tp,fp,fn

def fit(train_files, validate=False):
    arts={}; anns=defaultdict(lambda: defaultdict(list))
    for f in train_files:
        arts.update(load_articles(f))
        aa=load_annotations(f)
        for p, sections in aa.items():
            for s, values in sections.items(): anns[p][s].extend(values)
    known=terms_from_annotations(anns)
    X,Y,meta=build_data(arts,anns,known)
    model=sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.02, c2=0.1, max_iterations=120,
        all_possible_transitions=True, verbose=False)
    model.fit(X,Y)
    if validate:
        dev=load_articles('development'); da=load_annotations('development')
        Dx,Dy,dm=build_data(dev,da,known)
        yp=model.predict(Dx)
        pred={(p,sec,a,b) for (p,sec,text,toks,anns_), labs in zip(dm,yp) for a,b in spans_from_labels(toks,labs)}
        gold=gold_spans(dm)
        print('DEV',score(pred,gold))
        print('token labels / gold / predicted',sum(map(len,Dy)),len(gold),len(pred))
        for item,labs in zip(dm,yp):
            p,sec,text,toks,ann=item
            pp=spans_from_labels(toks,labs); gg=[(a,b,t) for a,b,t,_ in ann]
            missed=[(a,b,text[a:b]) for a,b,t in gg if (p,sec,a,b) not in pred]
            extra=[(a,b,text[a:b]) for a,b in pp if (p,sec,a,b) not in gold]
            if missed or extra:
                print('ERR',p,sec,'miss',missed[:8],'extra',extra[:8])
    return model,known

if __name__=='__main__':
    model,known=fit(['training'],validate=True)
