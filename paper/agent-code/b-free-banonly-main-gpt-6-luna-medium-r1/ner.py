import collections
import re
import sys
from pathlib import Path
import pycrfsuite

ROOT = Path('/workspace')
DATA = ROOT / 'data'
OUT = ROOT / 'output'
TOKEN_RE = re.compile(r"[A-Za-z]+|[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)
CHEM_END = ('ol','one','ide','ate','ine','ium','ane','ene','yne','ose','azole','azine','amide','yl','oxy','thio','phenyl','benz','pyrid','quin')
CHEM_START = ('amino','hydroxy','methyl','ethyl','chloro','bromo','fluoro','nitro','oxo','acetyl','benz','phenyl','pyrid','sulfo','carboxy')

def read_articles(path):
    result={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            pmid,title,abstract=line.rstrip('\n').split('\t')
            result[pmid]={'T':title,'A':abstract}
    return result

def read_annotations(path):
    result=collections.defaultdict(list)
    with open(path,encoding='utf-8') as f:
        for line in f:
            pmid,sec,s,e,txt,cls=line.rstrip('\n').split('\t')
            result[(pmid,sec)].append((int(s),int(e),txt,cls))
    return result

def tokenize(text):
    return [(m.group(),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]

def shape(w):
    last=''; out=[]
    for c in w:
        k='A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c
        if k!=last: out.append(k); last=k
    return ''.join(out)[:8]

def labels_for(toks,spans):
    labels=['O']*len(toks)
    starts={s:i for i,(_,s,_) in enumerate(toks)}
    ends={e:i for i,(_,_,e) in enumerate(toks)}
    for s,e,*_ in spans:
        a=starts.get(s); b=ends.get(e)
        if a is None or b is None or a>b:
            ids=[i for i,(_,x,y) in enumerate(toks) if x<e and y>s]
            if not ids: continue
            a,b=ids[0],ids[-1]
        labels[a]='B'
        for i in range(a+1,b+1): labels[i]='I'
    return labels

def lexicon_from(articles,annotations):
    lex=set()
    for (pmid,sec),spans in annotations.items():
        text=articles[pmid][sec]
        toks=tokenize(text)
        starts={s:i for i,(_,s,_) in enumerate(toks)}
        ends={e:i for i,(_,_,e) in enumerate(toks)}
        for s,e,*_ in spans:
            a=starts.get(s); b=ends.get(e)
            if a is not None and b is not None and a<=b:
                lex.add(tuple(toks[i][0].casefold() for i in range(a,b+1)))
    return lex

def features(toks,lex):
    words=[x[0] for x in toks]
    lows=[w.casefold() for w in words]
    # Annotated mention strings are weak evidence. Use only exact lexical matches.
    gaz=[set() for _ in toks]
    for i in range(len(toks)):
        for n in range(1,min(10,len(toks)-i)+1):
            if tuple(lows[i:i+n]) in lex:
                gaz[i].add('gazB='+str(n)); gaz[i+n-1].add('gazE='+str(n))
                for j in range(i+1,i+n-1): gaz[j].add('gazI='+str(n))
    out=[]
    for i,w in enumerate(words):
        lo=lows[i]; sh=shape(w)
        f=['bias','w='+lo,'shape='+sh,'len='+str(min(12,len(w))),
           'p1='+lo[:1],'p2='+lo[:2],'p3='+lo[:3],'p4='+lo[:4],'p5='+lo[:5],'p6='+lo[:6],
           's1='+lo[-1:],'s2='+lo[-2:],'s3='+lo[-3:],'s4='+lo[-4:],'s5='+lo[-5:],'s6='+lo[-6:],
           'upper='+str(w.isupper()),'title='+str(w.istitle()),'digit='+str(w.isdigit()),
           'hasdigit='+str(any(c.isdigit() for c in w)),
           'suf='+next((x for x in CHEM_END if lo.endswith(x)),'none'),
           'pre='+next((x for x in CHEM_START if lo.startswith(x)),'none')]
        if len(lo)>=5:
            f.extend('c3='+lo[j:j+3] for j in range(len(lo)-2))
        if len(lo)>=7:
            f.extend('c4='+lo[j:j+4] for j in range(len(lo)-3))
        f.extend(gaz[i])
        for d in (-2,-1,1,2):
            j=i+d
            if 0<=j<len(words):
                f.append(str(d)+'w='+lows[j]); f.append(str(d)+'sh='+shape(words[j]))
            else: f.append(str(d)+'=BOUNDARY')
        if i>0: f.append('prevcur='+lows[i-1]+'_'+lo)
        if i+1<len(words): f.append('curnext='+lo+'_'+lows[i+1])
        out.append([sys.intern(x) for x in f])
    return out

def append_corpus(trainer,articles,annotations,lex):
    n=nt=0; misses=0
    for pmid,sections in articles.items():
        for sec,text in sections.items():
            toks=tokenize(text); spans=annotations.get((pmid,sec),[])
            y=labels_for(toks,spans)
            n+=len(spans); nt+=len(toks)
            for s,e,*_ in spans:
                if not any(a==s for _,a,_ in toks) or not any(b==e for _,_,b in toks): misses+=1
            trainer.append(features(toks,lex),y)
    return n,nt,misses

def train(articles,annotations,lex,path,iters=120,c1=0.08,c2=0.04):
    trainer=pycrfsuite.Trainer(verbose=False)
    n,nt,miss=append_corpus(trainer,articles,annotations,lex)
    print('prepared',len(articles)*2,'sequences',nt,'tokens',n,'mentions','boundary fallbacks',miss,flush=True)
    trainer.set_params({'c1':c1,'c2':c2,'max_iterations':iters,'feature.possible_transitions':True})
    trainer.train(str(path))

def get_spans(toks,tags):
    out=[]; start=end=None
    for (w,s,e),tag in zip(toks,tags):
        if tag=='B':
            if start is not None: out.append((start,end))
            start,end=s,e
        elif tag=='I' and start is not None: end=e
        else:
            if start is not None: out.append((start,end)); start=end=None
    if start is not None: out.append((start,end))
    return out

def evaluate(articles,annotations,lex,model):
    tagger=pycrfsuite.Tagger();tagger.open(str(model))
    gold=set(); pred=set()
    for pmid,sections in articles.items():
        for sec,text in sections.items():
            toks=tokenize(text)
            gold.update((pmid,sec,s,e) for s,e,*_ in annotations.get((pmid,sec),[]))
            pred.update((pmid,sec,s,e) for s,e in get_spans(toks,tagger.tag(features(toks,lex))))
    tp=len(gold&pred); p=tp/max(1,len(pred)); r=tp/max(1,len(gold))
    print('dev exact: F1 %.4f P %.4f R %.4f (%d/%d/%d)'%(2*p*r/max(1e-12,p+r),p,r,tp,len(pred),len(gold)),flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    tr=read_articles(DATA/'training.abstracts.txt'); ta=read_annotations(DATA/'training.annotations.txt')
    dv=read_articles(DATA/'development.abstracts.txt'); da=read_annotations(DATA/'development.annotations.txt')
    print('loaded',len(tr),len(dv),sum(map(len,ta.values())),sum(map(len,da.values())),flush=True)
    lex=lexicon_from(tr,ta)
    print('train lexicon',len(lex),flush=True)
    model=OUT/'model.crfsuite'
    train(tr,ta,lex,model)
    evaluate(dv,da,lex,model)
    all_articles=dict(tr); all_articles.update(dv)
    all_ann=collections.defaultdict(list)
    for d in (ta,da):
        for k,v in d.items(): all_ann[k].extend(v)
    lex=lexicon_from(all_articles,all_ann)
    print('combined lexicon',len(lex),flush=True)
    train(all_articles,all_ann,lex,model,140)
    test=read_articles(DATA/'test.abstracts.txt')
    tagger=pycrfsuite.Tagger();tagger.open(str(model))
    with open(OUT/'predictions.tsv','w',encoding='utf-8') as f:
        for pmid,sections in test.items():
            for sec,text in sections.items():
                toks=tokenize(text)
                for s,e in get_spans(toks,tagger.tag(features(toks,lex))):
                    f.write(f'{pmid}\t{sec}:{s}:{e}\n')
    print('wrote predictions',flush=True)

if __name__=='__main__': main()
