import re, sys, math, collections
from pathlib import Path
import sklearn_crfsuite

DATA=Path('/workspace/data')
TOKEN_RE=re.compile(r'\w+|[^\w\s]', re.UNICODE)

def read_abstracts(path):
    out={}
    for line in open(path,encoding='utf8'):
        if line.strip():
            pmid,t,a=line.rstrip('\n').split('\t',2); out[pmid]={'T':t,'A':a}
    return out

def read_annotations(path):
    out=collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path,encoding='utf8'):
        if line.strip():
            pmid,s,b,e,txt,cl=line.rstrip('\n').split('\t')
            out[pmid][s].append((int(b),int(e),txt,cl))
    return out

def tokenized(text):
    return [(m.group(),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]

def shape(w):
    s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
    return re.sub(r'(.)\1+',r'\1',s)

CHEM_END=('ine','ol','one','ium','ide','ate','ose','ose','ane','ene','yne','yl','oxy','azole','azole','amide','acid','ester','itol','amine','phenyl','benzene','pyridine','purine','pyrimidine','quinone','terpene','sterol','lipid','saccharide','alkaloid','flavonoid','polyphenol','hydrocarbon','nucleotide','peptide')

def build_lexicon(annotations):
    lex=collections.Counter()
    for pm in annotations.values():
        for spans in pm.values():
            for b,e,txt,cl in spans: lex[tuple(x[0].casefold() for x in tokenized(txt))]+=1
    return lex

def features_for(tokens, lex):
    words=[x[0] for x in tokens]; n=len(words)
    matches=[[] for _ in words]
    # Exact annotated surface occurrences; used as a feature only.
    for phrase,freq in lex.items():
        L=len(phrase)
        if L==0 or L>n: continue
        for i in range(n-L+1):
            if tuple(w.casefold() for w in words[i:i+L])==phrase:
                for j in range(i,i+L): matches[j].append((i,j,L,freq))
    feats=[]
    for i,(w,b,e) in enumerate(tokens):
        lw=w.casefold(); alpha=''.join(c for c in w if c.isalpha()); num=any(c.isdigit() for c in w)
        f={'bias':1.0,'w':lw,'w[-3:]':lw[-3:],'w[-4:]':lw[-4:],'w[:3]':lw[:3], 'w[:4]':lw[:4],
           'shape':shape(w),'len':min(len(w),20),'isupper':w.isupper(),'istitle':w.istitle(),'islower':w.islower(),
           'hasdigit':num,'alpha':alpha.casefold()==lw and bool(alpha),'hasdash':'-' in w,'hasplus':'+' in w,
           'nonascii':any(ord(c)>127 for c in w),'suffix_chem':any(lw.endswith(x) for x in CHEM_END),
           'formulaish':bool(re.fullmatch(r'(?=.*[A-Z])[A-Za-z0-9()+−−.\[\]±]+',w) and (num or (w.isupper() and len(w)<=6))),
           'lexstart':any(m[0]==i for m in matches[i]),'lexend':any(m[1]==i for m in matches[i]),
           'lexinside':bool(matches[i]),'lexmax':max((m[2] for m in matches[i]),default=0),
           'lexfreq':max((m[3] for m in matches[i]),default=0)}
        if i>0:
            p=words[i-1]; f.update({'-1:w':p.casefold(),'-1:shape':shape(p),'-1:w[-3:]':p.casefold()[-3:]})
        else: f['BOS']=True
        if i+1<n:
            q=words[i+1]; f.update({'+1:w':q.casefold(),'+1:shape':shape(q),'+1:w[-3:]':q.casefold()[-3:]})
        else: f['EOS']=True
        if i>1: f['-2:w']=words[i-2].casefold()
        if i+2<n: f['+2:w']=words[i+2].casefold()
        feats.append(f)
    return feats

def labels_for(text,tokens,spans):
    labels=['O']*len(tokens); covered=0
    for b,e,*_ in spans:
        ids=[i for i,(_,tb,te) in enumerate(tokens) if tb>=b and te<=e]
        if not ids or tokens[ids[0]][1]!=b or tokens[ids[-1]][2]!=e:
            continue
        covered+=1
        for k,i in enumerate(ids): labels[i]=('B' if k==0 else 'I')
    return labels,covered

def examples(abstracts,annotations,lex):
    X=[];Y=[]; loc=[]; skipped=0
    for pm,sections in abstracts.items():
        for sec,text in sections.items():
            toks=tokenized(text); spans=annotations.get(pm,{}).get(sec,[])
            ys,c=labels_for(text,toks,spans); skipped+=len(spans)-c
            X.append(features_for(toks,lex));Y.append(ys);loc.append((pm,sec,text,toks))
    return X,Y,loc,skipped

def spans_from_labels(text,tokens,labels):
    spans=[]; start=None; end=None
    for i,(tag,(w,b,e)) in enumerate(zip(labels,tokens)):
        if tag=='B' or (tag=='I' and start is None):
            if start is not None: spans.append((start,end))
            start,end=b,e
        elif tag=='I': end=e
        else:
            if start is not None: spans.append((start,end));start=end=None
    if start is not None: spans.append((start,end))
    return spans

def score(gold,pred):
    tp=len(gold&pred); p=tp/len(pred) if pred else 0; r=tp/len(gold) if gold else 0
    return p,r,2*p*r/(p+r) if p+r else 0,tp,len(pred),len(gold)

def main():
    tr=read_abstracts(DATA/'training.abstracts.txt'); ta=read_annotations(DATA/'training.annotations.txt')
    dv=read_abstracts(DATA/'development.abstracts.txt'); da=read_annotations(DATA/'development.annotations.txt')
    lex=build_lexicon(ta)
    X,Y,_,skip=examples(tr,ta,lex); print('train entities',sum(y.count('B') for y in Y),'skipped',skip,'lex',len(lex))
    crf=sklearn_crfsuite.CRF(algorithm='lbfgs',c1=.08,c2=.08,max_iterations=200,all_possible_transitions=True,verbose=False)
    crf.fit(X,Y)
    Xd,Yd,loc,skip=examples(dv,da,lex); yp=crf.predict(Xd)
    gold=set();pred=set(); examples_bad=[]
    for (pm,sec,text,toks), labels, pl in zip(loc,Yd,yp):
        for b,e in spans_from_labels(text,toks,labels):gold.add((pm,sec,b,e))
        for b,e in spans_from_labels(text,toks,pl):pred.add((pm,sec,b,e))
    print('dev skipped',skip,'scores p/r/f1,tp/pred/gold',score(gold,pred))
    # Output missed and false for manual diagnosis
    print('false positives:')
    n=0
    for pm,sec,b,e in sorted(pred-gold):
        t=dv[pm][sec]; print(pm,sec,repr(t[b:e]));n+=1
        if n>=60:break
    print('false negatives:')
    n=0
    for pm,sec,b,e in sorted(gold-pred):
        t=dv[pm][sec]; print(pm,sec,repr(t[b:e]));n+=1
        if n>=100:break
if __name__=='__main__': main()
