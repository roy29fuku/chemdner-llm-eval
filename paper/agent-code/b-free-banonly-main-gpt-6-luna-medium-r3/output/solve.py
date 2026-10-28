import os, re, sys, collections, math, pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics

ROOT='/workspace/data/'
PAT=re.compile(r'[^\W\d_]+|\d+|[^\w\s]', re.UNICODE)

def read_articles(path):
    out={}
    with open(path,encoding='utf8') as f:
        for line in f:
            pmid,title,abstract=line.rstrip('\n').split('\t',2)
            out[pmid]=(title,abstract)
    return out

def read_anns(path):
    out=collections.defaultdict(lambda: {'T':[],'A':[]})
    with open(path,encoding='utf8') as f:
        for line in f:
            pmid,sec,s,e,txt,cl=line.rstrip('\n').split('\t',5)
            out[pmid][sec].append((int(s),int(e),txt,cl))
    return out

def tokenize(text):
    return [(m.group(),m.start(),m.end()) for m in PAT.finditer(text)]

def shape(w):
    s=[]
    for c in w:
        if c.isupper(): z='A'
        elif c.islower(): z='a'
        elif c.isdigit(): z='0'
        else: z=c
        if not s or s[-1]!=z: s.append(z)
    return ''.join(s)[:20]

def term_lexicon(anns):
    lex=collections.Counter()
    for secs in anns.values():
        for mentions in secs.values():
            for s,e,txt,cl in mentions:
                toks=tuple(x[0].lower() for x in tokenize(txt))
                if toks: lex[toks]+=1
    return lex

_LEX_TRIE_CACHE={}
def add_lex_flags(tokens,lex):
    # Mark tokens covered by any annotated surface-form match.
    words=[w.lower() for w,_,_ in tokens]
    flags=[[] for _ in tokens]
    cachekey=id(lex)
    trie=_LEX_TRIE_CACHE.get(cachekey)
    if trie is None:
        trie={}
        for term,n in lex.items():
            node=trie
            for word in term: node=node.setdefault(word,{})
            node[None]=n
        _LEX_TRIE_CACHE.clear();_LEX_TRIE_CACHE[cachekey]=trie
    for i,w in enumerate(words):
        node=trie
        j=i
        while j<len(words) and words[j] in node:
            node=node[words[j]]; j+=1
            if None in node:
                n=node[None]; length=j-i
                for k in range(i,j):flags[k].append((length,n,k-i,k==i,k==j-1))
    return flags

def token_features(tokens,i,lexflags,section):
    w,s,e=tokens[i]; low=w.lower(); sh=shape(w)
    # keep frequency/context terms from becoming too sparse, while morphology handles new names.
    f={
      'bias':1.0,'section='+section:True,
      'w':low,'shape':sh,'len':min(len(w),20),
      'isalpha':w.isalpha(),'isdigit':w.isdigit(),'isalnum':w.isalnum(),
      'upper':w.isupper(),'title':w.istitle(),'lower':w.islower(),
      'hasdigit':any(c.isdigit() for c in w),'hasalpha':any(c.isalpha() for c in w),
      'hasupper':any(c.isupper() for c in w),'n_upper':min(sum(c.isupper() for c in w),4),
      'n_digit':min(sum(c.isdigit() for c in w),4),
    }
    if len(low)>1: f['p1']=low[:1];f['s1']=low[-1:]
    for n in (2,3,4,5):
        if len(low)>=n: f['p'+str(n)]=low[:n];f['s'+str(n)]=low[-n:]
    if any(c.isdigit() for c in w):
        f['digitshape']=re.sub(r'\d','0',w.lower())[:20]
    # surface forms learned from train annotations.
    for n,cnt,offset,isstart,isend in lexflags[i]:
        f['lexlen='+str(min(n,8))]=True
        f['lexfreq='+str(min(cnt,5))]=True
        if isstart:f['lexstart']=True
        if isend:f['lexend']=True
        f['lexpos='+str(offset)]=True
    # positions and neighboring token attributes
    if i==0:f['BOS']=True
    if i==len(tokens)-1:f['EOS']=True
    for d in (-2,-1,1,2):
        j=i+d
        if 0<=j<len(tokens):
            v,_,_=tokens[j]; vl=v.lower()
            f[f'{d}:w']=vl;f[f'{d}:shape']=shape(v)
            if v.isupper():f[f'{d}:upper']=True
            if any(c.isdigit() for c in v):f[f'{d}:digit']=True
        else:f[f'{d}:BND']=True
    # Local orthographic contexts help identify list boundaries and chemical suffixes.
    return f

def make_doc(text, mentions, lex, section):
    toks=tokenize(text)
    ys=['O']*len(toks)
    # spans don't overlap; mark token labels on exact offsets.
    starts={s:(e,txt) for s,e,txt,cl in mentions}
    for s,e,txt,cl in mentions:
        inside=[i for i,(w,a,b) in enumerate(toks) if a>=s and b<=e]
        # all mention boundaries are aligned after alphabetic/numeric tokenizer refinement.
        if inside:
            ys[inside[0]]='B-CHEM'
            for i in inside[1:]:ys[i]='I-CHEM'
    flags=add_lex_flags(toks,lex)
    X=[token_features(toks,i,flags,section) for i in range(len(toks))]
    return X,ys,toks

def build_data(articles,anns,lex,include_labels=True):
    X=[];Y=[]; meta=[]
    for pmid,(title,abstract) in articles.items():
        for sec,text in [('T',title),('A',abstract)]:
            mentions=anns.get(pmid,{}).get(sec,[]) if include_labels else []
            xx,yy,toks=make_doc(text,mentions,lex,sec)
            X.append(xx);Y.append(yy);meta.append((pmid,sec,text,toks))
    return X,Y,meta

def spans_from_labels(labels,tokens):
    out=[]; active=None
    for i,label in enumerate(labels):
        if label.startswith('B-'):
            if active is not None: out.append((tokens[active][1],tokens[i-1][2]))
            active=i
        elif label=='O':
            if active is not None:out.append((tokens[active][1],tokens[i-1][2]));active=None
        elif label.startswith('I-'):
            if active is None:active=i
    if active is not None:out.append((tokens[active][1],tokens[-1][2]))
    return out

def gold_set(articles,anns):
    z=set()
    for pmid,secs in anns.items():
      for sec,ms in secs.items():
       for s,e,_,_ in ms:z.add((pmid,sec,s,e))
    return z

def train_eval():
    train=read_articles(ROOT+'training.abstracts.txt'); trann=read_anns(ROOT+'training.annotations.txt')
    dev=read_articles(ROOT+'development.abstracts.txt'); dvann=read_anns(ROOT+'development.annotations.txt')
    lex=term_lexicon(trann)
    print('lex',len(lex),flush=True)
    X,Y,_=build_data(train,trann,lex)
    print('train tokens',sum(map(len,X)),flush=True)
    model=sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.08,c2=0.08,max_iterations=120,all_possible_transitions=True,verbose=True)
    model.fit(X,Y)
    Xd,Yd,meta=build_data(dev,dvann,lex)
    pred=model.predict(Xd)
    gold=gold_set(dev,dvann); predset=set()
    for labels,(pmid,sec,text,toks) in zip(pred,meta):
        for s,e in spans_from_labels(labels,toks):predset.add((pmid,sec,s,e))
    tp=len(predset&gold);p=tp/max(1,len(predset));r=tp/max(1,len(gold));f=2*p*r/(p+r) if p+r else 0
    print('DEV',len(gold),len(predset),tp,'P/R/F1',p,r,f,flush=True)
    # Provide component metrics and dumps for calibration/inspection.
    with open('/workspace/output/dev_model.pkl','wb') as f:pickle.dump((model,lex),f)
    with open('/workspace/output/dev_predictions.tsv','w') as f:
      for pmid,sec,s,e in sorted(predset): f.write(f'{pmid}\t{sec}:{s}:{e}\n')

if __name__=='__main__':train_eval()
