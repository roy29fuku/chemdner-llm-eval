import re, sys, os, math, collections
import pycrfsuite

DATA='/workspace/data'
TOK=re.compile(r'[^\W\d_]+|\d+|_+|[^\w\s]', re.UNICODE)

def read_articles(split):
    out={}
    with open(f'{DATA}/{split}.abstracts.txt',encoding='utf8') as f:
        for line in f:
            pmid,title,abstract=line.rstrip('\n').split('\t',2)
            out[pmid]={'T':title,'A':abstract}
    return out

def read_annotations(split):
    out=collections.defaultdict(list)
    with open(f'{DATA}/{split}.annotations.txt',encoding='utf8') as f:
        for line in f:
            p,s,a,b,txt,cl=line.rstrip('\n').split('\t')
            out[p,s].append((int(a),int(b),cl))
    return out

def token_data(text):
    ms=list(TOK.finditer(text))
    words=[m.group() for m in ms]
    spans=[m.span() for m in ms]
    return words,spans

def get_labels(spans, mentions):
    labels=['O']*len(spans)
    for a,b,_ in mentions:
        inds=[i for i,(s,e) in enumerate(spans) if s<b and e>a]
        if not inds: continue
        labels[inds[0]]='B'
        for i in inds[1:]: labels[i]='I'
    return labels

def word_shape(w):
    s=''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c for c in w)
    return re.sub(r'(.)\1+',r'\1',s)[:12]

def token_features(words,i):
    w=words[i]; lo=w.lower(); n=len(w)
    f={'bias':1.0,'w':lo,'raw':w,'shape':word_shape(w), 'len':min(n,20),
       'isupper':w.isupper(),'islower':w.islower(),'istitle':w.istitle(),
       'isdigit':w.isdigit(),'hasdigit':any(c.isdigit() for c in w),
       'hasalpha':any(c.isalpha() for c in w),
       'hasGreek':any('GREEK' in __import__('unicodedata').name(c,'') for c in w),
       'hyphen': '-' in w,'plus': '+' in w}
    for z in range(1,min(6,n+1)):
        f[f'pre{z}']=lo[:z];f[f'suf{z}']=lo[-z:]
    if i==0: f['BOS']=True
    else:
        p=words[i-1];pl=p.lower()
        f.update({'-1:w':pl,'-1:shape':word_shape(p),'bi-1':pl+'_'+lo,
                  '-1:punct':len(p)==1 and not p.isalnum()})
        # Two-character suffix/prefix morphology of adjacent words.
        f['-1:suf3']=pl[-3:]
    if i==len(words)-1:f['EOS']=True
    else:
        nx=words[i+1];nl=nx.lower()
        f.update({'+1:w':nl,'+1:shape':word_shape(nx),'bi+1':lo+'_'+nl,
                  '+1:punct':len(nx)==1 and not nx.isalnum(),'+1:pre3':nl[:3]})
    if i>1:f['-2:w']=words[i-2].lower()
    if i+2<len(words):f['+2:w']=words[i+2].lower()
    return f

def sequences(split, with_labels=True):
    articles=read_articles(split)
    anns=read_annotations(split) if with_labels else {}
    seqs=[]; refs=[]
    for pmid,sections in articles.items():
      for sec,text in sections.items():
        words,spans=token_data(text)
        x=[token_features(words,i) for i in range(len(words))]
        seqs.append(x)
        refs.append((pmid,sec,text,spans,get_labels(spans,anns.get((pmid,sec),[])) if with_labels else None))
    return seqs,refs

def spans_from_tags(tags,spans):
    out=[]; start=None; end=None
    for i,t in enumerate(tags):
        if t in ('B','I'):
            if start is None or t=='B':
                if start is not None:out.append((start,end))
                start=spans[i][0];end=spans[i][1]
            else:end=spans[i][1]
        elif start is not None:
            out.append((start,end));start=None;end=None
    if start is not None:out.append((start,end))
    return out

def eval_model(model, split):
    X,refs=sequences(split)
    gold=set();pred=set()
    for x,(p,s,text,spans,ys) in zip(X,refs):
        for a,b in spans_from_tags(ys,spans):gold.add((p,s,a,b))
        for a,b in spans_from_tags(model.tag(x),spans):pred.add((p,s,a,b))
    tp=len(gold&pred);pr=tp/max(1,len(pred));rc=tp/max(1,len(gold));f=2*pr*rc/max(1e-9,pr+rc)
    print(split,'P/R/F1',round(pr,4),round(rc,4),round(f,4),'n',len(pred),len(gold),'tp',tp,flush=True)
    print('FN sample',list(gold-pred)[:5], 'FP sample',list(pred-gold)[:5],flush=True)
    return f

def main():
    x,refs=sequences('training')
    y=[r[-1] for r in refs]
    trainer=pycrfsuite.Trainer(verbose=True)
    for xx,yy in zip(x,y):trainer.append(xx,yy)
    trainer.set_params({'c1':0.05,'c2':0.05,'max_iterations':100,'feature.possible_transitions':True,'feature.possible_states':True})
    trainer.train('/workspace/model.crfsuite')
    tagger=pycrfsuite.Tagger();tagger.open('/workspace/model.crfsuite')
    eval_model(tagger,'development')

if __name__=='__main__':main()
