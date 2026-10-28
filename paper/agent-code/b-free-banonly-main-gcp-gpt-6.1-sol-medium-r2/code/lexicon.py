import ahocorasick,re,collections,pickle
from postprocess import process,span_margin,valid_word

def boundary(text,a,b):
    return (a==0 or not(text[a-1].isalnum() or text[a-1]=='_')) and (b==len(text) or not(text[b].isalnum() or text[b]=='_'))
def build(docs,mincount=2):
    terms=collections.Counter()
    for d in docs:
        for sec,a,b in d['gold']:
            t=d['texts'][sec][a:b]
            if len(t)>=3 and valid_word(t):terms[t]+=1
    terms={t:c for t,c in terms.items() if c>=mincount}
    matcher=ahocorasick.Automaton()
    for t in terms:matcher.add_word(t,t)
    matcher.make_automaton();positive=collections.Counter();negative=collections.Counter()
    for d in docs:
        for sec,text in d['texts'].items():
            gold=[(a,b) for s,a,b in d['gold'] if s==sec]
            for end,t in matcher.iter(text):
                a=end+1-len(t);b=end+1
                if not boundary(text,a,b):continue
                if (sec,a,b) in d['gold']:positive[t]+=1
                elif any(x<=a and b<=y for x,y in gold):continue
                else:negative[t]+=1
    chosen={t for t in terms if positive[t]/max(positive[t]+negative[t],1)>=.95}
    matcher=ahocorasick.Automaton()
    for t in chosen:matcher.add_word(t,t)
    matcher.make_automaton();print('Lexicon',len(chosen),flush=True)
    return matcher

def augment(doc,scores,matcher,target=-1,**kwargs):
    pred=process(doc,scores,**kwargs);new=set()
    for sec,text in doc['texts'].items():
        candidates=[]
        for end,t in matcher.iter(text):
            a=end+1-len(t);b=end+1
            if (sec,a,b) in pred or not boundary(text,a,b):continue
            if any(s==sec and x<b and a<y for s,x,y in pred):continue
            if span_margin(doc,scores,sec,a,b)>=target:candidates.append((a,b))
        for a,b in sorted(candidates,key=lambda z:-(z[1]-z[0])):
            if any(s==sec and x<b and a<y for s,x,y in new):continue
            new.add((sec,a,b))
    return pred|new

def score(docs,scores,matcher,target=-1,**kwargs):
    tp=fp=fn=0
    for d,s in zip(docs,scores):
        pred=augment(d,s,matcher,target=target,**kwargs);gold=d['gold'];tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return dict(f1=2*p*r/max(p+r,1e-10),precision=p,recall=r,tp=tp,fp=fp,fn=fn)
