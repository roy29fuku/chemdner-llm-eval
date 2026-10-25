import bisect, collections, json, math, pickle, re
import numpy as np
from pathlib import Path
from lex_eval import load, make_automaton
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline

WORD_RE=re.compile(r"\w+|[^\w\s]",re.UNICODE)

def canonical(q):
    if ' ' in q:
        head,tail=q.rsplit(' ',1)
    else: head,tail='',q
    low=tail.lower()
    if len(low)>5 and low.endswith('ies'): tail=tail[:-3]+'y'
    elif len(low)>5 and low.endswith(('ches','shes','sses','xes','zes')): tail=tail[:-2]
    elif len(low)>4 and low.endswith('s') and not low.endswith(('ss','us','is')): tail=tail[:-1]
    return (head+' ' if head else '')+tail

def augment_aliases(stats):
    out=dict(stats)
    for q,v in list(stats.items()):
        if len(q)<4 or q.endswith('-'): continue
        if ' ' in q: head,tail=q.rsplit(' ',1)
        else: head,tail='',q
        low=tail.lower()
        variants=[]
        if low.endswith('y') and len(low)>4: variants.append(tail[:-1]+'ies')
        elif low.endswith(('s','x','z','ch','sh')): variants.append(tail+'es')
        else: variants.append(tail+'s')
        if low.endswith('ies') and len(low)>5: variants.append(tail[:-3]+'y')
        elif low.endswith(('ches','shes','sses','xes','zes')): variants.append(tail[:-2])
        elif low.endswith('s') and len(low)>4 and not low.endswith(('ss','us','is')): variants.append(tail[:-1])
        for t in variants:
            alias=(head+' ' if head else '')+t
            if alias not in out: out[alias]=[v[0],set(v[1])]
    return out

def surfaces(train_docs, gold):
    st=collections.defaultdict(lambda:[0,set()])
    for p,s,a,b in gold:
        q=train_docs[p][s][a:b].lower(); st[q][0]+=1; st[q][1].add(p)
    return st

def candidate_docs(docs, pats, go, fail, out):
    results={}
    for p,secs in docs.items():
      for sec,text in secs.items():
        st=0; found=collections.defaultdict(set); low=text.lower()
        for i,ch in enumerate(low):
          while st and ch not in go[st]: st=fail[st]
          st=go[st].get(ch,0)
          for pi in out[st]:
            q=pats[pi]; a=i-len(q)+1; b=i+1
            if a<0 or (q[0].isalnum() and a and low[a-1].isalnum()) or (q[-1].isalnum() and b<len(low) and low[b].isalnum()): continue
            found[(a,b)].add(pi)
        toks=[(m.start(),m.end(),m.group().lower()) for m in WORD_RE.finditer(text)]
        starts=[x[0] for x in toks]; ends=[x[1] for x in toks]
        rows=[]
        for (a,b),ids in found.items():
          q=low[a:b]
          rows.append((p,sec,a,b,q,text[a:b],starts,ends,toks))
        results.update({(p,sec,a,b):(q,orig,starts,ends,toks) for p,sec,a,b,q,orig,starts,ends,toks in rows})
    return results

def featurize(q,orig,a,b,starts,ends,toks,stat):
    freq,df=stat
    f={"TERM="+q:1}
    f["TERMROOT="+canonical(q)]=1
    n=len(q); words=len(q.split())
    f["len="+str(min(30,n//2*2))]=1; f["words="+str(min(words,6))]=1
    f["freq="+str(min(15,int(math.log2(freq+1))))]=1; f["df="+str(min(10,int(math.log2(df+1))))]=1
    f["has_digit="+str(any(c.isdigit() for c in q))]=1
    f["all_upper="+str(orig.isupper())]=1
    f["initial_upper="+str(bool(orig[:1].isupper()))]=1
    f["mixed_case="+str(any(c.isupper() for c in orig[1:]))]=1
    f["has_hyphen="+str('-' in q)]=1; f["has_slash="+str('/' in q)]=1
    f["has_comma="+str(',' in q)]=1; f["paren="+str(('(' in q) or (')' in q))]=1
    f["initial_digit="+str(q[:1].isdigit())]=1
    f["suffix3="+q[-3:]]=1; f["prefix3="+q[:3]]=1
    for n in (2,3,4):
        if len(q)>=n:
            f[f"prefix{n}="+q[:n]]=1; f[f"suffix{n}="+q[-n:]]=1
        for j in range(max(0,len(q)-n+1)):
            f[f"char{n}="+q[j:j+n]]=1
    # Context words immediately outside the candidate boundary.
    li=bisect.bisect_right(ends,a)-1
    ri=bisect.bisect_left(starts,b)
    left=[toks[j][2] for j in range(max(0,li-2),li+1)]
    right=[toks[j][2] for j in range(ri,min(len(toks),ri+3))]
    for pos,word in enumerate(reversed(left),1): f[f"L{pos}="+word]=1
    for pos,word in enumerate(right,1): f[f"R{pos}="+word]=1
    if len(left)>=2: f["L12="+left[-2]+"_"+left[-1]]=1
    if len(right)>=2: f["R12="+right[0]+"_"+right[1]]=1
    if left and right: f["LR="+left[-1]+"_"+right[0]]=1
    return f

def build_features(docs, candidates, stats):
    rows=[]; meta=[]
    for (p,sec,a,b),(q,orig,starts,ends,toks) in candidates.items():
      rows.append(featurize(q,orig,a,b,starts,ends,toks,(stats[q][0],len(stats[q][1]))))
      meta.append((p,sec,a,b))
    return rows,meta

def main():
    trdocs,tgold=load('training'); dvdocs,dgold=load('development')
    stats=surfaces(trdocs,tgold); pats=list(stats); go,fail,out=make_automaton(pats)
    print('vocab',len(pats),flush=True)
    tc=candidate_docs(trdocs,pats,go,fail,out); vc=candidate_docs(dvdocs,pats,go,fail,out)
    print('candidates',len(tc),len(vc),flush=True)
    tx,tm=build_features(trdocs,tc,stats); vx,vm=build_features(dvdocs,vc,stats)
    tgset=set(tgold); vgset=set(dgold)
    y=[int(x in tgset) for x in tm]
    print('positives',sum(y),'negatives',len(y)-sum(y),flush=True)
    vec=DictVectorizer(dtype=np.float32)
    X=vec.fit_transform(tx); V=vec.transform(vx)
    clf=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=35,tol=1e-4,random_state=42,class_weight=None)
    clf.fit(X,y)
    probs=clf.predict_proba(V)[:,1]
    vy=np.asarray([int(x in vgset) for x in vm])
    precision,recall,thr=precision_recall_curve(vy,probs)
    recall_all=recall[:-1]*vy.sum()/len(vgset)
    f1=2*precision[:-1]*recall_all/(precision[:-1]+recall_all+1e-12)
    ix=f1.argmax(); threshold=thr[ix]
    print('candidate gold',vy.sum(),'of',len(vgset),'best threshold',threshold,'f1',f1[ix],'p',precision[ix],'r',recall_all[ix],flush=True)
    pred={x for x,p in zip(vm,probs) if p>=threshold}
    tp=len(pred&vgset)
    print('candidate-model',len(pred),tp,2*tp/(len(pred)+len(vgset)),tp/len(pred),tp/len(vgset),flush=True)
    # Report upper bound for exact-match lexicon candidates and the simple surface baseline.
    simple={x for x in vm if len(dvdocs[x[0]][x[1]][x[2]:x[3]])>=4}
    print('simple baseline',len(simple),len(simple&vgset),2*len(simple&vgset)/(len(simple)+len(vgset)))
    json.dump(sorted(pred),open('/workspace/lexclf_dev.json','w'))
    serial_stats={q:(v[0],len(v[1])) for q,v in stats.items()}
    with open('/workspace/lexclf.pkl','wb') as f: pickle.dump((vec,clf,serial_stats,threshold),f)

if __name__=='__main__': main()
