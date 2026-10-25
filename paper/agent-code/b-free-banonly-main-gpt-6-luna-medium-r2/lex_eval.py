import collections, json, re
from pathlib import Path

def load(split):
    docs={}
    for line in open(f"/workspace/data/{split}.abstracts.txt",encoding="utf8"):
        p,t,a=line.rstrip("\n").split("\t"); docs[p]={"T":t,"A":a}
    gold=set()
    ap=Path(f"/workspace/data/{split}.annotations.txt")
    if ap.exists():
        for line in open(ap,encoding="utf8"):
            p,s,a,b,txt,cl=line.rstrip("\n").split("\t"); gold.add((p,s,int(a),int(b)))
    return docs,gold

def make_automaton(patterns):
    go=[{}]; out=[[]]
    for pi,pat in enumerate(patterns):
        st=0
        for ch in pat:
            nx=go[st].get(ch)
            if nx is None:
                nx=len(go); go[st][ch]=nx; go.append({}); out.append([])
            st=nx
        out[st].append(pi)
    fail=[0]*len(go); q=collections.deque()
    for ch,nx in go[0].items(): q.append(nx)
    while q:
        v=q.popleft()
        for ch,u in go[v].items():
            q.append(u); f=fail[v]
            while f and ch not in go[f]: f=fail[f]
            fail[u]=go[f].get(ch,0); out[u].extend(out[fail[u]])
    return go,fail,out

def main():
    trdocs, tg=load("training"); dvdocs,gold=load("development")
    stats=collections.defaultdict(lambda:[0,set()])
    for p,s,a,b in tg:
        txt=trdocs[p][s][a:b].lower()
        stats[txt][0]+=1; stats[txt][1].add(p)
    patterns=list(stats); go,fail,out=make_automaton(patterns)
    print("lexicon",len(patterns),"trie states",len(go),flush=True)
    candidates=collections.defaultdict(lambda: collections.defaultdict(set))
    total=0
    for p,secs in dvdocs.items():
        for sec,text in secs.items():
            text=text.lower(); st=0
            for i,ch in enumerate(text):
                while st and ch not in go[st]: st=fail[st]
                st=go[st].get(ch,0)
                for pi in out[st]:
                    phrase=patterns[pi]; a=i-len(phrase)+1; b=i+1
                    if a<0: continue
                    if phrase[0].isalnum() and a and text[a-1].isalnum(): continue
                    if phrase[-1].isalnum() and b<len(text) and text[b].isalnum(): continue
                    candidates[(p,sec)][(a,b)].add(pi)
            total+=len(text)
    # Keep exact-case spellings as a separate source so short uppercase formulas and
    # abbreviations (ATP, Fe, CAS fragments) do not inherit lowercase ambiguities.
    rawstats=collections.defaultdict(lambda:[0,set()])
    for line in open("/workspace/data/training.annotations.txt",encoding="utf8"):
        p,s,a,b,txt,cl=line.rstrip("\n").split("\t")
        rawstats[txt][0]+=1; rawstats[txt][1].add(p)
    rawpatterns=list(rawstats); rgo,rfail,rout=make_automaton(rawpatterns)
    rawcandidates=collections.defaultdict(lambda: collections.defaultdict(set))
    for p,secs in dvdocs.items():
        for sec,text in secs.items():
            st=0
            for i,ch in enumerate(text):
                while st and ch not in rgo[st]: st=rfail[st]
                st=rgo[st].get(ch,0)
                for pi in rout[st]:
                    phrase=rawpatterns[pi]; a=i-len(phrase)+1; b=i+1
                    if a<0: continue
                    if phrase[0].isalnum() and a and text[a-1].isalnum(): continue
                    if phrase[-1].isalnum() and b<len(text) and text[b].isalnum(): continue
                    rawcandidates[(p,sec)][(a,b)].add(pi)
    print("scan chars",total,"candidate article buckets",len(candidates),flush=True)
    bertpath=Path("/workspace/output/biobert/dev_predictions.json")
    bert=set(tuple(x) for x in json.load(open(bertpath))) if bertpath.exists() else set()
    results=[]
    for minf in [1,2,3,4,5,6,8,10,15]:
      for mindf in [1,2,3,4,6]:
        for minlen in [0,4,6,8,10]:
            lex=set()
            for (p,sec),cands in candidates.items():
                for (a,b),pis in cands.items():
                    if any(stats[patterns[i]][0]>=minf and len(stats[patterns[i]][1])>=mindf and len(patterns[i])>=minlen for i in pis):
                        lex.add((p,sec,a,b))
            if minlen>=4:
                for (p,sec),cands in rawcandidates.items():
                    for (a,b),pis in cands.items():
                        if any(len(rawpatterns[i])<4 and (rawpatterns[i] != rawpatterns[i].lower() or any(c.isdigit() for c in rawpatterns[i])) and rawstats[rawpatterns[i]][0]>=minf and len(rawstats[rawpatterns[i]][1])>=mindf for i in pis):
                            lex.add((p,sec,a,b))
            tp=len(lex&gold); f1=2*tp/(len(lex)+len(gold))
            results.append((f1,minf,mindf,minlen,len(lex),tp))
    for f1,minf,mindf,minlen,n,tp in sorted(results,reverse=True)[:30]:
        print(f"f1={f1:.4f} minfreq={minf} mindf={mindf} minlen={minlen} n={n} p={tp/n:.4f} r={tp/len(gold):.4f}")
    if bert:
        print("bert",len(bert),"f1",2*len(bert&gold)/(len(bert)+len(gold)))

if __name__=="__main__": main()
