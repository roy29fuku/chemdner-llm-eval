import collections, json
from lex_eval import load, make_automaton

trdocs,tg=load('training'); docs,gold=load('development')
stats=collections.defaultdict(lambda:[0,set()])
for p,s,a,b in tg:
    q=trdocs[p][s][a:b].lower(); stats[q][0]+=1; stats[q][1].add(p)
patterns=list(stats); go,fail,out=make_automaton(patterns)
cands=collections.defaultdict(lambda:collections.defaultdict(set))
for p,secs in docs.items():
  for sec,text in secs.items():
    text=text.lower(); st=0
    for i,ch in enumerate(text):
      while st and ch not in go[st]: st=fail[st]
      st=go[st].get(ch,0)
      for pi in out[st]:
        q=patterns[pi]; a=i-len(q)+1; b=i+1
        if a<0 or (q[0].isalnum() and a and text[a-1].isalnum()) or (q[-1].isalnum() and b<len(text) and text[b].isalnum()): continue
        cands[(p,sec)][(a,b)].add(pi)
pred={tuple(x) for x in json.load(open('/workspace/devpred_best.json'))}
gold=set(gold)
def f1(x):
  tp=len(x&gold); return (2*tp/(len(x)+len(gold)),tp,len(x),tp/len(x),tp/len(gold))
print('base',f1(pred))
results=[]
for minfreq in [1,2,3,4,5,8]:
  lex=collections.defaultdict(list)
  for k,spans in cands.items():
    for (a,b),pis in spans.items():
      if b-a>=4 and any(stats[patterns[i]][0]>=minfreq for i in pis): lex[k].append((a,b,max(stats[patterns[i]][0] for i in pis)))
  # Keep the model span unless an overlapping train-lexicon mention supplies a plausible full boundary.
  for ratio in [0.25,0.4,0.55,0.7,0.85,1.0]:
    fixed=set()
    for p,s,a,b in pred:
      opts=[]
      for x,y,freq in lex.get((p,s),[]):
        ov=max(0,min(b,y)-max(a,x))
        den=min(b-a,y-x)
        if den and ov/den>=ratio:
          iou=ov/(max(b,y)-min(a,x))
          opts.append((iou,freq,y-x,x,y))
      if opts:
        _,_,_,x,y=max(opts)
        fixed.add((p,s,x,y))
      else: fixed.add((p,s,a,b))
    results.append((f1(fixed)[0],minfreq,ratio,*f1(fixed)[1:]))
  # Add only lex matches which overlap any model output, keeping the model prediction too.
  for ratio in [0.25,0.4,0.55,0.7,0.85,1.0]:
    predplus=set(pred)
    for p,s,a,b in pred:
      for x,y,freq in lex.get((p,s),[]):
        ov=max(0,min(b,y)-max(a,x)); den=min(b-a,y-x)
        if den and ov/den>=ratio: predplus.add((p,s,x,y))
    results.append((f1(predplus)[0],minfreq,ratio,*f1(predplus)[1:]))
for r in sorted(results,reverse=True)[:20]: print(r)
