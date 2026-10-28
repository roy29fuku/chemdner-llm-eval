import sys
sys.path.insert(0,'/workspace/output')
import solve
pbert=sys.argv[1]
pcrf=sys.argv[2]
dev=solve.read_articles(solve.ROOT+'development.abstracts.txt'); anns=solve.read_anns(solve.ROOT+'development.annotations.txt'); gold=solve.gold_set(dev,anns)
def read(path):
 out=set()
 with open(path) as f:
  for ln in f:
   p,spec=ln.rstrip().split('\t');sec,s,e=spec.split(':');out.add((p,sec,int(s),int(e)))
 return out
B=read(pbert);C=read(pcrf)
def score(name,S):
 tp=len(S&gold);prec=tp/len(S);rec=tp/len(gold);f=2*prec*rec/(prec+rec)
 print(name,'n=',len(S),'tp=',tp,'P=',round(prec,4),'R=',round(rec,4),'F1=',round(f,4))
score('bert',B);score('crf',C);score('union',B|C);score('intersect',B&C)
print('CRF-only candidates',len(C-B),'correct',len((C-B)&gold),'with BERT',len(B&C))
