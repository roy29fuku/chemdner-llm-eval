import argparse,pickle,os,json
from postprocess import process,score
p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
data=pickle.load(open(args.cache,'rb'));dev=pickle.load(open(args.run+'/development.logits.pkl','rb'))
best=None
for bias in (-1,-.5,0,.5,1):
 for source in (100,1,2):
  m=score(data['development'],dev,bias=bias,source=source,target=-1)
  if best is None or m['f1']>best[0]:best=(m['f1'],bias,source,m)
print('Best',best,flush=True)
_,bias,source,m=best;test=pickle.load(open(args.run+'/test.logits.pkl','rb'))
with open(args.out,'w') as f:
 for d,s in zip(data['test'],test):
  for sec,a,b in sorted(process(d,s,bias=bias,source=source,target=-1)):f.write(f"{d['pmid']}\t{sec}:{a}:{b}\n")
json.dump(dict(bias=bias,source=source,metrics=m),open(args.out+'.json','w'),indent=2)
