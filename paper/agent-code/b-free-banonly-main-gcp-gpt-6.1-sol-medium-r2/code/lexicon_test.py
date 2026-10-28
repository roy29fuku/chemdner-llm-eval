import pickle,sys
from lexicon import build,score
cache=pickle.load(open(sys.argv[1],'rb'));scores=pickle.load(open(sys.argv[2],'rb'));docs=cache['development']
matcher=build(cache['training'])
for target in [-2,-1,0,1]:
 print(target,score(docs,scores,matcher,target=target,bias=.4,source=1),flush=True)
