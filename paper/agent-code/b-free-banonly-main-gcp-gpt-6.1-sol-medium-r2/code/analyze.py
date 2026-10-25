import pickle,sys,collections
from decoding import decode,evaluate
D=pickle.load(open(sys.argv[1],'rb'));S=pickle.load(open(sys.argv[2],'rb'))
print(evaluate(D,S,.2))
errors=collections.Counter();samples=[]
for d,s in zip(D,S):
 p=decode(d,s,.2);g=d['gold']
 for kind,spans in [('FP',p-g),('FN',g-p)]:
  for sec,a,b in spans:
   word=d['texts'][sec][a:b];errors[kind,word]+=1
   if len(samples)<30:samples.append((kind,word,d['texts'][sec][max(0,a-50):min(len(d['texts'][sec]),b+50)]))
print(errors.most_common(60))
for z in samples: print(z)
