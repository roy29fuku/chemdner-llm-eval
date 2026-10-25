import ahocorasick,json,argparse
from collections import Counter,defaultdict
from pathlib import Path

def read(split):
 docs=[];gold=set()
 for line in open('/workspace/data/'+split+'.abstracts.txt'):
  p,t,a=line.rstrip('\n').split('\t');docs.extend([(p,'T',t),(p,'A',a)])
 if split!='test':
  for line in open('/workspace/data/'+split+'.annotations.txt'):
   p,s,a,b,t,c=line.rstrip('\n').split('\t');gold.add((p,s,int(a),int(b)))
 return docs,gold

def match(auto,text):
 lowered=text.lower();mapping=None
 if len(lowered)!=len(text):
  mapping=[]
  for i,c in enumerate(text):mapping.extend([i]*len(c.lower()))
 for end,term in auto.iter(lowered):
  start=end-len(term)+1
  if mapping is not None:start,end=mapping[start],mapping[end]+1
  else:end+=1
  if start and text[start-1].isalnum():continue
  if end<len(text) and text[end].isalnum():continue
  yield start,end,term

def build(splits):
 pos=Counter(); auto=ahocorasick.Automaton(); docs=[]; gold=set()
 for split in splits:
  d,g=read(split);docs+=d;gold|=g
  for line in open('/workspace/data/'+split+'.annotations.txt'):
   p,s,a,b,t,c=line.rstrip('\n').split('\t');pos[t.lower()]+=1
 for term in pos:auto.add_word(term,term)
 auto.make_automaton();tot=Counter();hit=Counter()
 for p,s,text in docs:
  for a,b,t in match(auto,text):
   tot[t]+=1
   if (p,s,a,b) in gold:hit[t]+=1
 stats={t:(hit[t],tot[t],pos[t]) for t in pos}
 return auto,stats

def predict(docs,auto,stats,threshold=.9,minfreq=1):
 out=set()
 for p,s,text in docs:
  spans=[]
  for a,b,t in match(auto,text):
   h,n,f=stats[t]
   if h>=minfreq and h/max(1,n)>=threshold:spans.append((a,b))
  spans.sort(key=lambda x:(x[0],-x[1]));last=-1
  for a,b in spans:
   if a>=last:out.add((p,s,a,b));last=b
 return out

def metric(pred,gold):
 hit=len(pred&gold);p=hit/max(1,len(pred));r=hit/max(1,len(gold));return (2*p*r/max(1e-10,p+r),p,r,len(pred),hit)

def main():
 auto,stats=build(['training']); dev,gold=read('development'); best=(0,None)
 for th in [.7,.8,.85,.9,.95,1]:
  pred=predict(dev,auto,stats,th);score=metric(pred,gold);print(th,score)
  if score[0]>best[0]:best=(score[0],th)
 auto,stats=build(['training','development']);Path('/workspace/output/lexicon_stats.json').write_text(json.dumps(stats));test,_=read('test');pred=predict(test,auto,stats,best[1])
 with open('/workspace/output/predictions.tsv','w') as f:
  for p,s,a,b in sorted(pred):f.write(f'{p}\t{s}:{a}:{b}\n')
 print('saved baseline',len(pred))

if __name__=='__main__':main()
