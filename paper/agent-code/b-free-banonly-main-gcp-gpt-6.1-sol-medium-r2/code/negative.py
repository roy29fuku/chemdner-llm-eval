import ahocorasick,collections
from lexicon import boundary
from postprocess import process

def build_negative(train,terms,min_count=3):
    terms={t.lower() for t in terms if len(t)>=3}
    positive=collections.Counter()
    for d in train:
        for sec,a,b in d['gold']:positive[d['texts'][sec][a:b].lower()]+=1
    terms-=positive.keys()
    matcher=ahocorasick.Automaton()
    for t in terms:matcher.add_word(t,t)
    if not terms:return set()
    matcher.make_automaton();negative=collections.Counter()
    for d in train:
        for sec,text in d['texts'].items():
            gold=[(a,b) for s,a,b in d['gold'] if s==sec]
            for end,t in matcher.iter(text.lower()):
                a=end+1-len(t);b=end+1
                if boundary(text,a,b) and not any(x<b and a<y for x,y in gold):negative[t]+=1
    return {t for t,c in negative.items() if c>=min_count}

def run_predictions(docs,scores,bias=0,source=100):
    predictions=[process(d,s,bias=bias,source=source,target=-1) for d,s in zip(docs,scores)]
    terms={d['texts'][sec][a:b] for d,p in zip(docs,predictions) for sec,a,b in p}
    return predictions,terms

def score(docs,preds,negative):
    tp=fp=fn=0
    for d,p in zip(docs,preds):
        p={z for z in p if d['texts'][z[0]][z[1]:z[2]].lower() not in negative};g=d['gold']
        tp+=len(p&g);fp+=len(p-g);fn+=len(g-p)
    return dict(f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn)
if __name__=='__main__':
 import pickle,sys
 cache=pickle.load(open(sys.argv[1],'rb'));S=pickle.load(open(sys.argv[2],'rb'));D=cache['development']
 pred,terms=run_predictions(D,S,bias=-.4,source=1)
 print('Before',score(D,pred,set()),flush=True)
 for count in [2,3,5,10]:
  neg=build_negative(cache['training'],terms,count)
  print('mincount',count,'After',score(D,pred,neg),'Negatives',sorted(neg),flush=True)
