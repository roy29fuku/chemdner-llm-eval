import os,sys,pickle,json,random,collections,math
import numpy as np
from decoding import decode,evaluate
from postprocess import valid_word,process,score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

class SpanFeatures:
    def __init__(self,doc,logits):
        self.doc=doc;self.logits=logits;self.maps={}
        for sec in ('T','A'):
            indices=[i for i,(s,a,b) in enumerate(doc['offsets']) if s==sec]
            self.maps[sec]=(indices,{doc['offsets'][i][1]:j for j,i in enumerate(indices)},{doc['offsets'][i][2]:j for j,i in enumerate(indices)})
        self.pred=decode(doc,logits,0)
        self.surfaces=collections.defaultdict(list)
        for span in self.pred:
            sec,a,b=span;word=doc['texts'][sec][a:b]
            self.surfaces[word].append(self.features(span)[2])
    def features(self,span):
        sec,a,b=span;indices,starts,ends=self.maps[sec]
        if a not in starts or b not in ends:return [int(span in self.pred),0,-8,-8,-8,-8,-8,0]
        lo=starts[a];hi=ends[b]
        if hi<lo:return [0,0,-8,-8,-8,-8,-8,0]
        ids=indices[lo:hi+1];tags=[4] if len(ids)==1 else [1]+[2]*(len(ids)-2)+[3]
        s=self.logits[ids];margin=s[np.arange(len(ids)),tags]-s[:,0]
        start=s[0,tags[0]]-max(s[0,k] for k in (0,2,3));end=s[-1,tags[-1]]-max(s[-1,k] for k in (0,1,2))
        return [int(span in self.pred),1,float(np.clip(margin.mean(),-8,8)),float(np.clip(margin.min(),-8,8)),float(np.clip(start,-8,8)),float(np.clip(end,-8,8)),float(np.clip(margin.sum()/5,-8,8)),math.log1p(len(ids))]

def make_candidates(model_docs,model_scores,selection=None,lexical=None):
    n=len(model_docs[0]);items=[];features=[];targets=[];groups=[];goldtotal=0
    for j in (range(n) if selection is None else selection):
        doc=model_docs[0][j];goldtotal+=len(doc['gold']);fs=[SpanFeatures(d[j],s[j]) for d,s in zip(model_docs,model_scores)]
        candidates=set()
        for d,s in zip(model_docs,model_scores):
            for bias in (-2,-1,0,1):candidates.update(decode(d[j],s[j],bias))
            candidates.update(process(d[j],s[j],bias=0,source=1,target=-2))
        # Average compatible tokenizations as additional span proposals.
        compatible=[]
        for k,ds in enumerate(model_docs):
            for subset in compatible:
                if ds[j]['offsets']==model_docs[subset[0]][j]['offsets']:
                    subset.append(k);break
            else:compatible.append([k])
        for subset in compatible:
            if len(subset)<2:continue
            ensemble=np.mean([model_scores[k][j] for k in subset],axis=0)
            for bias in (-2,-1,0,1):candidates.update(decode(model_docs[subset[0]][j],ensemble,bias))
        if lexical is not None:candidates.update(lexical.candidates(doc))
        for span in sorted(candidates):
            sec,a,b=span;word=doc['texts'][sec][a:b]
            if not valid_word(word):continue
            f=[v for sf in fs for v in sf.features(span)];votes=[span in sf.pred for sf in fs]
            surface_margin=[max(sf.surfaces.get(word,[-8])) for sf in fs]
            f += [sum(word in sf.surfaces for sf in fs)/len(fs),max(surface_margin),np.mean(surface_margin),math.log1p(sum(len(sf.surfaces.get(word,[])) for sf in fs))]
            f += list(lexical.features(word)) if lexical is not None else [0,0,0]
            f += [sum(any(s==sec and x<=a and b<=y and (x<a or b<y) for s,x,y in sf.pred) for sf in fs)/len(fs)]
            f += [sum(votes)/len(votes),math.log1p(b-a),word.count(' '),sum(c.isupper() for c in word)/len(word),sum(c.isdigit() for c in word)/len(word),sum(not c.isalnum() and not c.isspace() for c in word)/len(word)]
            items.append((j,span));features.append(f);targets.append(int(span in doc['gold']));groups.append(j)
    return items,np.array(features,np.float32),np.array(targets),np.array(groups),goldtotal

def choose(items,prob,threshold):
    ordered=sorted(zip(items,prob),key=lambda x:-x[1]);pred=collections.defaultdict(set)
    for (j,(sec,a,b)),p in ordered:
        if p<threshold:break
        if any(s==sec and x<b and a<y for s,x,y in pred[j]):continue
        pred[j].add((sec,a,b))
    return pred

def metrics(pred,docs,selection):
    tp=fp=fn=0
    for j in selection:
        p=pred[j];g=docs[j]['gold'];tp+=len(p&g);fp+=len(p-g);fn+=len(g-p)
    return dict(f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn)

def train(model_docs,model_scores,out,lexical=None):
    docs=model_docs[0];inds=list(range(len(docs)));random.Random(1234).shuffle(inds);inds=inds[:500]
    items,X,y,groups,goldtotal=make_candidates(model_docs,model_scores,inds,lexical=lexical)
    print('Candidates',len(y),'positives',sum(y),'gold',goldtotal,'max recall',sum(y)/goldtotal,flush=True)
    best=None
    def create(method,param):
        if method=='linear':return make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=1000))
        return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.05,l2_regularization=param,random_state=1337)
    for method,param in [('linear',c) for c in (.003,.01,.03,.1,.3,1,3)]+[('trees',10),('trees',30)]:
        prob=np.zeros(len(y))
        for tr,va in GroupKFold(5).split(X,y,groups):
            clf=create(method,param)
            clf.fit(X[tr],y[tr]);prob[va]=clf.predict_proba(X[va])[:,1]
        results=[]
        for threshold in np.arange(.15,.86,.025):
            m=metrics(choose(items,prob,threshold),docs,inds);results.append((m['f1'],float(threshold),m))
        r=max(results);print('CV',method,param,r,flush=True)
        if best is None or r[0]>best[0]:best=(r[0],method,param,r[1],r[2])
    _,method,param,threshold,metric=best;clf=create(method,param);clf.fit(X,y)
    os.makedirs(out,exist_ok=True);pickle.dump(clf,open(os.path.join(out,'stacker.pkl'),'wb'));json.dump(dict(method=method,parameter=param,threshold=threshold,metrics=metric),open(os.path.join(out,'stacker.json'),'w'),indent=2)
    print('BEST',best,flush=True);return clf,threshold

def predict(docs,scores,clf,threshold,out,lexical=None):
    items,X,y,g,gold=make_candidates(docs,scores,lexical=lexical)
    pred=choose(items,clf.predict_proba(X)[:,1],threshold)
    with open(out,'w') as f:
        for j,d in enumerate(docs[0]):
            for sec,a,b in sorted(pred[j]):f.write(f"{d['pmid']}\t{sec}:{a}:{b}\n")
    print('Wrote',out,sum(map(len,pred.values())),flush=True)
