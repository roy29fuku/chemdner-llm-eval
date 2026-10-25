"""Project BILOU emissions between tokenizations and combine trained models."""
from ner import *

def logsoftmax(x):
    z=x-x.max(axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))

def project_scores(source_docs,source_scores,target_docs):
    projected=[]
    for sd,ss,td in zip(source_docs,source_scores,target_docs):
        assert sd['pmid']==td['pmid']
        if sd['offsets']==td['offsets'] and sd['sections']==td['sections']:
            projected.append(logsoftmax(ss));continue
        probs=np.exp(logsoftmax(ss))
        result=np.full((len(td['ids']),5),-30.,np.float32)
        pointers={'T':0,'A':0}
        sources={sec:[i for i,s in enumerate(sd['sections']) if s==sec] for sec in ['T','A']}
        for i,(sec,(a,b)) in enumerate(zip(td['sections'],td['offsets'])):
            if sec=='S':result[i,0]=0;continue
            indices=sources[sec];j=pointers[sec]
            while j<len(indices) and sd['offsets'][indices[j]][1]<=a:j+=1
            pointers[sec]=j
            pieces=[];k=j
            while k<len(indices):
                si=indices[k];aa,bb=sd['offsets'][si]
                if aa>=b:break
                p=probs[si];left=a<=aa;right=b>=bb
                if left and right:q=p
                elif left:q=np.array([p[0],p[1]+p[4],p[2]+p[3],0.,0.])
                elif right:q=np.array([p[0],0.,p[1]+p[2],p[3]+p[4],0.])
                else:q=np.array([p[0],0.,p[1:].sum(),0.,0.])
                pieces.append(np.log(np.clip(q,1e-12,1)))
                k+=1
            if not pieces:result[i,0]=0;continue
            if len(pieces)==1:result[i]=pieces[0]
            else:
                p=np.stack(pieces)
                inside=p[:,2].sum()
                result[i]=[p[:,0].sum(),p[0,1]+p[1:,2].sum(),inside,p[:-1,2].sum()+p[-1,3],
                           p[0,1]+p[1:-1,2].sum()+p[-1,3]]
        projected.append(logsoftmax(result))
    return projected

def combine(all_scores,weights=None,arithmetic=False):
    if weights is None:weights=[1/len(all_scores)]*len(all_scores)
    if arithmetic:
        return [np.log(np.maximum(sum(w*np.exp(s[i]) for w,s in zip(weights,all_scores)),1e-12)) for i in range(len(all_scores[0]))]
    return [sum(w*s[i] for w,s in zip(weights,all_scores)) for i in range(len(all_scores[0]))]

if __name__=='__main__':
    import sys
    records=[pickle.load(open(p,'rb')) for p in sys.argv[1:]]
    target=records[0]['docs']
    scores=[project_scores(r['docs'],r['scores'],target) for r in records]
    gold=gold_spans(target)
    for arithmetic in [False,True]:
        merged=combine(scores,arithmetic=arithmetic)
        for bias in [-1,-.5,0,.5,1]:
            print(arithmetic,bias,metric(spans_from_scores(target,merged,bias),gold),flush=True)
