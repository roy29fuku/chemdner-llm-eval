import os,json,pickle,argparse,random
import numpy as np
from postprocess import score,process
from stacking import train,predict
from transformers import AutoTokenizer
from ner import read_data
from lexical_features import LexicalFeatures

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',default='data');p.add_argument('--out',required=True);p.add_argument('--mode',default='train',choices=['train','predict']);p.add_argument('--stacker');args=p.parse_args()
 conf=json.load(open(args.config));all_docs=[];all_dev=[];all_test=[]
 for m in conf:
  data=read_data(args.data,AutoTokenizer.from_pretrained(m['model']))
  all_docs.append(data)
  if args.mode=='train':all_dev.append(pickle.load(open(m['run']+'/development.logits.pkl','rb')))
  all_test.append(pickle.load(open(m['run']+'/test.logits.pkl','rb')))
 os.makedirs(args.out,exist_ok=True)
 if args.mode=='train':
  dev_order=all_docs[0]['development'].copy();random.Random(1234).shuffle(dev_order);lexical=LexicalFeatures(all_docs[0]['training']+dev_order[500:])
 else:lexical=LexicalFeatures(all_docs[0]['training']+all_docs[0]['development'])
 if args.mode=='train':
  clf,threshold=train([d['development'] for d in all_docs],all_dev,args.out,lexical=lexical)
  # Compare the learned combination against direct logit averaging for compatible models.
  compatible=[i for i,d in enumerate(all_docs) if all(a['offsets']==b['offsets'] for a,b in zip(d['development'],all_docs[0]['development']))]
  dev=all_docs[0]['development'];inds=list(range(len(dev)));random.Random(1234).shuffle(inds);inds=inds[:500];docs=[dev[j] for j in inds]
  best=None
  for subset in [compatible]+[[i] for i in compatible]:
   logits=[np.mean([all_dev[k][j] for k in subset],axis=0) for j in inds]
   for bias in np.arange(-1.0,1.01,.2):
    for source in (100,1):
     result=score(docs,logits,bias=bias,source=source,target=-1)
     if best is None or result['f1']>best[0]:best=(result['f1'],subset,bias,source,result)
  print('BEST_LOGIT',best,flush=True)
  json.dump(best,open(args.out+'/logit-comparison.json','w'),indent=2)
  metric=json.load(open(args.out+'/stacker.json'))['metrics']['f1']
  if best[0]>metric:
   _,subset,bias,source,_=best
   with open(args.out+'/predictions.tsv','w') as f:
    for j,d in enumerate(all_docs[0]['test']):
     logits=np.mean([all_test[k][j] for k in subset],axis=0)
     for sec,a,b in sorted(process(d,logits,bias=bias,source=source,target=-1)):f.write(f"{d['pmid']}\t{sec}:{a}:{b}\n")
   json.dump(dict(method='logit',subset=subset,bias=bias,source=source),open(args.out+'/chosen.json','w'))
   return
 else:
  saved=json.load(open(args.stacker+'/chosen.json'))
  if saved['method']=='logit':
   subset=saved['subset'];bias=saved['bias'];source=saved['source']
   with open(args.out+'/predictions.tsv','w') as f:
    for j,d in enumerate(all_docs[0]['test']):
     logits=np.mean([all_test[k][j] for k in subset],axis=0)
     for sec,a,b in sorted(process(d,logits,bias=bias,source=source,target=-1)):f.write(f"{d['pmid']}\t{sec}:{a}:{b}\n")
   return
  clf=pickle.load(open(args.stacker+'/stacker.pkl','rb'));threshold=json.load(open(args.stacker+'/stacker.json'))['threshold']
 predict([d['test'] for d in all_docs],all_test,clf,threshold,args.out+'/predictions.tsv',lexical=lexical)
 if args.mode=='train':json.dump(dict(method='stacking'),open(args.out+'/chosen.json','w'))
if __name__=='__main__':main()
