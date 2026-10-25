import argparse,json,itertools
from pathlib import Path
from collections import Counter
from ensemble import load_records,merge
from evaluate import load_corpus,decode,write
from lexicon import metric,build
from mutations import remove_mutations
from boundaries import build_rules,correct
from calibrate import oof_fit,adjust,category
import numpy as np
from postprocess import exclusions,abbreviation,propagation,expansion,dictionary

def apply(pred,docs,auto,stats,config):
 if config.get('expand'):pred=expansion(pred,docs,auto,stats,config.get('expand_threshold',.95),config.get('expand_count',2))
 if config.get('abbreviation'):pred=abbreviation(pred,docs)
 if config.get('propagate'):pred=propagation(pred,docs,mincount=config['propagate'])
 if config.get('dictionary'):pred=dictionary(pred,docs,auto,stats,.95,10)
 if config.get('exclude'):pred=exclusions(pred,docs,stats.keys())
 return pred

def main():
 ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--split',default='development');ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);ap.add_argument('--tune',action='store_true');ap.add_argument('--calibrate',action='store_true');ap.add_argument('--weights',nargs='+',type=float);args=ap.parse_args()
 docs,gold,defaultoffs=load_corpus(args.split);records=[load_records(f,defaultoffs) for f in args.files]
 if args.tune:
  weights=args.weights or [1]*len(records);scores,offs=merge(records,docs,weights);best=(0,None,None);auto,stats=build(['training'])
  variants=[(0,scores)]
  if args.calibrate:
   cats,deltas,allweights=oof_fit(scores,docs,offs,gold);variants.append((1,adjust(scores,cats,deltas,1)))
  for strength,current in variants:
   for bias in [-.3,-.15,0,.15,.3]:
    pred=decode(current,docs,offs,bias);print('base',strength,bias,metric(pred,gold),flush=True)
    for expand,abbr,prop,exclude in itertools.product([False,True],[False,True],[0,1,2],[False,True]):
     config={'bias':bias,'weights':weights,'expand':expand,'abbreviation':abbr,'propagate':prop,'exclude':exclude,'dictionary':False,'method':'probability','case_strength':strength}
     if strength:config['case_weights']=allweights.tolist()
     pp=apply(pred,docs,auto,stats,config);m=metric(pp,gold)
     if m[0]>best[0]:best=(m[0],config,pp);print('new best',m,config,flush=True)
  boundary_rules=build_rules(['training']);cropped=correct(best[2],docs,boundary_rules);cropped_metric=metric(cropped,gold)
  if cropped_metric[0]>best[0]:
   best=(cropped_metric[0],dict(best[1],boundaries=True),cropped);print('Boundary improvement',cropped_metric,flush=True)
  cleaned=remove_mutations(best[2],docs);cleaned_metric=metric(cleaned,gold)
  if cleaned_metric[0]>best[0]:
   best=(cleaned_metric[0],dict(best[1],mutations=True),cleaned);print('Mutation improvement',cleaned_metric,flush=True)
  Path(args.config).write_text(json.dumps({'config':best[1],'development_metrics':metric(best[2],gold)},indent=2));write(args.out,best[2]);print('FINAL BEST',best[:2],flush=True)
 else:
  config=json.loads(Path(args.config).read_text())['config'];weights=config.get('weights') or [1]*len(records);scores,offs=merge(records,docs,weights,config.get('method','probability'))
  if config.get('case_strength'):
   cats=[np.array([category(t,a,b) for a,b in off],np.int8) for (_,_,t),off in zip(docs,offs)];scores=adjust(scores,cats,np.array(config['case_weights'],np.float32),config['case_strength'])
  pred=decode(scores,docs,offs,config['bias']);auto,stats=build(['training','development'] if args.split=='test' else ['training']);pred=apply(pred,docs,auto,stats,config)
  if config.get('boundaries'):pred=correct(pred,docs,build_rules(['training','development'] if args.split=='test' else ['training']))
  if config.get('mutations'):pred=remove_mutations(pred,docs)
  write(args.out,pred)
  if gold:print(metric(pred,gold),flush=True)
  print('Saved',len(pred),'mentions',flush=True)

if __name__=='__main__':main()
