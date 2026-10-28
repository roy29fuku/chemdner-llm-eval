from ensemble import *
from postprocess import *
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('scores',nargs='+')
    parser.add_argument('--settings',required=True)
    parser.add_argument('--lexicon',required=True)
    parser.add_argument('--boundary-map',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    config=json.load(open(args.settings))
    setting=config.get('settings',config)
    records=[pickle.load(open(p,'rb')) for p in args.scores]
    assert len(records)==len(setting['weights']), 'Number of model score files must match ensemble weights'
    docs=records[0]['docs']
    scores=[project_scores(r['docs'],r['scores'],docs) for r in records]
    merged=combine(scores,setting['weights'],setting['arithmetic'])
    if setting['lexicon_strength']:
        merged=lexicon_boost(docs,merged,json.load(open(args.lexicon)),setting['lexicon_strength'])
    if setting.get('role_strength',0):
        merged=lexicon_boost(docs,merged,json.load(open(args.lexicon)),setting['role_strength'],min_count=1,min_precision=0,role_context=True)
    pred=spans_from_scores(docs,merged,setting['bias'],json.load(open(args.boundary_map)))
    pred=filter_predictions(pred,docs,short=setting['short_filter'])
    if setting['propagate_min_length']:
        pred=propagate(pred,docs,setting['propagate_min_length'])
        pred=filter_predictions(pred,docs,short=setting['short_filter'])
    save_predictions(pred,args.out)
    print(f'{len(pred)} mentions from {len(docs)} articles written to {args.out}',flush=True)

if __name__=='__main__':main()
