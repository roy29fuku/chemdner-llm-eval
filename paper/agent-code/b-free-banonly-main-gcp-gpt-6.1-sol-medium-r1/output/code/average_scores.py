from ensemble import *
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('scores',nargs='+')
    parser.add_argument('--out',required=True)
    parser.add_argument('--arithmetic',action='store_true')
    args=parser.parse_args()
    records=[pickle.load(open(p,'rb')) for p in args.scores]
    docs=records[0]['docs']
    scores=[project_scores(r['docs'],r['scores'],docs) for r in records]
    merged=combine(scores,arithmetic=args.arithmetic)
    pickle.dump(dict(docs=docs,scores=merged),open(args.out,'wb'))
    print('Averaged',len(records),'score files into',args.out,flush=True)

if __name__=='__main__':main()
