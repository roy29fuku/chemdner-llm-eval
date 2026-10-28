import argparse, json
from ner import read_docs,read_gold,write_predictions,metrics

# Exact standalone exclusions in the supplied annotation guidelines.
EXCLUDED=set('water aqueous organic inorganic proton protons lead gold atom atoms ion ions molecule molecules polymer polymers isomer isomers enantiomer enantiomers monomer monomers dimer dimers lipid lipids protein proteins aglycone aglycones oligosaccharide oligosaccharides glycoside glycosides insulin dna mrna collagen starch cellulose glycogen peptidoglycan peptidoglycans glycoprotein glycoproteins melanin analogue analogues inhibitor inhibitors agonist agonists antioxidant antioxidants drug drugs compound compounds hormone hormones antibiotic antibiotics toxin toxins metabolite metabolites solvent solvents reagent reagents buffer buffers catalyst catalysts electrophile electrophiles cation cations anion anions acid base metal metals alkali'.split())|{'functional group','functional groups'}

def filter_mentions(pred,docs):
    byid={d['pmid']:d for d in docs};out=set()
    for p,s,b,e in pred:
        d=byid[p];shift=0 if s=='T' else d['split'];name=d['text'][b+shift:e+shift]
        if name.casefold() not in EXCLUDED:out.add((p,s,b,e))
    return out

def load_predictions(path):
    out=set()
    for l in open(path):
        p,x=l.strip().split('\t');s,b,e=x.split(':');out.add((p,s,int(b),int(e)))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True);ap.add_argument('--out',required=True);ap.add_argument('--data',default='/workspace/data');ap.add_argument('--split',default='development');a=ap.parse_args()
    docs=read_docs(f'{a.data}/{a.split}.abstracts.txt');pred=load_predictions(a.predictions);filtered=filter_mentions(pred,docs)
    print('removed',len(pred)-len(filtered))
    if a.split!='test':
        _,gold=read_gold(f'{a.data}/{a.split}.annotations.txt',docs)
        print('before',json.dumps(metrics(pred,gold)));print('after',json.dumps(metrics(filtered,gold)))
    write_predictions(filtered,a.out)

if __name__=='__main__':main()
