import re
from collections import defaultdict
from lexicon import read,metric
from postprocess import load_preds
AA='Ala Arg Asn Asp Cys Glu Gln Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val'.split()
MUT=re.compile(r'(?<!\w)(?:'+ '|'.join(AA) +r')\d+(?:'+ '|'.join(AA) +r')(?!\w)')

def remove_mutations(pred,docs):
 invalid=defaultdict(list)
 for p,s,text in docs:invalid[p,s]=[m.span() for m in MUT.finditer(text)]
 return {x for x in pred if not any(a<=x[2] and x[3]<=b for a,b in invalid[x[0],x[1]])}

if __name__=='__main__':
 import sys
 docs,g=read('development');p=load_preds(sys.argv[1]);q=remove_mutations(p,docs);print('before',metric(p,g),'after',metric(q,g),'removed',len(p-q),'correct removed',len((p-q)&g))
