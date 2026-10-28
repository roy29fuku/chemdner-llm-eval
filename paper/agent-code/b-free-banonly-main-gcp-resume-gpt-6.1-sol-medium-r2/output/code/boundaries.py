import re,json
from pathlib import Path
from collections import defaultdict,Counter
from lexicon import read,metric
from postprocess import load_preds

WORD=re.compile(r'[A-Za-z0-9_]+')

def build_rules(splits=['training'],mincount=2):
 partial=defaultdict(Counter);full=Counter();numeric=defaultdict(Counter);numeric_full=Counter();contextual=defaultdict(Counter);named=Counter()
 for split in splits:
  docs,gold=read(split);bysec=defaultdict(list)
  for p,s,a,b in gold:bysec[p,s].append((a,b))
  for p,s,text in docs:
   anns=sorted(bysec[p,s]);j=0
   for a,b in anns:
    named[text[a:b].lower()]+=1;wa=a;wb=b
    while wa>0 and text[wa-1].isalnum() and a-wa<12:wa-=1
    while wb<len(text) and text[wb].isalnum() and wb-b<12:wb+=1
    surface=text[wa:wb]
    if (wa<a or wb>b) and len(surface.lower())==len(surface):contextual[surface.lower()][(a-wa,b-wa)]+=1
   for m in WORD.finditer(text):
    wa,wb=m.span();word=m.group();key=word.lower()
    while j<len(anns) and anns[j][1]<=wa:j+=1
    overlapping=[];k=j
    while k<len(anns) and anns[k][0]<wb:
     if anns[k][1]>wa:overlapping.append(anns[k])
     k+=1
    if any(a<=wa and b>=wb for a,b in overlapping):
     full[key]+=1
     if re.fullmatch(r'[A-Za-z]+\d+',word):numeric_full[re.sub(r'\d+$','',word)]+=1
    for a,b in overlapping:
     if wa<=a<b<=wb and (a>wa or b<wb):
      partial[key][(a-wa,b-wa)]+=1
      if a==wa and text[a:b].isalpha() and text[b:wb].isdigit():numeric[text[a:b]][(0,b-a)]+=1
 exact={};numbers={}
 for word,counts in partial.items():
  span,n=counts.most_common(1)[0]
  if n>=mincount and n/(sum(counts.values())+full[word])>=.9:exact[word]=span
 for word,counts in contextual.items():
  span,n=counts.most_common(1)[0]
  if n>=mincount and n/(sum(counts.values())+named[word])>=.9:exact[word]=span
 for base,counts in numeric.items():
  span,n=counts.most_common(1)[0]
  if n>=mincount and n/(sum(counts.values())+numeric_full[base])>=.9:numbers[base]=span
 return {'exact':exact,'numeric':numbers}

def correct(pred,docs,rules):
 texts={(p,s):t for p,s,t in docs};out=set()
 for p,s,a,b in pred:
  word=texts[p,s][a:b]
  span=rules['exact'].get(word.lower()) if len(word.lower())==len(word) else None
  if WORD.fullmatch(word) or span is not None:
   if span is None:
    m=re.fullmatch(r'([A-Za-z]+)\d+',word)
    if m:span=rules['numeric'].get(m.group(1))
   if span is not None:
    a,b=a+span[0],a+span[1]
  out.add((p,s,a,b))
 return out

if __name__=='__main__':
 import sys
 rules=build_rules();print('rules',rules);docs,g=read('development');pred=load_preds(sys.argv[1]);p=correct(pred,docs,rules);print('before',metric(pred,g),'after',metric(p,g));print('changed',len(p-pred),'correct added',len((p-pred)&g),'correct removed',len((pred-p)&g));Path('/workspace/results/validation/boundary_rules.json').write_text(json.dumps(rules,indent=2))
