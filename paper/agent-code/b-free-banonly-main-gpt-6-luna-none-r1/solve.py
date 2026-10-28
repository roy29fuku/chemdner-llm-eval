import re, os, collections, unicodedata
import ahocorasick

BASE='/workspace/data/'
def read_abstracts(path):
    for line in open(path,encoding='utf8'):
        pm,title,abstract=line.rstrip('\n').split('\t')
        yield pm,{'T':title,'A':abstract}

def read_annotations(path):
    d=collections.defaultdict(lambda:{'T':[],'A':[]})
    for line in open(path,encoding='utf8'):
        pm,sec,a,b,text,cls=line.rstrip('\n').split('\t')
        d[pm][sec].append((int(a),int(b),text,cls))
    return d

def norm(s):
    return unicodedata.normalize('NFKC',s).casefold().replace('−','-').replace('–','-')

def main():
    train=read_annotations(BASE+'training.annotations.txt')
    dev=read_annotations(BASE+'development.annotations.txt')
    counts=collections.Counter(); classes=collections.defaultdict(collections.Counter)
    for d in (train,dev):
        for secs in d.values():
            for vals in secs.values():
                for a,b,text,cl in vals:
                    n=norm(text)
                    counts[n]+=1; classes[n][cl]+=1
    # Keep observed full spans, weighted toward frequent, chemically informative mentions.
    # Very common single words are mostly genuine; a stoplist suppresses common non-chemical ambiguous forms.
    stop=set('a an as at be by do go he if in is it me my no of on or so to up us we'.split())
    # Discover aliases/formula patterns from labeled text, then use strict lexical shape criteria.
    test=list(read_abstracts(BASE+'test.abstracts.txt'))
    out=[]
    # A single alternation trie-style regex is substantially faster than one scan per lexicon key.
    keys=[k for k in counts if len(k)>=2 and k not in stop]
    # Frequent single-token common words are retained; rare one-off terms are excluded below.
    keys=[k for k in keys if (' ' in k or not re.fullmatch(r'[a-z]{1,5}',k) or counts[k]>=5)]
    keys.sort(key=len,reverse=True)
    lex=ahocorasick.Automaton()
    for k in keys: lex.add_word(k,(len(k),k))
    lex.make_automaton()
    for pm,sections in test:
        for sec,text in sections.items():
            nt=norm(text); spans=set()
            # Exact dictionary mentions by longest-first literal case-insensitive matching.
            nt=norm(text)
            hits=[]
            for end,(ln,key) in lex.iter(nt):
                start=end-ln+1
                if (start==0 or not (nt[start-1].isalnum() or nt[start-1]=='_')) and (end+1==len(nt) or not (nt[end+1].isalnum() or nt[end+1]=='_')):
                    hits.append((start,end+1,key))
            # An observed longer chemical name subsumes nested dictionary hits.
            for start,end,key in hits:
                if not any(s<=start and e>=end and (s<start or e>end) for s,e,k in hits):
                    spans.add((start,end))
            # Chemical-looking formulas and registry identifiers. Require dense digits/element symbols,
            # avoiding ordinary alphanumeric study labels by checking training-derived lexicon only for IDs.
            # Explicit formulae/ionic forms are already present in the annotated lexicon;
            # avoid broad formula patterns that confuse gene names and study IDs.
            for a,b in sorted(spans): out.append(f'{pm}\t{sec}:{a}:{b}\n')
    os.makedirs('/workspace/output',exist_ok=True)
    open('/workspace/output/predictions.tsv','w',encoding='utf8').writelines(out)
    print('predictions',len(out))

if __name__=='__main__': main()
