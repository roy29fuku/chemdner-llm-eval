"""Corpus-derived mention lexicon, used only for measured postprocessing experiments."""
from ner import *
import ahocorasick

def lower(text):
    return ''.join(c.lower()[0] for c in text)

def build_automaton(names):
    automaton=ahocorasick.Automaton()
    for name in names:
        automaton.add_word(name,name)
    automaton.make_automaton()
    return automaton

def matches(automaton,text):
    for end,name in automaton.iter(lower(text)):
        b=end+1;a=b-len(name)
        if a and text[a-1].isalnum() and name[0].isalnum():continue
        if b<len(text) and text[b].isalnum() and name[-1].isalnum():continue
        yield a,b,name

def derive_lexicon(docs):
    positive=Counter()
    for d in docs:
        for text,anns in zip([d['title'],d['abstract']],d['annotations']):
            for a,b in anns:positive[lower(text[a:b])]+=1
    automaton=build_automaton(positive)
    exact=Counter();occurrences=Counter()
    for d in docs:
        for text,anns in zip([d['title'],d['abstract']],d['annotations']):
            gold=set(anns)
            for a,b,name in matches(automaton,text):
                occurrences[name]+=1
                if (a,b) in gold:exact[name]+=1
    return {name:dict(positive=n,exact=exact[name],occurrences=occurrences[name],precision=exact[name]/max(1,occurrences[name])) for name,n in positive.items()}

def predict(docs,lexicon,min_count=2,min_precision=.9):
    names={n for n,v in lexicon.items() if v['exact']>=min_count and v['precision']>=min_precision}
    auto=build_automaton(names)
    predictions=set()
    for d in docs:
        for sec,text in [('T',d['title']),('A',d['abstract'])]:
            ms=sorted(matches(auto,text),key=lambda x:(-(x[1]-x[0]),x[0]))
            chosen=[]
            for a,b,name in ms:
                if any(a<bb and b>aa for aa,bb in chosen):continue
                predictions.add((d['pmid'],sec,a,b));chosen.append((a,b))
    return predictions

if __name__=='__main__':
    train=read_documents('/workspace/data','training')
    dev=read_documents('/workspace/data','development')
    lex=derive_lexicon(train)
    json.dump(lex,open('/workspace/output/training_lexicon.json','w'))
    for n in [1,2,3]:
        print(n,metric(predict(dev,lex,n,.95),gold_spans(dev)),flush=True)
    lex=derive_lexicon(train+dev)
    json.dump(lex,open('/workspace/output/all_lexicon.json','w'))
    save_predictions(predict(read_documents('/workspace/data','test'),lex,1,.9),'/workspace/output/predictions.tsv')
