import collections,math,ahocorasick
from postprocess import valid_word
from lexicon import boundary

def normalize(s):return ''.join(c.lower() if len(c.lower())==1 else c for c in s)

class LexicalFeatures:
    def __init__(self,docs):
        positive=collections.Counter()
        for d in docs:
            for sec,a,b in d['gold']:
                term=d['texts'][sec][a:b]
                if valid_word(term):positive[normalize(term)]+=1
        matcher=ahocorasick.Automaton()
        for t in positive:matcher.add_word(t,t)
        matcher.make_automaton();negative=collections.Counter()
        for d in docs:
            for sec,text in d['texts'].items():
                gold=[(a,b) for s,a,b in d['gold'] if s==sec]
                for end,t in matcher.iter(normalize(text)):
                    a=end+1-len(t);b=end+1
                    if not boundary(text,a,b):continue
                    if (sec,a,b) in d['gold']:continue
                    if any(x<b and a<y for x,y in gold):continue
                    negative[t]+=1
        self.stats={t:(math.log1p(p),p/max(p+negative[t],1),math.log1p(negative[t])) for t,p in positive.items()}
        self.matcher=ahocorasick.Automaton()
        for t,s in self.stats.items():
            if s[1]>=.25 or len(t)<=2:self.matcher.add_word(t,t)
        self.matcher.make_automaton()
        print('Lexical candidate terms',len(self.matcher),flush=True)
    def candidates(self,doc):
        spans=set()
        for sec,text in doc['texts'].items():
            for end,t in self.matcher.iter(normalize(text)):
                a=end+1-len(t);b=end+1
                if boundary(text,a,b):spans.add((sec,a,b))
        return spans
    def features(self,word):return self.stats.get(normalize(word),(0,0,0))
