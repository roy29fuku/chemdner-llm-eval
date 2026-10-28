import re,collections, numpy as np
from decoding import decode
# Explicitly excluded words in the task's annotation guidelines.
EXCLUDE={'lead','gold','water','aqueous','organic','inorganic','proton','protons'}
ELEMENTS=set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())
SHORT=set('D OH HO SH HS NH CH CN NC NO ON CO CS SC OD HD Me Et Ph 1H 2H H2 O2 O₃ N2 D2'.split())

def valid_word(word,filter_level=1):
    if not filter_level:return True
    if word.lower() in EXCLUDE:return False
    if len(word)<=2 and re.fullmatch('[A-Za-z0-9]+',word):
        if word in ELEMENTS or word in SHORT:return True
        if re.fullmatch('[HBCNOFPSKVWIU][0-9]',word):return True
        if len(word)==2 and all(x in 'HBCNOFPSKVWIU' for x in word):return True
        return False
    return True

def span_margin(doc,scores,sec,a,b):
    inds=[i for i,(s,x,y) in enumerate(doc['offsets']) if s==sec and y>a and x<b]
    if not inds:return -100
    if doc['offsets'][inds[0]][1]!=a or doc['offsets'][inds[-1]][2]!=b:return -100
    tags=[4] if len(inds)==1 else [1]+[2]*(len(inds)-2)+[3]
    return float(np.mean([scores[i,tag]-scores[i,0] for i,tag in zip(inds,tags)]))

def process(doc,scores,bias=0,filter_level=1,source=100,target=-1):
    pred=decode(doc,scores,bias)
    pred={x for x in pred if valid_word(doc['texts'][x[0]][x[1]:x[2]],filter_level)}
    if source>=100:return pred
    terms=collections.defaultdict(lambda:-100)
    for sec,a,b in pred:
        word=doc['texts'][sec][a:b]
        if len(word)<3:continue
        terms[word]=max(terms[word],span_margin(doc,scores,sec,a,b))
    additions=[]
    for word,confidence in terms.items():
        if confidence<source:continue
        pattern=re.compile(r'(?<![\w])'+re.escape(word)+r'(?![\w])')
        for sec,text in doc['texts'].items():
            for m in pattern.finditer(text):
                a,b=m.span();span=(sec,a,b)
                if span in pred:continue
                if any(s==sec and x<b and y>a for s,x,y in pred):continue
                if span_margin(doc,scores,sec,a,b)>=target:additions.append(span)
    return pred|set(additions)

def score(docs,scores,**kwargs):
    tp=fp=fn=0
    for d,s in zip(docs,scores):
        pred=process(d,s,**kwargs);gold=d['gold'];tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return dict(f1=2*p*r/max(p+r,1e-10),precision=p,recall=r,tp=tp,fp=fp,fn=fn)

if __name__=='__main__':
    import pickle,sys
    docs=pickle.load(open(sys.argv[1],'rb'));scores=pickle.load(open(sys.argv[2],'rb'))
    for filt in (0,1):
        for bias in (0,.2,.4,.6):
            print('filter',filt,'bias',bias,score(docs,scores,filter_level=filt,bias=bias),flush=True)
    for source in (0,1,2):
        for target in (-2,-1,0):
            print('source',source,'target',target,score(docs,scores,source=source,target=target,bias=.2),flush=True)
