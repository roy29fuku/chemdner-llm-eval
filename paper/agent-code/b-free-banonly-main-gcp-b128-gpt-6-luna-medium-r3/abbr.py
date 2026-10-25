import re
import ner
SHORT=re.compile(r'^[A-Za-z][A-Za-z0-9+\-/]{2,14}$')
WORD=re.compile(r"[\w+\-/]+",re.UNICODE)

def matches_abbr(short,long):
    a=''.join(c for c in short.lower() if c.isalnum())
    l=''.join(c for c in long.lower() if c.isalpha())
    if len(a)<3 or len(a)>len(l):return False
    # abbreviation characters must be an ordered subsequence, matched from the end
    i=len(a)-1
    for c in reversed(l):
        if c==a[i]:
            i-=1
            if i<0:return True
    return False

def abbreviation_candidates(text):
    found=[]
    for m in re.finditer(r'\(([^()]{2,20})\)',text):
        short=m.group(1).strip()
        if not SHORT.fullmatch(short):continue
        # Ignore ordinary sentence labels and very short initialisms.
        if not any(c.isupper() for c in short) or len(short)<3:continue
        pre=text[max(0,m.start()-180):m.start()]
        words=list(WORD.finditer(pre))
        # Choose the shortest ending phrase whose letters expand the short form.
        expansion=None
        for start in range(max(0,len(words)-15),len(words)):
            phrase=pre[words[start].start():].strip()
            if matches_abbr(short,phrase):
                expansion=phrase;break
        if expansion:
            found.append((m.start(1)+m.group(1).find(short),m.start(1)+m.group(1).find(short)+len(short),short,expansion))
    out=set()
    for b,e,short,expansion in found:
        pattern=re.compile(r'(?<!\w)'+re.escape(short)+r'(?!\w)',re.I)
        for m in pattern.finditer(text):out.add((m.start(),m.end()))
    return out,found
