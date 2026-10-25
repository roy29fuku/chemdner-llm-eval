from model_fast import *
from collections import Counter
import re, time, pickle
tr=read_labels('training')
forms=Counter(m.lower() for aa in tr.values() for a,b,c,m in aa if len(m)>2 and any(ch.isalpha() for ch in m))
terms=sorted(forms,key=lambda x:-len(x))
pat=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in terms)+r')(?![A-Za-z0-9])',re.I)
pickle.dump(pat,open('/workspace/dictpat.pkl','wb'))
print('built',len(terms),flush=True)
