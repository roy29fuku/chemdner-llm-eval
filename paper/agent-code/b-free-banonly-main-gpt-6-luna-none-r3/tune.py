import solve,collections
arts=solve.load_articles('development.abstracts.txt'); gold=solve.load_ann('development.annotations.txt')
# evaluate baseline but classify predicted candidates into lex / pat / morph by reproduction; alternatives thresholds frequency

def evalmin(minn=2, minshort=6, morph=True, patterns=True, extras=True):
 ff={k:v for k,v in solve.freq.items() if v>=minn and (len(k)>3 or v>=minshort)}
 # rebuild regex
 forms=[]
 for f,n in ff.items():
  if len(f)<2 or f in {'no','co','in','as','or','at','is','it','he','h','n','c','s','o','ca','mg','al','and','of','to','by','on','up','do','be','we','all','may','can','was','are','not','use','used','cell','cells','acid','base','salt','metal','drug','drugs','compound','compounds','protein','proteins','water','group','groups'}: continue
  forms.append(f)
 forms.sort(key=len,reverse=True); rx=__import__('re').compile(r'(?<![\w])(?:'+'|'.join(__import__('re').escape(x) for x in forms)+r')(?![\w])',__import__('re').I)
 # temporarily patch
 old=solve.lex_re; solve.lex_re=rx
 tot=[0,0,0]; fpw=collections.Counter(); fnw=collections.Counter(); count=collections.Counter()
 for pm,t,a in arts:
  pred=set()
  for sec,s in [('T',t),('A',a)]:
   for x,y in solve.predict(s): pred.add((sec,x,y))
  g=gold[pm]; tot[0]+=len(pred&g);tot[1]+=len(pred-g);tot[2]+=len(g-pred)
  for sec,x,y in pred-g: fpw[(t if sec=='T' else a)[x:y].lower()]+=1
  for sec,x,y in g-pred: fnw[(t if sec=='T' else a)[x:y].lower()]+=1
 solve.lex_re=old
 p,r=tot[0]/(tot[0]+tot[1]),tot[0]/(tot[0]+tot[2]); return 2*p*r/(p+r),tot,fpw,fnw
for n in [1,2,3,4,5,8,10,15]:
 f,tot,fp,fn=evalmin(n,5)
 print(n,round(f,4),tot,'FP common',fp.most_common(12),'FN',fn.most_common(8))
