s=open('/workspace/train_v2.py').read()
a=s.index('def chunks(');b=s.index('def collate(',a)
s=s[:a]+'''def chunks(docs,train=False):
 items=[]
 for d in range(0,len(docs),2):
  title=docs[d]; abstract=docs[d+1]; tids=title['ids']; tlabs=title['labs']; prefix=tids[:128]; cap=args.length-len(prefix)-3
  n=len(abstract['ids']); start=0
  while start<n or (start==0 and n==0):
   end=min(start+cap,n)
   ids=[tok.cls_token_id]+prefix+[tok.sep_token_id]+abstract['ids'][start:end]+[tok.sep_token_id]
   labs=[-100]+(tlabs[:128] if start==0 else [-100]*len(prefix))+[-100]+abstract['labs'][start:end]+[-100]
   maps=[None]+([(d,i) for i in range(len(prefix))] if start==0 else [None]*len(prefix))+[None]+[(d+1,i) for i in range(start,end)]+[None]
   items.append(dict(input_ids=ids,labels=labs,maps=maps))
   if end==n:break
   start+=max(cap-48,1)
  if len(tids)>128:
   for st in range(128,len(tids),args.length-2):
    en=min(st+args.length-2,len(tids));items.append(dict(input_ids=[tok.cls_token_id]+tids[st:en]+[tok.sep_token_id],labels=[-100]+tlabs[st:en]+[-100],maps=[None]+[(d,i) for i in range(st,en)]+[None]))
 return items
''' + s[b:]
a=s.index("   for x,l in zip(part,logits):");b=s.index('\n return [s/',a)
s=s[:a]+'''   for x,l in zip(part,logits):
    n=len(x['maps'])
    for j,m in enumerate(x['maps']):
     if m is None:continue
     d,i=m;w=min(j+1,n-j,32)/32
     sums[d][i]+=l[j]*w;weights[d][i]+=w
''' + s[b:]
open('/workspace/train_joint.py','w').write(s)
