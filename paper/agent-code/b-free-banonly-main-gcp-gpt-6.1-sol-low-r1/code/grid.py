import sys,subprocess,re,json,itertools,os,time
runs=sys.argv[1:];records=[];os.makedirs('/workspace/results',exist_ok=True)
settings=[]
if len(runs)==2:
 for avg in ['log','prob']:
  for second in [.5,1,2]:settings.append((runs,[1,second],avg))
else:
 for avg in ['log','prob']:
  for n in range(2,len(runs)+1):
   for subset in itertools.combinations(runs,n):settings.append((list(subset),[1]*n,avg))
  for weighted in ([2 if r.endswith(('run1','run5')) else 1 for r in runs],[2 if r.endswith('run4') else 1 for r in runs]):settings.append((runs,weighted,avg))
for i,(subset,weights,avg) in enumerate(settings):
 out=f'/workspace/results/grid{len(runs)}_{i}.tsv'
 cmd=['python3','/workspace/code/combine.py']+subset+['--devonly','--constraint','both','--average',avg,'--weights']+list(map(str,weights))+['--output',out]
 t=time.time();p=subprocess.run(cmd,text=True,capture_output=True)
 if p.returncode:print('FAILED',p.stderr,flush=True);continue
 open(out+'.log','w').write(p.stdout)
 m=re.search(r'BEST \(([^,]+), ([^)]+)\)',p.stdout);f,bias=map(float,m.groups());row={'runs':subset,'weights':weights,'average':avg,'f1':f,'bias':bias,'path':out+'.dev'};records.append(row)
 print(i+1,'/',len(settings),row,'seconds',round(time.time()-t),flush=True)
 json.dump(sorted(records,key=lambda x:-x['f1']),open(f'/workspace/results/grid{len(runs)}.json','w'),indent=2)
print('BEST',max(records,key=lambda x:x['f1']),flush=True)
