import sys,subprocess,json,re,time
runs=sys.argv[1:];rows=[]
for avg in ['log','prob']:
 for w in [.5,1,1.5,2,3]:
  weights=[1,1,w];idx=len(rows);out=f'/workspace/results/weight_search_{idx}.tsv';t=time.time()
  cmd=['python3','/workspace/code/combine.py']+runs+['--devonly','--constraint','both','--average',avg,'--weights']+list(map(str,weights))+['--output',out]
  p=subprocess.run(cmd,text=True,capture_output=True);open(out+'.log','w').write(p.stdout)
  if p.returncode:print(p.stderr,flush=True);continue
  m=re.search(r'BEST \(([^,]+), ([^)]+)\)',p.stdout);f,bias=map(float,m.groups());row={'f1':f,'bias':bias,'begin_bias':0,'runs':runs,'average':avg,'weights':weights,'path':out+'.dev'};rows.append(row);print(row,'seconds',round(time.time()-t),flush=True)
  json.dump(sorted(rows,key=lambda r:-r['f1']),open('/workspace/results/weight_search.json','w'),indent=2)
print('BEST',max(rows,key=lambda r:r['f1']),flush=True)
