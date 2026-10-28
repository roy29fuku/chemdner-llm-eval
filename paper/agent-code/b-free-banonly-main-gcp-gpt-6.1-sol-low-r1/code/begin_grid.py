import subprocess,re,json,sys,time,argparse
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--weights',nargs='+',type=float);p.add_argument('--name');p.add_argument('--average',default='log');args=p.parse_args();runs=args.runs;weights=args.weights or [1]*len(runs);label=args.name or f'begin{len(runs)}';rows=[]
for i,b in enumerate([-.5,-.25,0,.25,.5]):
 out=f'/workspace/results/{label}_{i}.tsv';t=time.time()
 cmd=['python3','/workspace/code/combine.py']+runs+['--devonly','--constraint','both','--average',args.average,'--weights']+list(map(str,weights))+['--begin-bias',str(b),'--output',out]
 p=subprocess.run(cmd,text=True,capture_output=True);open(out+'.log','w').write(p.stdout)
 if p.returncode:print(p.stderr,flush=True);continue
 m=re.search(r'BEST \(([^,]+), ([^)]+)\)',p.stdout);f,bias=map(float,m.groups());row={'f1':f,'bias':bias,'begin_bias':b,'runs':runs,'average':args.average,'weights':weights,'path':out+'.dev'};rows.append(row);print(row,'seconds',round(time.time()-t),flush=True)
 json.dump(sorted(rows,key=lambda x:-x['f1']),open(f'/workspace/results/{label}.json','w'),indent=2)
print('BEST',max(rows,key=lambda x:x['f1']),flush=True)
