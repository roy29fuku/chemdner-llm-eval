import glob,os,json
logs=glob.glob('/home/agent/full_*.log')+glob.glob('/home/agent/pred_full_*.log')
if logs:
 latest=max(logs,key=os.path.getmtime)
 print('Active log:',os.path.basename(latest))
 print(''.join(open(latest).readlines()[-4:]).strip())
print('Complete prediction models:',[p.split('/')[-2] for p in sorted(glob.glob('/home/agent/pred_full_*/test_probs.npz'))])
print('Pipeline:',open('/home/agent/final_pipeline.log').read().strip())
