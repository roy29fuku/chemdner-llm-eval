import subprocess,os,time,json
ROOT='/workspace/work';DEST='/workspace/output/predictions.tsv';BASE='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
SSH=['ssh','-i','/home/agent/.ssh/google_compute_engine','-o','ProxyCommand=nc -X connect -x proxy:8080 %h %p','agent@35.192.119.95']
SCP=['scp','-i','/home/agent/.ssh/google_compute_engine','-o','ProxyCommand=nc -X connect -x proxy:8080 %h %p']
os.makedirs(ROOT+'/test_probs',exist_ok=True);last=[]
while True:
 try:
  result=subprocess.run(SSH+["find /home/agent -maxdepth 3 -path '/home/agent/pred_full_*/test_probs.npz'"],capture_output=True,text=True,check=True)
  paths=sorted(result.stdout.strip().splitlines());spec=[]
  weights=json.load(open(ROOT+'/weights.json')) if os.path.exists(ROOT+'/weights.json') else {}
  for path in paths:
   name=path.split('/')[-2].removeprefix('pred_');local=ROOT+'/test_probs/'+name+'.npz'
   if not os.path.exists(local):subprocess.run(SCP+['agent@35.192.119.95:'+path,local],check=True)
   weight=weights.get(name,1)
   if weight:
    spec.append(dict(path=local,model=BASE if '_abs' not in name else 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',weight=weight))
  scale=weights.get('_scale',1)
  if spec and (spec,scale)!=last:
   json.dump(spec,open(ROOT+'/test_spec.json','w'),indent=2)
   subprocess.run(['python3',ROOT+'/ensemble_lexword.py','--split','test','--spec',ROOT+'/test_spec.json','--out',DEST+'.new','--scale',str(scale)],check=True)
   os.replace(DEST+'.new',DEST)
   subprocess.run(['python3','/workspace/check_format.py',DEST],check=True)
   last=(spec,scale);print('Updated predictions from',len(spec),'models at',time.strftime('%H:%M:%S'),flush=True)
 except Exception as e:print('Monitor retry',str(e),flush=True)
 time.sleep(45)
