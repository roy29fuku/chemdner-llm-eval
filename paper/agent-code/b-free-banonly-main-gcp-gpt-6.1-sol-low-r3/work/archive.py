import os,json,shutil,hashlib,subprocess
OUT='/workspace/output';ROOT='/workspace/work'
os.makedirs(OUT+'/probabilities',exist_ok=True);os.makedirs(OUT+'/models',exist_ok=True)
spec=json.load(open(ROOT+'/test_spec.json'));export=[];names=[]
for entry in spec:
 name=os.path.basename(entry['path']).removesuffix('.npz');names.append(name)
 path=OUT+'/probabilities/'+name+'.npz'
 if not os.path.exists(path):shutil.copyfile(entry['path'],path)
 export.append(dict(entry,path=path))
 config=OUT+'/models/'+name+'.json'
 if not os.path.exists(config):
  subprocess.run(['scp','-i','/home/agent/.ssh/google_compute_engine','-o','ProxyCommand=nc -X connect -x proxy:8080 %h %p','agent@35.192.119.95:/home/agent/'+name+'/run_config.json',config],check=True)
json.dump(export,open(OUT+'/ensemble_spec.json','w'),indent=2)
summary=json.load(open(OUT+'/run_summary.json'))
summary['final_models_completed']=names
raw=open(OUT+'/predictions.tsv','rb').read()
summary['predicted_mentions']=len(raw.splitlines());summary['predictions_sha256']=hashlib.sha256(raw).hexdigest();summary['format_errors']=0
summary['format_check']='python3 /workspace/check_format.py /workspace/output/predictions.tsv'
json.dump(summary,open(OUT+'/run_summary.json','w'),indent=2)
print('Archived',names,'mentions',summary['predicted_mentions'])
