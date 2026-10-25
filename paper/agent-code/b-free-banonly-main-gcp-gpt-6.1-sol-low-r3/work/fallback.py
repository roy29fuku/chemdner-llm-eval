import subprocess,time,json,os
ROOT='/workspace/work';DEST='/workspace/output/predictions.tsv'
SSH=['ssh','-i','/home/agent/.ssh/google_compute_engine','-o','ProxyCommand=nc -X connect -x proxy:8080 %h %p','agent@35.192.119.95']
SCP=['scp','-i','/home/agent/.ssh/google_compute_engine','-o','ProxyCommand=nc -X connect -x proxy:8080 %h %p']
while subprocess.run(SSH+['test -f /home/agent/baseline_test/test_probs.npz']).returncode:time.sleep(45)
subprocess.run(SCP+['agent@35.192.119.95:/home/agent/baseline_test/test_probs.npz',ROOT+'/baseline_test.npz'],check=True)
json.dump([dict(path=ROOT+'/baseline_test.npz')],open(ROOT+'/baseline_spec.json','w'))
if not os.path.exists(DEST):
 subprocess.run(['python3',ROOT+'/ensemble_lexword.py','--split','test','--spec',ROOT+'/baseline_spec.json','--out',DEST,'--scale','.6'],check=True)
 subprocess.run(['python3','/workspace/check_format.py',DEST],check=True)
 print('Baseline predictions safely saved',flush=True)
