import time,json,shutil
from pathlib import Path
root=Path.home()/'runs/validation';seen=set();best=0
while True:
 path=root/'history.json'
 if path.exists():
  try:hist=json.loads(path.read_text())
  except:hist=[]
  for row in hist:
   e=row['epoch']
   if e in seen:continue
   seen.add(e)
   if row['f1']>best:
    best=row['f1']
    if (root/'best').exists() and e==hist[-1]['epoch']:
     shutil.copytree(root/'best',root/f'best_epoch{e}',dirs_exist_ok=True)
     print('Copied epoch',e,flush=True)
  if len(hist)>=7:break
 time.sleep(8)
