import argparse,json,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--count',type=int,default=3);args=ap.parse_args();root=Path(args.root);history=json.loads((root/'history.json').read_text());candidates=[]
for row in history:
 e=row['epoch']
 if e<3:continue
 choices=[root/f'epoch{e}',root/f'best_epoch{e}']
 path=next((p for p in choices if (p/'model.safetensors').exists()),None)
 if path:candidates.append((row['f1'],e,path))
candidates.sort(reverse=True);chosen=candidates[:args.count]
if not chosen:raise ValueError('No available checkpoints')
print('Average selection',[(score,e,str(p)) for score,e,p in chosen],flush=True)
subprocess.run(['python3',str(Path(__file__).with_name('average.py')),*[str(p) for _,_,p in chosen],'--out',str(root/'average')],check=True)
(root/'average_selection.json').write_text(json.dumps([{'f1':f,'epoch':e,'path':str(p)} for f,e,p in chosen],indent=2))
