import argparse,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ap=argparse.ArgumentParser();ap.add_argument('models',nargs='+');ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(8)
root=Path(args.models[0]);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
for p in root.iterdir():
 if p.suffix in ['.json','.txt']:shutil.copy2(p,out/p.name)
state=None
for path in args.models:
 d=load_file(str(Path(path)/'model.safetensors'))
 if state is None:state={k:v.clone()/len(args.models) for k,v in d.items()}
 else:
  for k,v in d.items():state[k].add_(v,alpha=1/len(args.models))
 del d
save_file(state,str(out/'model.safetensors'),metadata={'format':'pt'})
print('Averaged',len(args.models),'models into',out)
