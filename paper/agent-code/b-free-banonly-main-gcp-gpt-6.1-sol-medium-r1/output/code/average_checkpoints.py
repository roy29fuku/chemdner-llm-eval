import argparse, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoints',nargs='+')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    paths=[Path(p) for p in args.checkpoints];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    state=load_file(str(paths[0]/'model.safetensors'))
    for key in state:state[key]=state[key].float()/len(paths)
    for path in paths[1:]:
        other=load_file(str(path/'model.safetensors'))
        for key in state:state[key].add_(other[key].float(),alpha=1/len(paths))
    save_file(state,str(out/'model.safetensors'),metadata={'format':'pt'})
    shutil.copy(paths[0]/'config.json',out/'config.json')
    print('Averaged',len(paths),'checkpoints to',out,flush=True)

if __name__=='__main__':main()
