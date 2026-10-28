import argparse,os
import torch
from transformers import AutoModelForTokenClassification,AutoTokenizer
from ner import load_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(6)
    model=load_model(a.models[0]);state=model.state_dict()
    for key in state:state[key]=state[key].float()/len(a.models)
    for path in a.models[1:]:
        other=load_model(path).state_dict()
        for key in state:state[key]+=other[key].float()/len(a.models)
        del other
    model.load_state_dict(state);model.save_pretrained(a.out);AutoTokenizer.from_pretrained(a.models[0]).save_pretrained(a.out)
    print('averaged',len(a.models),'models to',a.out)

if __name__=='__main__':main()
