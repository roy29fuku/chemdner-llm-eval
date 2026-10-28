"""Regenerate a submission from the locally saved models and manifest."""
import argparse,json,pathlib
import torch
from transformers import AutoTokenizer
from ner import read_docs,Corpus,infer,load_model,collapse_typed,decode,write_predictions
from ensemble import probabilities,project,combine
from filter_rules import filter_mentions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--abstracts',required=True);ap.add_argument('--output',required=True);ap.add_argument('--batch',type=int,default=24);a=ap.parse_args()
    torch.set_num_threads(6)
    path=pathlib.Path(a.manifest).resolve();m=json.load(open(path));root=path.parent
    docs=read_docs(a.abstracts);canonical=Corpus(docs,AutoTokenizer.from_pretrained(root/m['canonical']))
    arrays=[];weights=[]
    for component in m['components']:
        if not component['weight']:continue
        model_path=root/component['model'];model=load_model(str(model_path)).to('cuda' if torch.cuda.is_available() else 'cpu')
        source=Corpus(docs,AutoTokenizer.from_pretrained(model_path),case_features=model.config.case_features)
        logits=infer(model,source,a.batch)
        arrays.append([project(probabilities(collapse_typed(s)),offsets,target,len(d['text'])) for s,offsets,target,d in zip(logits,source.offsets,canonical.offsets,docs)])
        weights.append(component['weight']);del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    decoder=m['decoder']
    pred=decode(combine(arrays,weights),canonical,decoder['bias'],decoder['method'],decoder['begin'])
    write_predictions(filter_mentions(pred,docs),a.output)

if __name__=='__main__':main()
