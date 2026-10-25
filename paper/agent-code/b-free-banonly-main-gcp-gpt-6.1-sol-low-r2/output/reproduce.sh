#!/bin/bash
set -e
cd /workspace
python3 output/code/eval.py --split test --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --scores output/scores/fulltext.npz output/scores/large.npz output/scores/typed.npz output/scores/cased.npz output/scores/auxiliary_context.npz --weights 1 .5 1 .5 1 --bias -.5 --ibias .5 --out output/predictions.tsv
python3 check_format.py output/predictions.tsv
