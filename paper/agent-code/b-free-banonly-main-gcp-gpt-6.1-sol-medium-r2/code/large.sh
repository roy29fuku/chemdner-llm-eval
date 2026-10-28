#!/bin/bash
set -e
cd /home/agent
while pgrep -f '^bash code/pipeline.sh$' > /dev/null; do sleep 5; done
python3 code/case_ner.py --full --typed --out runs/typed-a --epochs 8 --seed 101 > runs/typed-a.log 2>&1
python3 code/case_ner.py --mode predict --checkpoint runs/typed-a/best --out runs/typed-a-pred > runs/typed-a-pred.log 2>&1
python3 code/ner.py --full --out runs/large-a --model microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract --epochs 8 --batch 24 --lr 2e-5 --seed 107 > runs/large-a.log 2>&1
python3 code/ner.py --mode predict --model microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract --checkpoint runs/large-a/best --out runs/large-a-pred --batch 24 > runs/large-a-pred.log 2>&1
