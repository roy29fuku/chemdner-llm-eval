#!/bin/bash
set -e
cd /home/agent
while kill -0 2095 2>/dev/null && [ "$(ps -o stat= -p 2095 | cut -c1)" != Z ]; do sleep 5; done
python3 code/ner.py --mode predict --checkpoint runs/baseline/best --out runs/baseline-pred > runs/baseline-pred.log 2>&1
python3 code/ner.py --full --checkpoint runs/baseline/best --out runs/holdout-a --epochs 5 --lr 2e-5 --seed 17 > runs/holdout-a.log 2>&1
python3 code/ner.py --mode predict --checkpoint runs/holdout-a/best --out runs/holdout-a-pred > runs/holdout-a-pred.log 2>&1
python3 code/ner.py --full --out runs/holdout-b --epochs 8 --seed 42 > runs/holdout-b.log 2>&1
python3 code/ner.py --mode predict --checkpoint runs/holdout-b/best --out runs/holdout-b-pred > runs/holdout-b-pred.log 2>&1
python3 code/ner.py --full --out runs/holdout-c --size 384 --batch 16 --epochs 8 --seed 73 > runs/holdout-c.log 2>&1
python3 code/ner.py --mode predict --checkpoint runs/holdout-c/best --out runs/holdout-c-pred --size 384 --batch 16 > runs/holdout-c-pred.log 2>&1
