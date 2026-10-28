#!/bin/bash
cd /home/agent
while [ ! -s runs/holdout-b-pred/test.predictions.tsv ]; do sleep 5; done
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 code/ensemble.py --config code/ensemble2.json --out runs/ensemble2 > runs/ensemble2.log 2>&1
