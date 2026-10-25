#!/bin/bash
set -e
cd /home/agent
for stage in 5 6 7; do
 case "$stage" in
  5) run_name=large-a;;
  6) run_name=link-a;;
  7) run_name=case-b;;
 esac
 while [ ! -s "runs/$run_name-pred/test.predictions.tsv" ]; do sleep 5; done
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 code/ensemble.py --config "code/ensemble$stage.json" --out "runs/ensemble$stage" > "runs/ensemble$stage.log" 2>&1
 done
