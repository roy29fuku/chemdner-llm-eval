#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
while kill -0 15941 2>/dev/null; do sleep 10; done
python3 -u /home/agent/code/ner.py --data /home/agent/data --out /home/agent/val1/pred --model /home/agent/val1/best --predict > /home/agent/val1.predict.log 2>&1
python3 -u /home/agent/code/ner.py --data /home/agent/data --out /home/agent/vallarge --model /home/agent/biolinklarge --epochs 6 --batch 8 --accum 2 --lr 2e-5 --snapshots > /home/agent/vallarge.log 2>&1
