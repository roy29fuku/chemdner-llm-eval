#!/bin/bash
set -e
cd /home/agent
while pgrep -f '^bash code/large.sh$' > /dev/null; do sleep 5; done
python3 code/ner.py --full --out runs/link-a --model michiyasunaga/BioLinkBERT-large --epochs 8 --batch 24 --lr 2e-5 --seed 137 > runs/link-a.log 2>&1
python3 code/ner.py --mode predict --model michiyasunaga/BioLinkBERT-large --checkpoint runs/link-a/best --out runs/link-a-pred --batch 24 > runs/link-a-pred.log 2>&1
