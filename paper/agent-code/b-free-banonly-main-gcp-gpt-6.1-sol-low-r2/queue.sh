#!/bin/bash
cd /home/agent
while pgrep -f '^python3 -u train.py ' >/dev/null; do sleep 10; done
python3 -u train_v2.py --model michiyasunaga/BioLinkBERT-large --batch 12 --length 256 --lr 2e-5 --epochs 7 --seed 43 --out largeval > largeval.log 2>&1
