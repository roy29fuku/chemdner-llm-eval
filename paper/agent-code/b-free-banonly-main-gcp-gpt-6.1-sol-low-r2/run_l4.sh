#!/bin/bash
cd /home/agent
python3 -u train_v2.py --model michiyasunaga/BioLinkBERT-large --batch 24 --length 384 --lr 2e-5 --epochs 7 --seed 43 --out largeval > largeval.log 2>&1
python3 -u train_joint.py --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --batch 32 --length 384 --lr 3e-5 --epochs 7 --seed 44 --out jointval > jointval.log 2>&1
python3 -u train_typed.py --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --batch 32 --length 384 --lr 3e-5 --epochs 7 --seed 45 --out typedval > typedval.log 2>&1
