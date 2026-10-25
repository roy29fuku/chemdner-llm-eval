#!/bin/bash
while ! test -f run2/test.tsv; do sleep 10; done
python3 -u code/train.py --out run3 --epochs 8 --seed 123 --model michiyasunaga/BioLinkBERT-base > run3.log 2>&1
python3 -u code/train.py --out run4 --epochs 6 --seed 1234 --lr 1e-5 --batch 8 --model michiyasunaga/BioLinkBERT-large > run4.log 2>&1
python3 -u code/train.py --out run5 --epochs 8 --seed 777 > run5.log 2>&1
python3 -u code/train.py --out full1 --epochs 6 --seed 42 --full > full1.log 2>&1
python3 -u code/train.py --out full2 --epochs 6 --seed 2026 --full --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract > full2.log 2>&1
python3 -u code/train.py --out full3 --epochs 6 --seed 123 --full --model michiyasunaga/BioLinkBERT-base > full3.log 2>&1
python3 -u code/train.py --out full5 --epochs 6 --seed 777 --full > full5.log 2>&1
python3 -u code/train.py --out full4 --epochs 5 --seed 1234 --lr 1e-5 --batch 8 --full --model michiyasunaga/BioLinkBERT-large > full4.log 2>&1
