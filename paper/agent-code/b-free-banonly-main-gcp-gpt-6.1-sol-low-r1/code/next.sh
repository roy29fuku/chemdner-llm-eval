#!/bin/bash
while kill -0 1702 2>/dev/null; do sleep 10; done
python3 -u code/train.py --out run2 --epochs 8 --seed 2026 --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract > run2.log 2>&1
