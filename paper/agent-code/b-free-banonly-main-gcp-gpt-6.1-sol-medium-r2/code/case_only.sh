#!/bin/bash
set -e
cd /home/agent
while pgrep -f '^bash code/(pipeline|large|link).sh$' > /dev/null; do sleep 5; done
python3 code/case_ner.py --full --out runs/case-b --epochs 8 --seed 211 > runs/case-b.log 2>&1
python3 code/case_ner.py --mode predict --checkpoint runs/case-b/best --out runs/case-b-pred > runs/case-b-pred.log 2>&1
