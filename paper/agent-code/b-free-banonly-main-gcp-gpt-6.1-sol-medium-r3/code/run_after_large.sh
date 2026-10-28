#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
while kill -0 23043 2>/dev/null; do sleep 10; done
python3 -u /home/agent/code/ner.py --data /home/agent/data --out /home/agent/vallarge/pred --model /home/agent/vallarge/best --predict > /home/agent/vallarge.predict.log 2>&1
python3 -u /home/agent/code/ner.py --data /home/agent/data --out /home/agent/valabstract --model /home/agent/pretrained-abstract --epochs 8 --batch 16 --seed 43 --snapshots > /home/agent/valabstract.log 2>&1
python3 -u /home/agent/code/ner.py --data /home/agent/data --out /home/agent/valabstract/pred --model /home/agent/valabstract/best --predict > /home/agent/valabstract.predict.log 2>&1
