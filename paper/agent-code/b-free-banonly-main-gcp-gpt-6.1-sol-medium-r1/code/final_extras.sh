#!/bin/bash
set -u
cd /home/agent/cem
export TOKENIZERS_PARALLELISM=false
while kill -0 13249 2>/dev/null; do sleep 5; done

python3 -u code/ner.py train --model allenai/biomed_roberta_base --data data --out runs/roberta_all --splits training,development --epochs 6 --batch 24 --length 384 --lr 0.00002 --seed 71 --eval-every 99 --save-epochs 3,4,5,6 --bucket --split-punct
python3 -u code/ner.py predict --model allenai/biomed_roberta_base --data data --out runs/roberta_all/test_eval --checkpoint runs/roberta_all/last --eval test --batch 24 --split-punct

if [ "$(date -u +%s)" -lt "$(date -u -d '2026-10-14 10:50:00' +%s)" ]; then
  python3 -u code/ner.py train --model michiyasunaga/BioLinkBERT-large --data data --out runs/linklarge_all2 --splits training,development --epochs 6 --batch 24 --length 384 --lr 0.00002 --seed 73 --eval-every 99 --bucket
  python3 -u code/ner.py predict --model michiyasunaga/BioLinkBERT-large --data data --out runs/linklarge_all2/test_eval --checkpoint runs/linklarge_all2/last --eval test --batch 24
fi

if [ "$(date -u +%s)" -lt "$(date -u -d '2026-10-14 11:20:00' +%s)" ]; then
  python3 -u code/ner.py train --data data --out runs/pubmed_all2 --splits training,development --epochs 6 --batch 24 --length 384 --lr 0.00002 --seed 79 --eval-every 99 --bucket
  python3 -u code/ner.py predict --data data --out runs/pubmed_all2/test_eval --checkpoint runs/pubmed_all2/last --eval test --batch 24
fi

if [ "$(date -u +%s)" -lt "$(date -u -d '2026-10-14 11:15:00' +%s)" ]; then
  python3 -u code/ner.py train --model dmis-lab/biobert-base-cased-v1.2 --data data --out runs/biobert_all2 --splits training,development --epochs 6 --batch 24 --length 384 --lr 0.00002 --seed 83 --eval-every 99 --bucket
  python3 -u code/ner.py predict --model dmis-lab/biobert-base-cased-v1.2 --data data --out runs/biobert_all2/test_eval --checkpoint runs/biobert_all2/last --eval test --batch 24
fi
date -u
echo EXTRA_PIPELINE_DONE
