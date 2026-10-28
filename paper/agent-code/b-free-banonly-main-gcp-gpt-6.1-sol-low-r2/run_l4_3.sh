#!/bin/bash
cd /home/agent
while pgrep -f '^bash run_l4_2.sh' >/dev/null; do sleep 10; done
if [ "$(date -u +%s)" -lt "$(date -u -d '2026-10-14 15:32:00' +%s)" ]; then
 python3 -u train_joint.py --model dmis-lab/biobert-base-cased-v1.2 --batch 32 --length 384 --lr 3e-5 --epochs 6 --seed 47 --out casedval > casedval.log 2>&1
 if [ "$(date -u +%s)" -lt "$(date -u -d '2026-10-14 15:42:00' +%s)" ]; then
  python3 -u train_joint.py --all --model dmis-lab/biobert-base-cased-v1.2 --batch 32 --length 384 --lr 3e-5 --epochs 6 --seed 47 --out casedall > casedall.log 2>&1
 fi
fi
