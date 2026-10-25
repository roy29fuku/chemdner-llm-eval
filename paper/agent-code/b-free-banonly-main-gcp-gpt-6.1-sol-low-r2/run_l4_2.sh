#!/bin/bash
cd /home/agent
while pgrep -f '^bash run_l4.sh' >/dev/null; do sleep 10; done
python3 -u train_joint.py --model michiyasunaga/BioLinkBERT-large --batch 24 --length 384 --lr 2e-5 --epochs 7 --seed 46 --out largejointval > largejointval.log 2>&1
python3 -u train_joint.py --model dmis-lab/biobert-base-cased-v1.2 --batch 32 --length 384 --lr 3e-5 --epochs 6 --seed 47 --out casedval > casedval.log 2>&1
python3 -u train_v2.py --all --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --batch 48 --length 256 --lr 3e-5 --epochs 7 --seed 42 --out fulltextall > fulltextall.log 2>&1
python3 -u train_v2.py --all --model michiyasunaga/BioLinkBERT-large --batch 24 --length 384 --lr 2e-5 --epochs 7 --seed 43 --out largeall > largeall.log 2>&1
if [ -f use_cased ]; then
 python3 -u train_joint.py --all --model dmis-lab/biobert-base-cased-v1.2 --batch 32 --length 384 --lr 3e-5 --epochs 6 --seed 47 --out casedall > casedall.log 2>&1
fi
if [ ! -f skip_joint ]; then
python3 -u train_joint.py --all --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --batch 32 --length 384 --lr 3e-5 --epochs 7 --seed 44 --out jointall > jointall.log 2>&1
fi
python3 -u train_typed.py --all --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --batch 32 --length 384 --lr 3e-5 --epochs 7 --seed 45 --out typedall > typedall.log 2>&1
chem_epochs=7
if [ "$(date -u +%s)" -gt "$(date -u -d '2026-10-14 15:20:00' +%s)" ]; then chem_epochs=5; fi
python3 -u train_joint.py --all --model michiyasunaga/BioLinkBERT-large --batch 24 --length 384 --lr 2e-5 --epochs "$chem_epochs" --seed 46 --out largejointall > largejointall.log 2>&1
