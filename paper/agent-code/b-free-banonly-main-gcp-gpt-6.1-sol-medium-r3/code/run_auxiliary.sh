#!/bin/bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
task_root=/home/agent
task_code="$task_root/code"
while [[ ! -f "$task_root/full_chain.done" ]]; do sleep 10; done
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valaux" --model "$task_root/pretrained" --epochs 5 --batch 16 --seed 71 --dev-fraction .8 --snapshots --aux-types > "$task_root/valaux.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valaux/pred" --model "$task_root/valaux/best" --predict > "$task_root/valaux.predict.log" 2>&1
python3 "$task_code/average_weights.py" --models "$task_root/valaux/epoch3" "$task_root/valaux/epoch4" "$task_root/valaux/epoch5" --out "$task_root/valaux/avg" > "$task_root/valaux.average.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valaux/avg-pred" --model "$task_root/valaux/avg" --predict > "$task_root/valaux.avg-predict.log" 2>&1
python3 "$task_code/select_checkpoint.py" --data "$task_root/data" --pmids "$task_root/valaux/heldout_pmids.json" --models "$task_root/valaux/best" "$task_root/valaux/avg" --logits "$task_root/valaux/pred/development.logits.npz" "$task_root/valaux/avg-pred/development.logits.npz" --out "$task_root/valaux/selected.json" > "$task_root/valaux.selection.log" 2>&1
aux_model_path=$(python3 -c 'import json; print(json.load(open("/home/agent/valaux/selected.json"))["model"])')
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allaux" --model "$aux_model_path" --epochs 2 --batch 16 --lr 1e-5 --seed 71 --all > "$task_root/allaux.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allaux/pred" --model "$task_root/allaux/best" --predict --splits test > "$task_root/allaux.predict.log" 2>&1
python3 - <<'PY'
import json
p='/home/agent/final_models.json'
m=json.load(open(p));m['aux']='/home/agent/allaux/best';json.dump(m,open(p,'w'),indent=2)
PY
date -u > "$task_root/auxiliary.done"
