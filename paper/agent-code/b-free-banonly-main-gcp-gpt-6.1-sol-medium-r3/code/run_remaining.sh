#!/bin/bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
task_root=/home/agent
task_code="$task_root/code"
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valmixed" --model "$task_root/pretrained" --epochs 6 --batch 16 --seed 17 --dev-fraction .8 --snapshots --typed --case-features > "$task_root/valmixed.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valmixed/pred" --model "$task_root/valmixed/best" --predict > "$task_root/valmixed.predict.log" 2>&1
python3 "$task_code/average_weights.py" --models "$task_root/valmixed/epoch4" "$task_root/valmixed/epoch5" "$task_root/valmixed/epoch6" --out "$task_root/valmixed/avg" > "$task_root/valmixed.average.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/valmixed/avg-pred" --model "$task_root/valmixed/avg" --predict > "$task_root/valmixed.avg-predict.log" 2>&1
python3 "$task_code/select_checkpoint.py" --data "$task_root/data" --pmids "$task_root/valmixed/heldout_pmids.json" --models "$task_root/valmixed/best" "$task_root/valmixed/avg" --logits "$task_root/valmixed/pred/development.logits.npz" "$task_root/valmixed/avg-pred/development.logits.npz" --out "$task_root/valmixed/selected.json" > "$task_root/valmixed.selection.log" 2>&1

mixed_model_path=$(python3 -c 'import json; print(json.load(open("/home/agent/valmixed/selected.json"))["model"])')
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allmixed" --model "$mixed_model_path" --epochs 2 --batch 16 --lr 1e-5 --seed 17 --all > "$task_root/allmixed.log" 2>&1
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allmixed/pred" --model "$task_root/allmixed/best" --predict --splits test > "$task_root/allmixed.predict.log" 2>&1

python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allbase" --model "$task_root/pretrained" --epochs 6 --batch 16 --seed 42 --all --snapshots > "$task_root/allbase.log" 2>&1
final_base_path="$task_root/allbase/best"
if [[ "$mixed_model_path" == */avg ]]; then
    python3 "$task_code/average_weights.py" --models "$task_root/allbase/epoch4" "$task_root/allbase/epoch5" "$task_root/allbase/epoch6" --out "$task_root/allbase/avg" > "$task_root/allbase.average.log" 2>&1
    final_base_path="$task_root/allbase/avg"
fi
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allbase/pred" --model "$final_base_path" --predict --splits test > "$task_root/allbase.predict.log" 2>&1

python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allabstract" --model "$task_root/pretrained-abstract" --epochs 6 --batch 16 --seed 43 --all --snapshots > "$task_root/allabstract.log" 2>&1
abstract_model_path=$(python3 -c 'import json; print(json.load(open("/home/agent/valabstract/selected.json"))["model"])')
final_abstract_path="$task_root/allabstract/best"
if [[ "$abstract_model_path" == */avg ]]; then
    python3 "$task_code/average_weights.py" --models "$task_root/allabstract/epoch4" "$task_root/allabstract/epoch5" "$task_root/allabstract/epoch6" --out "$task_root/allabstract/avg" > "$task_root/allabstract.average.log" 2>&1
    final_abstract_path="$task_root/allabstract/avg"
fi
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/allabstract/pred" --model "$final_abstract_path" --predict --splits test > "$task_root/allabstract.predict.log" 2>&1

if ! python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/alllarge" --model "$task_root/biolinklarge" --epochs 5 --batch 16 --lr 2e-5 --all --snapshots > "$task_root/alllarge.log" 2>&1; then
    mv "$task_root/alllarge.log" "$task_root/alllarge.batch16.failed.log"
    python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/alllarge" --model "$task_root/biolinklarge" --epochs 5 --batch 8 --accum 2 --lr 2e-5 --all --snapshots > "$task_root/alllarge.log" 2>&1
fi
large_model_path=$(python3 -c 'import json; print(json.load(open("/home/agent/vallarge/selected.json"))["model"])')
final_large_path="$task_root/alllarge/best"
if [[ "$large_model_path" == */avg ]]; then
    python3 "$task_code/average_weights.py" --models "$task_root/alllarge/epoch3" "$task_root/alllarge/epoch4" "$task_root/alllarge/epoch5" --out "$task_root/alllarge/avg" > "$task_root/alllarge.average.log" 2>&1
    final_large_path="$task_root/alllarge/avg"
fi
python3 -u "$task_code/ner.py" --data "$task_root/data" --out "$task_root/alllarge/pred" --model "$final_large_path" --predict --splits test > "$task_root/alllarge.predict.log" 2>&1
python3 - "$final_base_path" "$final_abstract_path" "$final_large_path" "$task_root/allmixed/best" <<'PY' > "$task_root/final_models.json"
import sys,json
json.dump(dict(zip(['base','abstract','large','mixed'],sys.argv[1:])),sys.stdout,indent=2)
PY
date -u > "$task_root/full_chain.done"
