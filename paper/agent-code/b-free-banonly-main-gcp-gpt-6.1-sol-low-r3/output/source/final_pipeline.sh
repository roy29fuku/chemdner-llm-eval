#!/bin/bash
set -e
export HF_HUB_DISABLE_XET=1
while kill -0 29297 2>/dev/null; do sleep 10; done
run_model () {
    name=$1
    seed=$2
    model=$3
    script=$4
    epochs=${5:-8}
    python3 "$script" --data data --out "$name" --full --epochs "$epochs" --seed "$seed" --model "$model" >"$name.log" 2>&1
    python3 "$script" --data data --out "pred_$name" --model "$model" --predict "$name/ep$((epochs-2))" "$name/ep$((epochs-1))" "$name/ep$epochs" >"pred_$name.log" 2>&1
}
python3 run.py --data data --out pred_full_ft42 --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --predict full_ft42/ep6 full_ft42/ep7 full_ft42/ep8 >pred_full_ft42.log 2>&1
run_model full_abs77 77 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract run.py
run_model full_bilou123 123 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext run_bilou.py
run_model full_ft2024 2024 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext run.py
run_model full_abs2025 2025 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract run.py
run_model full_ctx314 314 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext run_context.py 7
run_model full_ft991 991 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext run.py 6
echo PIPELINE_COMPLETE
