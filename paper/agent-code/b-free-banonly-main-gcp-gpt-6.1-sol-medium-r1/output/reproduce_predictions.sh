#!/bin/bash
set -euo pipefail
task_output_dir="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$task_output_dir/grouped_scores"
for task_model_name in pubmed linklarge biobert roberta; do
  task_score_files=("$task_output_dir/${task_model_name}_all_test_scores.pkl")
  if [ -f "$task_output_dir/${task_model_name}_all2_test_scores.pkl" ]; then
    task_score_files+=("$task_output_dir/${task_model_name}_all2_test_scores.pkl")
  fi
  python3 "$task_output_dir/code/average_scores.py" "${task_score_files[@]}" --out "$task_output_dir/grouped_scores/${task_model_name}.pkl"
done
python3 "$task_output_dir/code/make_predictions.py" \
  "$task_output_dir/grouped_scores/pubmed.pkl" \
  "$task_output_dir/grouped_scores/linklarge.pkl" \
  "$task_output_dir/grouped_scores/biobert.pkl" \
  "$task_output_dir/grouped_scores/roberta.pkl" \
  --settings "$task_output_dir/final_calibration.json" \
  --lexicon "$task_output_dir/all_lexicon.json" \
  --boundary-map "$task_output_dir/final_boundary_map.json" \
  --out "$task_output_dir/predictions.tsv"
python3 /workspace/check_format.py "$task_output_dir/predictions.tsv"
