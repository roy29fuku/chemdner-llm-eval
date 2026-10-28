# Chemical mention recognition

All chemical supervision comes from the supplied training and development annotations. The base models are publicly available, pretrained biomedical language encoders, not chemical NER checkpoints. Their model cards and configs are preserved in `provenance/`.

`train.py` fine-tunes three-label BIO token classifiers. It tokenizes each title and abstract with character offsets, creates overlapping windows of at most 382 content tokens, and combines window logits with larger weights toward their centers. Training uses AdamW, linear learning-rate decay with 6% warmup, mixed precision, and gradient clipping. Length-grouped batches and normalization by a fixed expected token count reduce padding while preserving the token-level objective. Checkpoints are selected by development exact-span F1.

`constrained.py` decodes BIO sequences and prevents entity boundaries inside contiguous alphabetic words. Its accelerated implementation was checked against the original decoder on randomized inputs and on the complete first-model development predictions.

`combine.py` projects model probabilities to a common character-offset tokenization, combines the model scores, and tunes entity and start biases on development data. `grid.py` compares model combinations and weights; `begin_grid.py` tests start biases. The final models are retrained with both provided labeled sets.

`abbreviations.py` propagates abbreviations explicitly defined next to predicted chemical names. `filter.py` removes standalone terms explicitly excluded by the task guidelines. `repeat.py` tests repetition propagation with a training-derived reliability filter; it is used only when development F1 improves. `dictionary.py` did not improve the later validation models and is disabled. `stack.py` evaluates a small calibration classifier using two article-level development folds; the initial experiments did not beat the simple ensemble.

## Environment

GPU training: one NVIDIA T4, PyTorch 2.9.1+cu129, Transformers 4.57.3, Python 3.10. CPU analysis: Python 3.12, Transformers 5.19.0, NumPy, Numba, scikit-learn, pyahocorasick. Saved tokenizer files preserve the offset mappings.

## Training runs

Run 1: `train.py --out run1 --epochs 8 --seed 42 --random-batches --unscaled-loss --raw-decoder --unsorted-inference`

Run 2: `train.py --out run2 --epochs 8 --seed 2026 --unscaled-loss --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`

Runs 3–5 and all final training commands are in `rest.sh`. Run 3 uses BioLinkBERT-base; run 4 uses BioLinkBERT-large; run 5 is an independent seed of BiomedBERT abstract-fulltext. The final runs use `--full` to include development annotations. Run 1's default model is `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`.

The working directory for training must contain `data/`; the analysis scripts use `/workspace/data/`. `download.py` transfers only predictions, logits, tokenizer metadata, and logs from completed jobs. The required output is `/workspace/output/predictions.tsv`.

Validation artifacts and comparisons are in `/workspace/results/`. The format checker checks offsets and identifiers, not accuracy. Hidden test annotations were never accessed.

`balance.py` makes limited bracket, punctuation, and whitespace repairs to existing model spans. It improved three-model development F1 and is applied after abbreviation propagation and excluded-word filtering. Repetition propagation ceased to help the three-model ensemble and is disabled for that pipeline.

## Final output

After completed final-run artifacts have been downloaded, run `python3 /workspace/code/finalize.py`. It reads `/workspace/results/final_config.json`, combines the final model scores, applies the selected postprocessing, verifies a temporary prediction file, and atomically replaces the required output. The validation F1 refers to the validation ensemble before retraining; no hidden test F1 is available.
