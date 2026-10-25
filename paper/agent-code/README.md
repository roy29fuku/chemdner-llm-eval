# Code written by the coding agent

For each coding-agent run used in the paper, this directory holds the source files that the agent (OpenAI Codex CLI) wrote in its working directory `/workspace`. Each subdirectory is named after the run (the same name as in `runs/agent/` of this repository and in the scoring scripts); paths below it are relative to the agent's `/workspace`. The predictions of these runs are in [`../predictions/agent/`](../predictions/agent/).

## What is included

Files with the extensions `.py .sh .md .txt .json .yaml .yml .toml .cfg .ini .ipynb` and at most 200 KB, except those listed under "What is excluded". Besides code, this keeps the agent's own notes (for example `STATE.md`) and small configuration or result files (ensemble weights, validation metrics, held-out PMID lists).

The code refers to files that are not included: the corpus copies under `/workspace/data/` (rebuild them with `python -m chemdner_bench.corpus fetch`; the corpus is not redistributed), trained models, and cloud VMs and buckets that were deleted at the end of each run. Some notes and scripts contain the external IP address of a VM or the name of the run's isolated GCP project; those resources no longer exist.

## What is excluded

| Rule | Reason |
|---|---|
| `task.md`, `guidelines.md`, `check_format.py` at the top of `/workspace` | Provided by the harness, not written by the agent (SHA-256 below) |
| Anything under `output/` except code files (`.py`, `.sh`), and anything under `data/`, `models/`, `checkpoints/` | Predictions, corpus copies and models |
| Model weights and binaries (`.pt .bin .safetensors .pkl .crfsuite .npz .npy .joblib` etc.) and data files (`.tsv .csv .jsonl`) | Not source code; may encode corpus content |
| Saved or downloaded model directories (any directory holding a tokenizer, a vocabulary or `config.json` with weights), and Hugging Face tokenizer/config files | Model artifacts, not agent-written |
| `code/provenance/` (one run) | Model cards and configs the agent downloaded from Hugging Face to check how the models were pretrained; third-party content |
| Files whose names suggest credentials (`sshkeys*`, `ssh-*`) | Not opened and not copied |
| Files that contain corpus text or annotation-derived strings | The CHEMDNER corpus is not redistributed (see the check below) |
| Other extensions (`.log .pyc .dev .filtered` ...) and files over 200 KB | Logs, caches and intermediate outputs |

Corpus check: every included file was tested for any substring of 60 characters or more that occurs in a CHEMDNER abstract (title or body, also in lower case), and for lists of gold mention strings (50 or more lines each equal to a gold mention string). Files containing many gold mention strings as delimited tokens (whole lines, tab- or comma-separated fields, quoted strings) were also reviewed by hand. Three files were excluded (marked below): a per-mention lexicon with corpus text, an error-analysis list of mention strings, and a table of corpus tokens with their gold sub-spans. One file was reviewed and kept: `b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r3/suffix_audit.py` contains 90 strings that also occur as gold mentions, but they are a hand-written list of chemical suffixes and class names (for example amino acids and functional groups), not taken from the annotations.

## Harness files (not included)

- `guidelines.md` (the task definition, same for every run): `c5c1f062c01b491787c486d13a8de977f5716793eca7218079db5ef2de31875f`
- `check_format.py` (format checker, same for every run): `994596da47a65b7fac7daedaca39546b8adee0de1c9e5602691e483f6c1c8216`
- `task.md` (the task statement, rendered from `agent/task/task.md` of this repository) differs between runs only in the line naming the run's GCP project and service accounts and, for the first attempt on a Mac, the GPU limit ("NVIDIA L4 only" instead of "NVIDIA L4 or T4 only"). Its SHA-256 is listed per run below.

These files are also in the run records, whose SHA-256 list is [`../run-records-sha256.txt`](../run-records-sha256.txt) (path `agent/<run>/workspace/<file>`).

## Runs

| Condition | Runs | Files included |
|---|---|---|
| Coding agent on GCP (main comparison; 6 settings x 3 runs, scored) | 18 | 235 |
| Coding agent on GCP, original GPT-6.1 sol medium r2 (stopped by an API capacity error; replaced by the `gcp-resume` run, not scored) | 1 | 30 |
| First main attempt, agent containers on one Mac (Supplementary Table S1; 4 settings x 3 runs, scored) | 12 | 35 |
| Coding agent given only 128 labelled abstracts, b(128) (GPT-6 luna medium x 3 runs, scored) | 3 | 27 |

### Coding agent on GCP (main comparison; 6 settings x 3 runs, scored)


## Code under output/ (added in a second pass)

The first pass excluded everything under `output/`. Some agents kept their code there, so code files (`.py`, `.sh`, ≤ 200 KB) under `output/` were added afterwards, after the same corpus check (no substring of 60 characters or more found in the CHEMDNER text) and secret check (none found). The per-run tables below were written in the first pass and still list these files under "under output/". 41 files were added:

| Run | File |
|---|---|
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/align_scores.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/eval.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/train.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/train_joint.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/train_typed.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/train_v2.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/tune_ensemble.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/code/viterbi.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r2` | `output/reproduce.sh` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/ensemble_lexword.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/final_pipeline.sh` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/post.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/run.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/run_bilou.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-low-r3` | `output/source/run_context.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/average_checkpoints.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/average_scores.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/calibrate.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/ensemble.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/lexicon.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/make_predictions.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/ner.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/code/postprocess.py` |
| `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1` | `output/reproduce_predictions.sh` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/average.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/boundaries.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/calibrate.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/ensemble.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/evaluate.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/finalize.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/lexicon.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/mutations.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/ner.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/postprocess.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/select_average.py` |
| `b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` | `output/code/viterbi.py` |
| `b-free-banonly-main-gpt-6-luna-medium-r3` | `output/bert_ner.py` |
| `b-free-banonly-main-gpt-6-luna-medium-r3` | `output/ensemble_eval.py` |
| `b-free-banonly-main-gpt-6-luna-medium-r3` | `output/lex_eval.py` |
| `b-free-banonly-main-gpt-6-luna-medium-r3` | `output/predict_bert.py` |
| `b-free-banonly-main-gpt-6-luna-medium-r3` | `output/solve.py` |

#### b-free-banonly-main-gcp-gpt-5.2-none-r1

Files included: 1. `task.md` SHA-256: `fe07bf1014fd56e7fa849b16278305d2acd1d503f7b95ab0c929c39adc1cd3c3`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-5.2-none-r2

Files included: 1. `task.md` SHA-256: `fe07bf1014fd56e7fa849b16278305d2acd1d503f7b95ab0c929c39adc1cd3c3`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-5.2-none-r3

Files included: 1. `task.md` SHA-256: `fe07bf1014fd56e7fa849b16278305d2acd1d503f7b95ab0c929c39adc1cd3c3`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 3 | `output/` (3 files) |
| bytecode cache | 1 | `__pycache__/chem_tagger.cpython-312.pyc` |

#### b-free-banonly-main-gcp-gpt-5.2-medium-r1

Files included: 1. `task.md` SHA-256: `cc1d52e7eb03d08d53155cc52a7a883dee35cbecf7cc3637642fb9bdb64b0040`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-5.2-medium-r2

Files included: 1. `task.md` SHA-256: `cc1d52e7eb03d08d53155cc52a7a883dee35cbecf7cc3637642fb9bdb64b0040`.

| Excluded because | Files | Paths |
|---|---|---|
| bytecode cache | 1 | `__pycache__/chemner_bert.cpython-312.pyc` |

#### b-free-banonly-main-gcp-gpt-5.2-medium-r3

Files included: 1. `task.md` SHA-256: `cc1d52e7eb03d08d53155cc52a7a883dee35cbecf7cc3637642fb9bdb64b0040`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-6-luna-none-r1

Files included: 8. `task.md` SHA-256: `906a0ad00ac2c261c1d26893ad145cea2d4d53fb33e5311f03728e6f4950a8bd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 3 | `__pycache__/` (3 files) |

#### b-free-banonly-main-gcp-gpt-6-luna-none-r2

Files included: 12. `task.md` SHA-256: `906a0ad00ac2c261c1d26893ad145cea2d4d53fb33e5311f03728e6f4950a8bd`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 1 | `dictpat.pkl` |
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 2 | `__pycache__/` (2 files) |

#### b-free-banonly-main-gcp-gpt-6-luna-none-r3

Files included: 3. `task.md` SHA-256: `906a0ad00ac2c261c1d26893ad145cea2d4d53fb33e5311f03728e6f4950a8bd`.

| Excluded because | Files | Paths |
|---|---|---|
| extension not in allowlist (none) | 1 | `core` |
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-6-luna-medium-r1

Files included: 3. `task.md` SHA-256: `203df2ad8873928ad0920931d3ce78d247ecd9c799631d78c10c8ff0151db9b7`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 1 | `model.crfsuite` |
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/ner.cpython-312.pyc` |

#### b-free-banonly-main-gcp-gpt-6-luna-medium-r2

Files included: 5. `task.md` SHA-256: `203df2ad8873928ad0920931d3ce78d247ecd9c799631d78c10c8ff0151db9b7`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 3 | `__pycache__/` (3 files) |

#### b-free-banonly-main-gcp-gpt-6-luna-medium-r3

Files included: 4. `task.md` SHA-256: `203df2ad8873928ad0920931d3ce78d247ecd9c799631d78c10c8ff0151db9b7`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |

#### b-free-banonly-main-gcp-gpt-6.1-sol-low-r1

Files included: 34. `task.md` SHA-256: `4af42bb91bfeb6d57276867bde865066a2f8f4988bf4540487055864a78b42ea`.

| Excluded because | Files | Paths |
|---|---|---|
| model/tokenizer artifact (HF save or download), not agent-written | 33 | `code/tokenizer/tokenizer_config.json`, `full1/` (3 files), `full2/` (3 files), `full3/` (3 files), `full4/` (3 files), `full5/` (3 files), `run1/` (3 files), `run2/` (3 files), `run3/` (3 files), `run4/` (3 files), `run5/` (3 files), `stack2_prob/tokenizer_config.json`, `stack2_ratio/tokenizer_config.json` |
| model binary/weights | 36 | `full1/` (2 files), `full2/` (2 files), `full3/` (2 files), `full4/` (2 files), `full5/` (2 files), `run1/` (4 files), `run2/` (4 files), `run3/` (4 files), `run4/` (4 files), `run5/` (4 files), `stack2_prob/` (3 files), `stack2_ratio/` (3 files) |
| inside a saved/downloaded model directory | 62 | `code/tokenizer/tokenizer.json`, `full1/` (4 files), `full2/` (4 files), `full3/` (4 files), `full4/` (4 files), `full5/` (4 files), `run1/` (21 files), `run2/` (4 files), `run3/` (4 files), `run4/` (4 files), `run5/` (4 files), `stack2_prob/` (2 files), `stack2_ratio/` (2 files) |
| data file (.tsv) | 21 | `full1/test.tsv`, `full2/test.tsv`, `full3/test.tsv`, `full4/test.tsv`, `full5/test.tsv`, `results/` (6 files), `run1/` (2 files), `run2/` (2 files), `run3/` (2 files), `run4/` (2 files), `run5/` (2 files) |
| under output/ | 1 | `output/predictions.tsv` |
| extension not in allowlist (.log) | 138 | `code/download.log`, `results/` (137 files) |
| bytecode cache | 16 | `code/__pycache__/` (16 files) |
| downloaded third-party model card/config (not agent-written) | 8 | `code/provenance/` (8 files) |
| extension not in allowlist (.dev) | 129 | `results/` (128 files), `results/early_output/predictions.tsv.dev` |
| extension not in allowlist (.filtered) | 12 | `results/` (10 files), `results/early_output/` (2 files) |
| extension not in allowlist (.abbr) | 12 | `results/` (10 files), `results/early_output/` (2 files) |
| extension not in allowlist (.repeat) | 6 | `results/` (4 files), `results/early_output/` (2 files) |
| extension not in allowlist (.balanced) | 8 | `results/` (8 files) |
| extension not in allowlist (.atoms) | 1 | `results/best4begin_2.tsv.dev.abbr.filtered.balanced.atoms` |

#### b-free-banonly-main-gcp-gpt-6.1-sol-low-r2

Files included: 33. `task.md` SHA-256: `cc1d52e7eb03d08d53155cc52a7a883dee35cbecf7cc3637642fb9bdb64b0040`.

| Excluded because | Files | Paths |
|---|---|---|
| credential-like name (not opened) | 1 | `sshkeys.txt` |
| extension not in allowlist (.log) | 31 | `experiments/casedall/mean.log`, `experiments/casedfixedval/grid456.log`, `experiments/` (13 files), `experiments/fulltextall/mean.log`, `experiments/jointval/grid567.log`, `experiments/largeall/mean.log`, `experiments/largejointval/grid567.log`, `experiments/largeval/` (3 files), `experiments/run1/` (5 files), `experiments/typedval/` (4 files) |
| data file (.tsv) | 21 | `experiments/casedfixedval/dev456.tsv`, `experiments/` (6 files), `experiments/jointval/dev567.tsv`, `experiments/largejointval/dev567.tsv`, `experiments/largeval/` (3 files), `experiments/run1/` (6 files), `experiments/typedval/` (3 files) |
| extension not in allowlist (.errors) | 20 | `experiments/casedfixedval/dev456.tsv.errors`, `experiments/` (5 files), `experiments/jointval/dev567.tsv.errors`, `experiments/largejointval/dev567.tsv.errors`, `experiments/largeval/` (3 files), `experiments/run1/` (6 files), `experiments/typedval/` (3 files) |
| model binary/weights | 53 | `experiments/casedall/` (4 files), `experiments/casedfixedval/` (5 files), `experiments/fulltextall/` (5 files), `experiments/jointval/` (4 files), `experiments/largeall/` (3 files), `experiments/largejointval/` (4 files), `experiments/largeval/` (9 files), `experiments/run1/` (11 files), `experiments/` (2 files), `experiments/typedval/` (6 files) |
| under output/ | 24 | `output/` (4 files), `output/code/__pycache__/` (8 files), `output/code/` (8 files), `output/scores/` (4 files) |
| bytecode cache | 20 | `__pycache__/` (20 files) |

#### b-free-banonly-main-gcp-gpt-6.1-sol-low-r3

Files included: 40. `task.md` SHA-256: `906a0ad00ac2c261c1d26893ad145cea2d4d53fb33e5311f03728e6f4950a8bd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 25 | `output/` (4 files), `output/models/` (7 files), `output/probabilities/` (7 files), `output/source/` (7 files) |
| extension not in allowlist (.log) | 28 | `work/` (28 files) |
| model binary/weights | 22 | `work/` (15 files), `work/test_probs/` (7 files) |
| larger than 200 KB | 2 | `work/` (2 files) |
| corpus check: contains >=60-char abstract substrings and ~2,000 gold mention strings (per-mention lexicon statistics) | 1 | `work/lex_stats.json` |
| bytecode cache | 5 | `work/__pycache__/` (5 files) |

#### b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1

Files included: 10. `task.md` SHA-256: `81aeb784396c7dea087223899a98136cb0a8107cb756ab4b82f30760f2b68c9f`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 116 | `output/` (48 files), `output/code/__pycache__/` (6 files), `output/code/` (8 files), `output/grouped_scores/` (4 files), `output/models/biobert/` (7 files), `output/models/linklarge/` (7 files), `output/models/linklarge2/` (7 files), `output/models/pubmed/` (7 files), `output/models/pubmed2/` (7 files), `output/models/roberta/` (8 files), `output/training_logs/` (7 files) |
| credential-like name (not opened) | 1 | `code/sshkeys` |
| model binary/weights | 2 | `code/` (2 files) |
| bytecode cache | 10 | `code/__pycache__/` (10 files) |
| under models/ | 18 | `models/link_tokenizer/` (4 files), `models/roberta_split_tokenizer/` (5 files), `models/roberta_tokenizer/` (5 files), `models/tokenizer/` (4 files) |

#### b-free-banonly-main-gcp-gpt-6.1-sol-medium-r3

Files included: 47. `task.md` SHA-256: `fe07bf1014fd56e7fa849b16278305d2acd1d503f7b95ab0c929c39adc1cd3c3`.

| Excluded because | Files | Paths |
|---|---|---|
| model/tokenizer artifact (HF save or download), not agent-written | 11 | `biolinklarge/` (3 files), `code/tokenizer-large/` (2 files), `code/tokenizer/` (2 files), `pretrained-abstract/` (2 files), `pretrained/` (2 files) |
| inside a saved/downloaded model directory | 39 | `biolinklarge/.cache/huggingface/.gitignore`, `biolinklarge/.cache/huggingface/download/` (12 files), `biolinklarge/` (2 files), `code/tokenizer-large/` (2 files), `code/tokenizer/` (2 files), `pretrained-abstract/.cache/huggingface/.gitignore`, `pretrained-abstract/.cache/huggingface/download/` (8 files), `pretrained-abstract/vocab.txt`, `pretrained/.cache/huggingface/.gitignore`, `pretrained/.cache/huggingface/download/` (8 files), `pretrained/vocab.txt` |
| model binary/weights | 26 | `biolinklarge/pytorch_model.bin`, `intermediate/` (3 files), `intermediate/all/` (5 files), `pretrained-abstract/pytorch_model.bin`, `pretrained/pytorch_model.bin`, `validation/abstract/development.logits.npz`, `validation/aux/development.logits.npz`, `validation/base/development.logits.npz`, `validation/early/development.logits.npz`, `validation/` (2 files), `validation/five/development.logits.npz`, `validation/four/development.logits.npz`, `validation/large-early/development.logits.npz`, `validation/large/development.logits.npz`, `validation/mixed-early/development.logits.npz`, `validation/mixed/development.logits.npz`, `validation/three/development.logits.npz`, `validation/two-early/development.logits.npz`, `validation/two/development.logits.npz` |
| under output/ | 46 | `output/` (6 files), `output/logs/` (10 files), `output/models/abstract/` (6 files), `output/models/aux/` (6 files), `output/models/base/` (6 files), `output/models/large/` (6 files), `output/models/mixed/` (6 files) |
| bytecode cache | 29 | `code/__pycache__/` (29 files) |
| extension not in allowlist (.log) | 24 | `intermediate/` (4 files), `validation/base/` (5 files), `validation/early/` (4 files), `validation/` (10 files), `validation/two-early/enzyme.log` |
| data file (.tsv) | 26 | `intermediate/all-five/test.tsv`, `intermediate/all-four/test.tsv`, `intermediate/base-submission/test.tsv`, `intermediate/three-submission/test.tsv`, `intermediate/two-submission/test.tsv`, `validation/base/` (4 files), `validation/base/refined/development.tsv`, `validation/early/consistency/development.tsv`, `validation/early/` (4 files), `validation/epoch4-consistency/development.tsv`, `validation/epoch4-decoder/development.tsv`, `validation/` (2 files), `validation/epoch4-smooth-decoder/development.tsv`, `validation/five/development.tsv`, `validation/four/development.tsv`, `validation/three/development.tsv`, `validation/two-early/` (2 files), `validation/two/development.tsv` |
| larger than 200 KB | 2 | `validation/` (2 files) |
| error analysis listing predicted/gold mention strings (annotation-derived) | 1 | `validation/early/errors.json` |

#### b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2

Files included: 30. `task.md` SHA-256: `81aeb784396c7dea087223899a98136cb0a8107cb756ab4b82f30760f2b68c9f`.

| Excluded because | Files | Paths |
|---|---|---|
| credential-like name (not opened) | 1 | `ssh-metadata` |
| model/tokenizer artifact (HF save or download), not agent-written | 5 | `results/validation/average/config.json`, `tokenizer/` (2 files), `tokenizer_abstract/` (2 files) |
| inside a saved/downloaded model directory | 4 | `tokenizer/` (2 files), `tokenizer_abstract/` (2 files) |
| under output/ | 24 | `output/code/` (13 files), `output/` (7 files), `output/training_history/` (4 files) |
| model binary/weights | 23 | `code/` (2 files), `results/full1/average/test_scores.npz`, `results/full1/test_scores.npz`, `results/full2/test_scores.npz`, `results/validation/average/` (2 files), `results/validation/` (8 files), `results/validation2/average/` (2 files), `results/validation2/` (6 files) |
| larger than 200 KB | 1 | `code/negative_training.json` |
| bytecode cache | 17 | `code/__pycache__/` (17 files) |
| data file (.tsv) | 24 | `results/` (7 files), `results/validation/average/processed_dev.tsv`, `results/validation/` (16 files) |
| extension not in allowlist (.log) | 31 | `results/` (10 files), `results/full1/interim_finalize.log`, `results/validation/average/` (2 files), `results/validation/` (18 files) |
| table of corpus tokens and their gold sub-spans learned from training annotations (annotation-derived) | 1 | `results/validation/boundary_rules.json` |

### Coding agent on GCP, original GPT-6.1 sol medium r2 (stopped by an API capacity error; replaced by the `gcp-resume` run, not scored)

#### b-free-banonly-main-gcp-gpt-6.1-sol-medium-r2

Files included: 30. `task.md` SHA-256: `81aeb784396c7dea087223899a98136cb0a8107cb756ab4b82f30760f2b68c9f`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 4 | `artifacts/` (4 files) |
| extension not in allowlist (.log) | 3 | `artifacts/` (3 files) |
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 9 | `code/__pycache__/` (9 files) |

### First main attempt, agent containers on one Mac (Supplementary Table S1; 4 settings x 3 runs, scored)

#### b-free-banonly-main-gpt-5.2-none-r1

Files included: 2. `task.md` SHA-256: `086ea970a8a509b864f0b4f821bca6c9a949e74ff319baad020bc7d8f446e4fd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/chem_ner.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-5.2-none-r2

Files included: 1. `task.md` SHA-256: `086ea970a8a509b864f0b4f821bca6c9a949e74ff319baad020bc7d8f446e4fd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-5.2-none-r3

Files included: 1. `task.md` SHA-256: `086ea970a8a509b864f0b4f821bca6c9a949e74ff319baad020bc7d8f446e4fd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-5.2-medium-r1

Files included: 1. `task.md` SHA-256: `6aa1923071e3663918759bab910023a25d2be12a7572e9f2c629e021c3c9163e`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 1 | `tmp.crfsuite` |
| under output/ | 2 | `output/` (2 files) |
| bytecode cache | 1 | `__pycache__/run_chemner.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-5.2-medium-r2

Files included: 1. `task.md` SHA-256: `6aa1923071e3663918759bab910023a25d2be12a7572e9f2c629e021c3c9163e`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/make_predictions.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-5.2-medium-r3

Files included: 1. `task.md` SHA-256: `6aa1923071e3663918759bab910023a25d2be12a7572e9f2c629e021c3c9163e`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-none-r1

Files included: 1. `task.md` SHA-256: `723aa7975aa286b77ad42b0b9e883bd871a758cf8988a8399a2e4c391ec1bc16`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/solve.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-none-r2

Files included: 9. `task.md` SHA-256: `723aa7975aa286b77ad42b0b9e883bd871a758cf8988a8399a2e4c391ec1bc16`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 2 | `__pycache__/` (2 files) |
| model/tokenizer artifact (HF save or download), not agent-written | 1 | `model/config.json` |
| inside a saved/downloaded model directory | 1 | `model/vocab.txt` |
| model binary/weights | 2 | `model/` (2 files) |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-none-r3

Files included: 2. `task.md` SHA-256: `723aa7975aa286b77ad42b0b9e883bd871a758cf8988a8399a2e4c391ec1bc16`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/solve.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-medium-r1

Files included: 5. `task.md` SHA-256: `31f3a61770518a9268d5bd3a220bf84e53049c98590ba54f9833c0656b6f2a50`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 12 | `output/` (12 files) |
| bytecode cache | 1 | `__pycache__/ner.cpython-312.pyc` |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-medium-r2

Files included: 11. `task.md` SHA-256: `31f3a61770518a9268d5bd3a220bf84e53049c98590ba54f9833c0656b6f2a50`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 3 | `lexclf.pkl`, `lexclf_alias.pkl`, `lexclf_full.pkl` |
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 8 | `__pycache__/` (8 files) |
| under data/ | 5 | `data/` (5 files) |

#### b-free-banonly-main-gpt-6-luna-medium-r3

Files included: 0. `task.md` SHA-256: `31f3a61770518a9268d5bd3a220bf84e53049c98590ba54f9833c0656b6f2a50`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 11 | `output/__pycache__/` (3 files), `output/` (8 files) |
| under data/ | 5 | `data/` (5 files) |

### Coding agent given only 128 labelled abstracts, b(128) (GPT-6 luna medium x 3 runs, scored)

#### b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r1

Files included: 9. `task.md` SHA-256: `fe07bf1014fd56e7fa849b16278305d2acd1d503f7b95ab0c929c39adc1cd3c3`.

| Excluded because | Files | Paths |
|---|---|---|
| model binary/weights | 4 | `chebi_lexicon.pkl`, `dev_bert_probs.pkl`, `oof_meta.pkl`, `oof_meta_partial.pkl` |
| extension not in allowlist (.gz) | 1 | `chebi.obo.gz` |
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 5 | `__pycache__/` (5 files) |

#### b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r2

Files included: 4. `task.md` SHA-256: `cc1d52e7eb03d08d53155cc52a7a883dee35cbecf7cc3637642fb9bdb64b0040`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 1 | `__pycache__/model.cpython-312.pyc` |

#### b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r3

Files included: 14. `task.md` SHA-256: `906a0ad00ac2c261c1d26893ad145cea2d4d53fb33e5311f03728e6f4950a8bd`.

| Excluded because | Files | Paths |
|---|---|---|
| under output/ | 1 | `output/predictions.tsv` |
| bytecode cache | 2 | `__pycache__/` (2 files) |

The complete list of excluded files (run, path, bytes, reason) is in [`excluded-files.txt`](excluded-files.txt).
