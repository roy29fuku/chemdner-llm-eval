# Predictions

Test-set predictions of every run scored in the paper. Each file holds character positions only; no text from the corpus.

## Format

One predicted mention per line, the same format as the BioCreative IV CHEMDNER CEM task:

```
PMID<TAB>section:start:end
```

- `section` is `T` (title) or `A` (abstract).
- `start` and `end` are 0-based character offsets within that section, end-exclusive (the mention is `text[start:end]`), as in the official CHEMDNER annotations.
- Duplicates count once. Lines are not ranked and carry no confidence.
- Every line of every file matches `^[0-9]+\t[TA]:[0-9]+:[0-9]+$` (checked for all files below).

## Layout

Paths mirror `runs/` of [roy29fuku/chemdner-llm-eval](https://github.com/roy29fuku/chemdner-llm-eval), so the directory can be copied into `runs/` and scored with the repository's scripts unchanged:

- `direct/<run>/predictions.tsv`: the LLM answers directly (one call per abstract).
- `agent/<run>/workspace/output/predictions.tsv`: the coding agent's submitted file (`/workspace/output/predictions.tsv`). The agent's code is in [`../agent-code/<run>/`](../agent-code/).

Each file is byte-identical to the one in the run records; its SHA-256 can be checked against [`../run-records-sha256.txt`](../run-records-sha256.txt) (same relative path). The 22 Experiment 15 files also match `frozen/exp15-predictions.sha256` of the code repository, committed before the test labels were used.

## Run names

- Direct runs: `evaluation-<model>-<effort>-r<rep>[-shots<K>-<random|similar>[-s<seed>]][-sub<list>]`. `r0`–`r2` are the three runs of 2026-10-14; `r10` is the no-example rerun of 2026-10-20. `shots<K>` is the number of labelled examples shown; `random` uses the same K examples for every abstract (example sets with seed 1, 2, 3), `similar` picks the K most similar examples per abstract (TF-IDF cosine). `-sub<list>` restricts the run to a frozen document list in `frozen/` (`official2478`: the 2,478 test abstracts that have gold mentions; `curve1000`: a 1,000-abstract subset of those). Runs without `-sub` cover all 3,000 test abstracts.
- Agent runs: `b-free-banonly-main[-gcp[-resume|-b128]]-<model>-<effort>-r<rep>`. `gcp`: the agent's container and proxy ran on a dedicated control VM on Google Cloud; no tag: the first main attempt, with all containers on one Mac. `gcp-resume`: rerun with automatic resumption after transient API errors. `gcp-b128`: the agent received only 128 labelled abstracts (the seed-1 example set of the 128-example random condition).

## Mapping to the paper's conditions

| Condition in the paper | Settings | Runs | Directories |
|---|---|---|---|
| Direct prompting, zero-shot (main comparison, Figure 1) | 6 settings × 3 runs | 18 | `evaluation-gpt-5.2-none-r0`<br>`evaluation-gpt-5.2-none-r1`<br>`evaluation-gpt-5.2-none-r2`<br>`evaluation-gpt-5.2-medium-r0`<br>`evaluation-gpt-5.2-medium-r1`<br>`evaluation-gpt-5.2-medium-r2`<br>`evaluation-gpt-6-luna-none-r0`<br>`evaluation-gpt-6-luna-none-r1`<br>`evaluation-gpt-6-luna-none-r2`<br>`evaluation-gpt-6-luna-medium-r0`<br>`evaluation-gpt-6-luna-medium-r1`<br>`evaluation-gpt-6-luna-medium-r2`<br>`evaluation-gpt-6.1-sol-low-r0`<br>`evaluation-gpt-6.1-sol-low-r1`<br>`evaluation-gpt-6.1-sol-low-r2`<br>`evaluation-gpt-6.1-sol-medium-r0`<br>`evaluation-gpt-6.1-sol-medium-r1`<br>`evaluation-gpt-6.1-sol-medium-r2` |
| Coding agent on GCP (main comparison, Figure 1) | 6 settings × 3 runs | 18 (17 files) | `b-free-banonly-main-gcp-gpt-5.2-none-r1`<br>`b-free-banonly-main-gcp-gpt-5.2-none-r2`<br>`b-free-banonly-main-gcp-gpt-5.2-none-r3`<br>`b-free-banonly-main-gcp-gpt-5.2-medium-r1`<br>`b-free-banonly-main-gcp-gpt-5.2-medium-r2` (no predictions; scored as empty)<br>`b-free-banonly-main-gcp-gpt-5.2-medium-r3`<br>`b-free-banonly-main-gcp-gpt-6-luna-none-r1`<br>`b-free-banonly-main-gcp-gpt-6-luna-none-r2`<br>`b-free-banonly-main-gcp-gpt-6-luna-none-r3`<br>`b-free-banonly-main-gcp-gpt-6-luna-medium-r1`<br>`b-free-banonly-main-gcp-gpt-6-luna-medium-r2`<br>`b-free-banonly-main-gcp-gpt-6-luna-medium-r3`<br>`b-free-banonly-main-gcp-gpt-6.1-sol-low-r1`<br>`b-free-banonly-main-gcp-gpt-6.1-sol-low-r2`<br>`b-free-banonly-main-gcp-gpt-6.1-sol-low-r3`<br>`b-free-banonly-main-gcp-gpt-6.1-sol-medium-r1`<br>`b-free-banonly-main-gcp-gpt-6.1-sol-medium-r3`<br>`b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2` |
| Coding agent on GCP, original GPT-6.1 sol medium r2 (stopped by an API capacity error; replaced by the `gcp-resume` run and not scored; reported for completeness) | GPT-6.1 sol medium | 1 | `b-free-banonly-main-gcp-gpt-6.1-sol-medium-r2` |
| Coding agent, first main attempt on one Mac (Supplementary Table S1) | 4 settings × 3 runs | 12 | `b-free-banonly-main-gpt-5.2-none-r1`<br>`b-free-banonly-main-gpt-5.2-none-r2`<br>`b-free-banonly-main-gpt-5.2-none-r3`<br>`b-free-banonly-main-gpt-5.2-medium-r1`<br>`b-free-banonly-main-gpt-5.2-medium-r2`<br>`b-free-banonly-main-gpt-5.2-medium-r3`<br>`b-free-banonly-main-gpt-6-luna-none-r1`<br>`b-free-banonly-main-gpt-6-luna-none-r2`<br>`b-free-banonly-main-gpt-6-luna-none-r3`<br>`b-free-banonly-main-gpt-6-luna-medium-r1`<br>`b-free-banonly-main-gpt-6-luna-medium-r2`<br>`b-free-banonly-main-gpt-6-luna-medium-r3` |
| Example-shown control: no-example rerun on the same day | 3 settings × 1 run | 3 | `evaluation-gpt-6-luna-medium-r10-subofficial2478`<br>`evaluation-gpt-6.1-sol-low-r10-subofficial2478`<br>`evaluation-gpt-6.1-sol-medium-r10-subofficial2478` |
| Example-shown control: 128 random examples | 3 settings × 3 example sets (seeds 1–3) | 9 | `evaluation-gpt-6-luna-medium-shots128-random-s1-r0-subofficial2478`<br>`evaluation-gpt-6-luna-medium-shots128-random-s2-r0-subofficial2478`<br>`evaluation-gpt-6-luna-medium-shots128-random-s3-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-low-shots128-random-s1-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-low-shots128-random-s2-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-low-shots128-random-s3-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-medium-shots128-random-s1-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-medium-shots128-random-s2-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-medium-shots128-random-s3-r0-subofficial2478` |
| Example-shown control: similar examples (K = 128 for GPT-6 luna medium, K = 32 for GPT-6.1 sol medium) | 2 settings × 1 run | 2 | `evaluation-gpt-6-luna-medium-shots128-similar-r0-subofficial2478`<br>`evaluation-gpt-6.1-sol-medium-shots32-similar-r0-subofficial2478` |
| Example-count curve (K = 8, 32, 256, 512, 1,024; K = 128 is the seed-1 128-example run rescored on the same 1,000 abstracts) | GPT-6 luna medium × 1 run each | 5 | `evaluation-gpt-6-luna-medium-shots8-random-s1-r0-subcurve1000`<br>`evaluation-gpt-6-luna-medium-shots32-random-s1-r0-subcurve1000`<br>`evaluation-gpt-6-luna-medium-shots256-random-s1-r0-subcurve1000`<br>`evaluation-gpt-6-luna-medium-shots512-random-s1-r0-subcurve1000`<br>`evaluation-gpt-6-luna-medium-shots1024-random-s1-r0-subcurve1000` |
| Coding agent with 128 labelled abstracts, b(128) | GPT-6 luna medium × 3 runs | 3 | `b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r1`<br>`b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r2`<br>`b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r3` |

Settings: GPT-5.2 (reasoning effort none, medium), GPT-6 luna (none, medium), GPT-6.1 sol (low, medium). The example-shown control and b(128) use the settings in which the agent outperformed direct prompting (GPT-6 luna medium, GPT-6.1 sol low, GPT-6.1 sol medium). 70 files in total; `b-free-banonly-main-gcp-gpt-5.2-medium-r2` ended without writing predictions and is scored as an empty prediction set, as specified in the analysis plan.

## Scoring

The scores in the paper come from two scripts of the code repository: `scripts/score_main.py` (direct prompting and the agent, main comparison and the first attempt) and `scripts/score_exp15.py` (example-shown control, curve and b(128)). Both read the test labels of the official 2013 CHEMDNER distribution, which is not redistributed here or in the code repository (no explicit license).

```sh
git clone https://github.com/roy29fuku/chemdner-llm-eval.git
cd chemdner-llm-eval
uv sync
uv run python -m chemdner_bench.corpus fetch --official <official-2013 directory> --with-test-gold
cp -R <this directory>/direct <this directory>/agent runs/
uv run python scripts/score_main.py --out runs/scores/main
uv run python scripts/score_exp15.py score --out runs/scores/exp15
```

`score_exp15.py score` first checks that the predictions match `frozen/exp15-predictions.sha256`.

To score a single file without the official distribution, use the public NCBI BioC version of the corpus (BioCreative VII NLM-Chem track). Its text is byte-identical to the official one, but one test abstract (PMID 23645677) lacks all its mentions, so 2,477 instead of 2,478 abstracts are scored and the values can differ slightly from the paper:

```sh
uv run python -m chemdner_bench.corpus fetch --ncbi
uv run python -m chemdner_bench.corpus gold --split evaluation --source ncbi --out gold.tsv
uv run python -m chemdner_bench.cem --gold gold.tsv --pred direct/evaluation-gpt-6-luna-medium-r0/predictions.tsv
```
