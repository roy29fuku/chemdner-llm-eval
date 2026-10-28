Task: chemical mention recognition; /workspace/task.md and guidelines.md read. Must write output/predictions.tsv, code-generated, no external chemical-annotated models or tools, no LLM API. Deadline approximately 2026-10-14 15:56 UTC (4h from 11:56). Current clock ~13:25 UTC.

Cloud: exactly ONE VM, chem-l4 in sa-agent-ws2/us-central1-a, g2-standard-8 (8 vCPU, ONE L4). IP 34.70.181.243. Boot disk chem-train retained from former T4 VM (former VM deleted before L4 creation). SSH/scp:
ssh -o 'ProxyCommand=nc -X connect -x proxy:8080 %h %p' -i /home/agent/.ssh/google_compute_engine agent@34.70.181.243 'COMMAND'
VM work directory /home/agent. All data/code there. No subagents permitted or used.

Current output already valid: 25,262 predictions, 2,465 articles, ZERO format errors. It is an initial ensemble: run1 best epoch7 (fulltext PubMedBERT) projected to canonical tokenizer + largeval best epoch7, weights [1,2], B bias -.5, I bias .5. Exact corresponding development F1 .9287596288. Both trained only training set. Need replace with full-data models once completed.

Models allowed: general unsupervised pretraining only, verified primary model cards / original repo:
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (30522 vocab)
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (28895 vocab)
- michiyasunaga/BioLinkBERT-large (28895 vocab, matches abstract tokenizer)
- dmis-lab/biobert-base-cased-v1.2 (original repo specifies raw PubMed pretraining).
No chemical-annotation-trained starting models.

Training scripts /workspace/train_v2.py (independent T/A windows), train_joint.py (title + abstract context), train_typed.py (joint, 17 labels BIO x 8 mention classes). torch BF16 on L4, AdamW, 8% warmup, 7 epochs usually. Typed infers raw17, saves devraw_epochX.npz / testraw_epochX.npz and folded3 outputs. All-mode saves model only last epoch; saves test predictions/logits for last THREE epochs plus final. Each script auto runs test after training. Output folders named below, logs at VM /home/agent/NAME.log.

Pipeline (one GPU sequential):
run_l4.sh: largeval (finished), jointval (finished), typedval (finished/finishing).
run_l4_2.sh waits run_l4.sh, then:
1 largejointval --epochs7 --seed46 --batch24 --length384 --lr2e-5 (currently running)
2 casedval --epochs6 --seed47 --batch32 --length384 --lr3e-5
3 fulltextall --all --epochs7 --seed42 --batch48 --length256 --lr3e-5
4 largeall --all --epochs7 --seed43 --batch24 --length384 --lr2e-5
5 casedall ONLY if /home/agent/use_cased exists (default absent)
6 jointall ONLY if /home/agent/skip_joint does NOT exist (default absent)
7 typedall --all --epochs7 --seed45 --batch32 --length384 --lr3e-5
8 largejointall --all --seed46 --batch24 --length384 --lr2e-5, epochs7 unless started after15:20 then epochs5.
Old run_l4_3.sh watcher was killed (no longer running). Decide flags based on validation ensemble before largeall finishes (~14:45). If weak joint contributes no F1, create skip_joint to save ~10min. If cased improves ensemble, create use_cased; costs ~12min. Always keep deadline in mind; partial/current output already safe.

Code helpers:
- eval.py: read data, average .npz arr_0..arr_N, decode Viterbi and exact span scoring; --split test writes predictions with --out. --model controls tokenizer, IMPORTANT. Canonical model is abstract or BioLinkBERT-large (same tokenizer). Supports --weights list, --bias (B), --ibias (I), --grid, --title-grid (no gain so far), --save-scores. Raw17 inputs fold to3 unless --typed; --posterior computes constrained marginal. --type-bias 8 numbers shifts both BIO states per class. Strict token count checks.
- align_scores.py: projects noncanonical token logits onto canonical abstract tokenizer. Needed for fulltext and cased. --source-model ... --target-model ... --split test/development --scores input --out output.
- tune_ensemble.py: positive coordinate weights and B/I biases, development F1 objective. Anchor first model weight1; optimize other weights [0,.5,1,1.5,2,3], two rounds, wide B/I grid. Run with OPENBLAS_NUM_THREADS=4. Saves JSON weights,bias,ibias,f1.
- viterbi.py: Numba fast 3-state, typed state Viterbi, posterior (checked against exact enumeration).
- vote.py: optional hard span vote additions/removals; not tested or used yet.
- stack.py: cross-fitted logistic calibration with shape, tested; no meaningful improvement over simple mean, do not use by default.
- postprocess.py: dictionary/repetition experiments, hurt strong models, rejected.

Local /workspace/experiments:
run1/dev_epoch1,2,5,6,7.npz and test.npz, test_aligned.npz; devmean567.npz (fulltext) and devmean567_aligned.npz canonical. Mean567 F1 .9197798; best individual raw7 .9175774.
largeval/dev_epoch2..7.npz, test.npz, devmean345.npz, devmean567.npz (last latter may currently processing). Mean345 F1 .9247167; raw7 .9220471.
jointval/dev_epoch5..7 and devmean567.npz, F1 .9194112 after calibration.
typedval/devraw_epoch5..7, devmean567_raw.npz (17), devmean567.npz (folded3) F1 .9190744; typed class-consistent decode worse .9171788. Posterior experiment ongoing.
Current/ongoing CPU jobs: functions exec sessions 65060 (mean largeval567 + tune 4-model ensemble, output experiments/ensemble4.json/log), 10258 (typed posterior eval) and 91787 (readonly SSH log) may need polling. Most complete soon. Check log tails.
Best tuned 2-model template (run1 mean567 canonical + large mean345), weights [1,3], B=-.5,I=1: development F1 .9290253.
4-model tuner includes baseline, large last3, joint last3, typed last3, starting weights [1,3,1,1]; inspect results.

Next:
1 Keep concise meaningful commentary updates during long work (~every minute). Check actual UTC regularly.
2 Finish validation of largejoint and cased, collect last-three val logits (largejoint5/6/7, cased4/5/6). Align cased. Tune ensemble incl all models; decide use_cased / skip_joint flags before respective training steps. Model types may be calibrated further if worthwhile; class-consistency was worse.
3 Copy each full-data model's last3 test logits to local, average to one per model and align fulltext/cased. Match validation templates (if largejointall5 epochs, compare val3/4/5). Save scores/recipe/source under output for reproducibility as practical. Replace predictions using selected weights/biases, run check_format.py after each meaningful update.
4 Source output/code is OUTDATED; refresh latest selected scripts, eval, align, viterbi, tune_ensemble (and vote only if used). Include config/README succinctly if useful.
5 Keep final predictions local before deadline, then clean own VM/disk if time after required artifacts copied. Final concise link to predictions and format check. Hidden test F1 unknown; report validation only if desired. Never inspect test text manually or hand-label mentions.

UPDATE ~13:54 UTC: Cased checkpoint defaults do_lower_case=True, which is WRONG for pretrained cased weights. Old casedval experiment F1 .8902 is invalid comparison and UNUSED. Fixed train_v2/joint/typed, eval.py and align_scores.py to pass do_lower_case=False specifically for biobert-base-cased; all uncased checkpoints unchanged. Tokenizer config recorded in new config.json. Need evaluate new casedfixedval (not old casedval).

Pipeline rerouted safely: killed wrapper run_l4_2.sh while fulltextall child continued; new ONLY wrapper run_final.sh waits for fulltextall process (including workers) to finish, then:
1 casedfixedval (6epochs correct case, seed47)
2 largeall (7epochs)
3 casedall if use_cased flag exists (6epochs correct case)
4 jointall unless skip_joint exists (7epochs)
5 typedall (7epochs)
6 largejointall only if starts before15:25; epochs7 before15:15, otherwise5.
No run_l4_2/_3 watchers active now. Fulltextall currently running ~epoch4/5, likely finish14:00. All future names retain configs. Decide use_cased and skip_joint from valid casedfixedval results before largeall ends (~15:00). Skipping weak joint can save ~11min if needed; largejoint validation gain very small .0002, can omit final largejoint if time tight.

All validation models done except corrected cased. largejointval raw best .9196, last3 calibrated F1 .9212944 (devmean567.npz). Baseline last3+large last3+joint last3+typed last3 tuned development F1 .9304249844, weights [1,1.5,.5,1], B=-1,I=.5. Adding largejoint last3 yields .9306261145, same weights plus1. Files experiments/ensemble4.json, ensemble5.json and logs. All mean567 arrays canonical except run1 which uses devmean567_aligned.npz.

Type-class-specific calibration experiment completed: typedval/type_calibration.json has type shifts [-.5,0,-.5,0,-.5,0,0,0], B=-.5,I=1, F1 .9194785 vs unshifted .9190744. Improvement tiny; can test in ensemble, not used yet. Consistent typed Viterbi and posterior WORSE (.91718/.91604), do not use. Whole-mention guideline filter guideline_filter.py created; applies only explicit generic/forbidden terms with zero labeled occurrences in both supplied sets, preserves longer chemical names. Removed only4 FPs on 2-model dev, tiny gain; not applied to test yet, optional.

Exec session44765 reroute may need poll (should complete). Fulltextall autosaves test_epoch5/6/7.npz plus final test.npz/predictions.tsv. Copy and average once ready, align canonical, update output. Source output/code refreshed before case correction; refresh again final. Initial ensemble output remains25262/0formaterrors.
