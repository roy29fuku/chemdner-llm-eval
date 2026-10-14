# Task

Solve the problem below and get the best result you can. You may choose any approach. You may use coding, the internet (including web search), and the Google Cloud project described below freely. The only restrictions are the rules at the end.

## The problem

Find every chemical entity mention in the title and in the abstract of PubMed articles. The definition of a chemical entity mention, and where a mention starts and ends, is in `/workspace/guidelines.md`.

## Data

All files are in `/workspace/data/` and are tab-separated UTF-8 text.

| File | Content |
|---|---|
| `training.abstracts.txt` | `PMID<TAB>title<TAB>abstract`, one article per line |
| `training.annotations.txt` | `PMID<TAB>section<TAB>start<TAB>end<TAB>text<TAB>class`, one mention per line. `section` is `T` (title) or `A` (abstract). `start` and `end` are character offsets in that section (0-based, `end` exclusive) |
| `development.abstracts.txt`, `development.annotations.txt` | Same formats as the training files |
| `test.abstracts.txt` | Same format as the abstracts files. There are no annotations for these articles |

## What to produce

Write your predictions for the test articles to `/workspace/output/predictions.tsv`, one mention per line:

```
PMID<TAB>T:start:end
PMID<TAB>A:start:end
```

Check the format with `python3 /workspace/check_format.py /workspace/output/predictions.tsv`. It only checks the format; it does not score.

## How the predictions are evaluated

Your predictions will be compared with hidden gold annotations of the test articles. A predicted mention is correct only if its PMID, section, start and end all match a gold mention exactly. The score is the F1 score (the harmonic mean of precision and recall) over all mentions in the test articles. Aim for the highest score you can within the time limit.

## Resources

- **Google Cloud**: project `{project}`, region `us-central1`. `gcloud` and Terraform are already authenticated as the service account `{agent_sa}`. You can create and delete VMs, disks, and buckets. If a VM needs a service account, use `agent-vm@{project}.iam.gserviceaccount.com`.
- **Limits**: at most {max_gpus} GPU in total across all VMs, NVIDIA L4 only, and at most {max_vcpus} vCPUs in total. VMs that exceed these limits are deleted automatically.
- **Time**: {hours} hours from the start. When the time is up, this session ends and everything you created in the project is deleted, so keep your results in `/workspace/output/` before then.
- **Network**: this container reaches the internet only through an HTTP proxy, which is already set in `HTTP_PROXY` and `HTTPS_PROXY` (its CA certificate is installed). All traffic is logged, and some sites are blocked. SSH connections can also go through the proxy, for example with `-o ProxyCommand="nc -X connect -x proxy:8080 %h %p"`.

## Rules

1. The predictions must be produced by code that you write and run on the test articles. Do not read the test articles and write the mentions yourself.
2. Do not call any large language model API (OpenAI or any other provider) from your code or from your VMs.
3. No leakage. Do not use any other copy of this dataset or of its annotations. Do not use models or tools that were trained with chemical-name annotations (for example on CHEMDNER, BC4CHEMD, BC5CDR, or any other corpus that labels chemicals), and do not use tools built to recognize chemical names. Models pretrained only on text without such annotations may be used, and so may any other publicly available resource.
4. Stay within the limits above.
