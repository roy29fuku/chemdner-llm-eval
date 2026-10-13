# Task

Solve the problem below. You may use coding and the Google Cloud project described below freely.

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
{evaluation}
## Resources

- **Google Cloud**: project `{project}`, region `us-central1`. `gcloud` and Terraform are already authenticated as the service account `agent-b@{project}.iam.gserviceaccount.com`. You can create and delete VMs, disks, and buckets. If a VM needs a service account, use `agent-vm@{project}.iam.gserviceaccount.com`.
- **Limits**: at most {max_gpus} GPU in total across all VMs, NVIDIA L4 only, and at most {max_vcpus} vCPUs in total. VMs that exceed these limits are deleted automatically.
- **Time**: {hours} hours from the start. When the time is up, this session ends and everything you created in the project is deleted, so keep your results in `/workspace/output/` before then.
- **Network**: this container reaches the internet only through an HTTP proxy, which is already set in `HTTP_PROXY` and `HTTPS_PROXY` (its CA certificate is installed). Some sites are blocked. SSH connections can also go through the proxy, for example with `-o ProxyCommand="nc -X connect -x proxy:8080 %h %p"`.

## Rules

1. The predictions must be produced by code that you write and run on the test articles. Do not read the test articles and write the mentions yourself.
2. Do not call any large language model API (OpenAI or any other provider) from your code or from your VMs.
3. Do not use existing pretrained models or tools for chemical named-entity recognition, whether downloaded as a model or installed as a package. Do not use other copies of this dataset or of related annotated corpora.
4. Stay within the limits above.
