"""論文の Figure 2（改訂稿。初稿の Figure 1）に使う数値を、本番の採点（scripts/score_main.py と同じ回・同じ採点）から書き出す。

- 回ごとの F1（official・正解のある 2,478 件）
- 条件ごとの 3 回の平均と、その 95% 区間（採点対象の文書を復元抽出し、同じ抽出で 3 回を採点し直して平均を取る bootstrap。
  score_main.py の差の区間と同じ方法・同じ seed）
- b のうち、GPU の VM を立てて事前学習済みモデルを fine-tuning し、予測まで出した回の数（監査で判定した値。
  原稿 repo の dev_logs「再実験 5/6」の表と同じ）

    uv run python scripts/figure1_data.py --out ../paper-bppb-2026-hands-and-yardstick/manuscript/figures/fig2-data.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chemdner_bench import cem  # noqa: E402
from chemdner_bench.corpus import TEST_GOLD_STAMP, gold_spans, load, official_dir  # noqa: E402
from score_main import CONDITIONS, runs_of  # noqa: E402

# 監査で判定した、GPU で fine-tuning して予測まで出した回（dev_logs 再実験 5/6）
GPU_FINE_TUNED = {("gpt-5.2", "none"): 0, ("gpt-5.2", "medium"): 0, ("gpt-6-luna", "none"): 0,
                  ("gpt-6-luna", "medium"): 3, ("gpt-6.1-sol", "low"): 3, ("gpt-6.1-sol", "medium"): 3}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gold = gold_spans(load("evaluation"))
    pmids = cem.official_pmids(gold)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(pmids), size=(args.bootstrap, len(pmids)))
    rows = []
    for model, effort in CONDITIONS:
        row = {"model": model, "effort": effort, "b_gpu_fine_tuned": GPU_FINE_TUNED[(model, effort)]}
        for group in ("a", "b-gcp"):
            arrays, runs = [], []
            for name, path in runs_of(group, model, effort):
                pred = cem.read_tsv(path) if path.exists() else {}
                r = cem.per_document(gold, pred, pmids)
                arrays.append(r)
                runs.append({"run": name, "predictions": path.exists(), "f1": float(cem._f1(r.sum(axis=0)))})
            stack = np.stack(arrays)  # (回, 文書, 3)
            boot = np.stack([cem._f1(stack[:, i, :].sum(axis=1)).mean() for i in idx])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            row[group] = {"runs": runs, "mean": float(np.mean([x["f1"] for x in runs])), "ci95": [float(lo), float(hi)]}
        rows.append(row)
    stamp = (official_dir() / TEST_GOLD_STAMP).read_text().strip()
    out = {"source": "roy29fuku/chemdner-llm-eval scripts/figure1_data.py", "test_gold_fetched_at": stamp,
           "scope": f"official ({len(pmids)} test abstracts with gold mentions)", "bootstrap": args.bootstrap,
           "seed": args.seed, "conditions": rows}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(r["model"], r["effort"], f"a {r['a']['mean']:.3f} {r['a']['ci95']}", f"b {r['b-gcp']['mean']:.3f} {r['b-gcp']['ci95']}",
              "gpu-ft", r["b_gpu_fine_tuned"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
