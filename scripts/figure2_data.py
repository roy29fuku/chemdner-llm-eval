"""論文の Figure 2 に使う数値を、実験 15 の採点（scripts/score_exp15.py の scores.json）から書き出す。

力のある 3 設定ごとに、次の 4 つのやり方の回ごとの F1（official・正解のある 2,478 件）:
- 直接答える: 例なしの再実行（a' と同じ日に走らせた回。1 回）
- 例を見せる（乱数で選んだ 128 件）: 種 1・2・3 の 3 回
- 例を見せる（似た例）: 1 回（決定的）。gpt-6-luna medium は 128 件、gpt-6.1-sol medium は 32 件、gpt-6.1-sol low は走らせていない
- エージェント: GCP の 3 回（Figure 1 と同じ回）

あわせて、走らせる前に決めた主の比較（b − a'(強い方)）の値と区間・区分も写す（図には描かず、キャプションと本文で使う）。

    uv run python scripts/figure2_data.py --scores runs/scores/exp15-2026-10-20/scores.json \\
        --out ../paper-bppb-2026-hands-and-yardstick/manuscript/figures/fig2-data.json
"""

import argparse
import json
import sys
from pathlib import Path

SETTINGS = [("gpt-6-luna", "medium"), ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
SIMILAR_K = {("gpt-6-luna", "medium"): 128, ("gpt-6.1-sol", "medium"): 32}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    f1 = {r["key"]: r["f1"] for r in scores["runs"] if r["scope"] == "official"}
    main_rows = {m["setting"]: m for m in scores["main"]}
    conditions = []
    for model, effort in SETTINGS:
        key = f"{model.removeprefix('gpt-')}-{effort}"
        groups = {"direct": [f"{key}/k0"], "random128": [f"{key}/r128-s{s}" for s in (1, 2, 3)],
                  "similar": [f"{key}/similar"] if (model, effort) in SIMILAR_K else [],
                  "agent": [f"{key}/b-r{r}" for r in (1, 2, 3)]}
        row = {"model": model, "effort": effort, "similar_k": SIMILAR_K.get((model, effort))}
        for g, keys in groups.items():
            runs = [f1[k] for k in keys]
            row[g] = {"runs": runs, "mean": sum(runs) / len(runs) if runs else None}
        m = main_rows[f"{model} {effort}"]["two_stage"]
        row["agent_minus_best_example"] = {"diff": m["diff"], "ci95": [m["ci_low"], m["ci_high"]],
                                           "category": m["category"], "best": m["strong_choice"]}
        conditions.append(row)
    out = {"source": "roy29fuku/chemdner-llm-eval scripts/figure2_data.py（scripts/score_exp15.py の出力から）",
           "test_gold_fetched_at": scores["test_gold_fetched_at"], "bootstrap": scores["bootstrap"],
           "scope": f"official ({scores['documents']['official']} test abstracts with gold mentions)",
           "conditions": conditions}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
