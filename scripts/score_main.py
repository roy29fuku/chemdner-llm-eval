"""本番（実験 13・14）の予測を、test の正解で一度にまとめて採点する。

test の正解は、すべての予測がそろってから一度だけ写す（`python -m chemdner_bench.corpus fetch --official <DIR> --with-test-gold`）。
採点の範囲は主が official（正解のある 2,478 件）、副が all（3,000 件）。

- 回ごとの P / R / F1（予測が無い回は、予測が空として採点する）
- 条件ごとの 3 回の平均 ± SD
- 主の比較: 条件ごとに「b の 3 回の平均 F1 − a の 3 回の平均 F1」。採点対象の文書を復元抽出し（1,000 回）、同じ抽出で
  全部の回を採点し直して平均の差を取る bootstrap で区間を出す（docs/experiment-protocol.md の「報告すること」）

    uv run python scripts/score_main.py --out runs/scores/main-2026-10-15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemdner_bench import cem  # noqa: E402
from chemdner_bench.corpus import TEST_GOLD_STAMP, gold_spans, load, official_dir  # noqa: E402

CONDITIONS = [("gpt-5.2", "none"), ("gpt-5.2", "medium"), ("gpt-6-luna", "none"), ("gpt-6-luna", "medium"),
              ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
AGENT = ROOT / "runs" / "agent"
# 走らせ直した回: 元の回は報告するが採点に使わない（決定の記録 2026-10-14。OpenAI の API の混雑で止まった）
REPLACED = {"b-free-banonly-main-gcp-gpt-6.1-sol-medium-r2": "b-free-banonly-main-gcp-resume-gpt-6.1-sol-medium-r2"}


def runs_of(group: str, model: str, effort: str) -> list[tuple[str, Path]]:
    """群（a・b-mac・b-gcp）と条件の、各回の (名前, 予測のファイル)。b-mac は gpt-6.1-sol を走らせていない。"""
    if group == "a":
        names = [f"evaluation-{model}-{effort}-r{r}" for r in (0, 1, 2)]
        return [(n, ROOT / "runs" / "direct" / n / "predictions.tsv") for n in names]
    prefix = "b-free-banonly-main-gcp" if group == "b-gcp" else "b-free-banonly-main"
    if group == "b-mac" and model == "gpt-6.1-sol":
        return []
    names = [REPLACED.get(n, n) for n in (f"{prefix}-{model}-{effort}-r{r}" for r in (1, 2, 3))]
    return [(n, AGENT / n / "workspace" / "output" / "predictions.tsv") for n in names]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    docs = load("evaluation")  # 正解が無ければここで止まる
    gold = gold_spans(docs)
    scopes = {"official": cem.official_pmids(gold), "all": sorted(gold)}
    stamp = (official_dir() / TEST_GOLD_STAMP).read_text().strip() if (official_dir() / TEST_GOLD_STAMP).exists() else None

    runs, rows = [], {s: [] for s in scopes}
    for group in ("a", "b-mac", "b-gcp"):
        for model, effort in CONDITIONS:
            for name, path in runs_of(group, model, effort):
                pred = cem.read_tsv(path) if path.exists() else {}
                unknown = sorted({pmid for pmid, spans in pred.items() if spans} - set(gold))
                rec = {"group": group, "model": model, "effort": effort, "run": name, "predictions": path.exists(),
                       "unknown_pmids": len(unknown)}
                for s, pmids in scopes.items():
                    r = cem.per_document(gold, pred, pmids)
                    rows[s].append(r)
                    sc = cem.Score(*map(int, r.sum(axis=0)))
                    rec[s] = {"tp": sc.tp, "fp": sc.fp, "fn": sc.fn, "precision": sc.precision, "recall": sc.recall, "f1": sc.f1}
                runs.append(rec)
    arrays = {s: np.stack(rows[s]) for s in scopes}  # (回, 文書, 3)

    def members(group: str, model: str, effort: str, with_predictions_only: bool = False) -> list[int]:
        return [i for i, r in enumerate(runs) if (r["group"], r["model"], r["effort"]) == (group, model, effort)
                and (r["predictions"] or not with_predictions_only)]

    rng = np.random.default_rng(args.seed)
    n_docs = {s: arrays[s].shape[1] for s in scopes}
    idx = {s: rng.integers(0, n_docs[s], size=(args.bootstrap, n_docs[s])) for s in scopes}
    boot = {s: np.stack([cem._f1(arrays[s][:, i, :].sum(axis=1)) for i in idx[s]]) for s in scopes}  # (抽出, 回)

    summary, comparisons = [], []
    for model, effort in CONDITIONS:
        for group in ("a", "b-mac", "b-gcp"):
            m = members(group, model, effort)
            if not m:
                continue
            row = {"group": group, "model": model, "effort": effort, "n": len(m),
                   "missing_predictions": sum(not runs[i]["predictions"] for i in m)}
            for s in scopes:
                f1 = np.array([runs[i][s]["f1"] for i in m])
                row[s] = {"f1_mean": float(f1.mean()), "f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                          "precision_mean": float(np.mean([runs[i][s]["precision"] for i in m])),
                          "recall_mean": float(np.mean([runs[i][s]["recall"] for i in m]))}
            summary.append(row)
        a = members("a", model, effort)
        for group in ("b-gcp", "b-mac"):
            for only in (False, True):
                b = members(group, model, effort, with_predictions_only=only)
                if not b or (only and len(b) == len(members(group, model, effort))):
                    continue  # 予測を出した回だけの比較は、予測の無い回があるときだけ補足として出す
                comp = {"b_group": group, "model": model, "effort": effort, "with_predictions_only": only, "n_b": len(b)}
                for s in scopes:
                    diff = boot[s][:, b].mean(axis=1) - boot[s][:, a].mean(axis=1)
                    point = float(np.mean([runs[i][s]["f1"] for i in b]) - np.mean([runs[i][s]["f1"] for i in a]))
                    lo, hi = np.quantile(diff, [0.025, 0.975])
                    comp[s] = {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "share_le_0": float((diff <= 0).mean())}
                comparisons.append(comp)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"test_gold_fetched_at": stamp, "bootstrap": args.bootstrap, "seed": args.seed,
              "documents": {s: len(v) for s, v in scopes.items()}, "replaced": REPLACED,
              "runs": runs, "summary": summary, "comparisons": comparisons}
    (args.out / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 本番の採点（test の正解の取得: {stamp}）", "",
             f"範囲: official {len(scopes['official'])} 件（主）・all {len(scopes['all'])} 件（副）。bootstrap {args.bootstrap} 回・seed {args.seed}", "",
             "## 条件ごと（3 回の平均 ± SD）", "",
             "| 条件 | 群 | 回 | 予測なし | F1（official） | P | R | F1（all） |", "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        o, al = r["official"], r["all"]
        lines.append(f"| {r['model']} {r['effort']} | {r['group']} | {r['n']} | {r['missing_predictions']} | "
                     f"{o['f1_mean']:.3f} ± {o['f1_sd']:.3f} | {o['precision_mean']:.3f} | {o['recall_mean']:.3f} | "
                     f"{al['f1_mean']:.3f} ± {al['f1_sd']:.3f} |")
    lines += ["", "## b − a（3 回の平均の差・95% 区間）", "",
              "| 条件 | b の群 | 予測を出した回だけ | 差（official） | 区間 | 差 ≤ 0 の割合 | 差（all） | 区間 |",
              "|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        o, al = c["official"], c["all"]
        lines.append(f"| {c['model']} {c['effort']} | {c['b_group']} | {'はい' if c['with_predictions_only'] else ''} | "
                     f"{o['diff']:+.3f} | {o['ci_low']:+.3f}〜{o['ci_high']:+.3f} | {o['share_le_0']:.3f} | "
                     f"{al['diff']:+.3f} | {al['ci_low']:+.3f}〜{al['ci_high']:+.3f} |")
    lines += ["", "## 回ごと", "", "| 群 | 回 | 予測 | P | R | F1（official） | F1（all） |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        o = r["official"]
        lines.append(f"| {r['group']} | {r['run']} | {'あり' if r['predictions'] else 'なし'} | {o['precision']:.3f} | "
                     f"{o['recall']:.3f} | {o['f1']:.3f} | {r['all']['f1']:.3f} |")
    (args.out / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
