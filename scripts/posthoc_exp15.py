"""実験 15 の事後の解析（探索的）。走らせる前に決めた判定（score_exp15.py）は変えず、査読で問われうる点に答える材料を出す。

新しい API の呼び出しはしない。既にある予測を、test の正解で採点し直すだけ。どれも結果を見た後に足した解析なので、
論文では「事後の探索的な解析」と書き、全体の型（B）は変えない。

1. 順位: 設定ごとに、エージェントの各回が、直接答える・例を見せた回のどれよりも高いか（区間の仮定に頼らない）
2. 回のばらつきの感度: 差の区間を、文書の抽出による分散と、回の分散（n が 1 の条件は 10-14 の直接答える 3 回の SD で代える）
   の和から、t 分布（自由度 2）で出す
3. 同等とみなす幅の感度: 幅を 0.5・1.0・2.0 ポイントにしたときの区分と全体の型
4. 同じ相手どうしの「差を埋めた割合」: 乱数の例 128 件だけ（3 設定）・似た例だけ（2 設定）
5. 重なりで数える F1 での残りの差
6. 多数決: 3 回（例の組の違う 3 組、または 10-14 の直接答える 3 回）の予測を、2 回以上に出た言及で合わせた F1
7. 正解の分類（CHEMDNER の 8 分類）ごとの再現率

    uv run python scripts/posthoc_exp15.py --scores runs/scores/exp15-2026-10-20/scores.json --out runs/scores/exp15-2026-10-20/posthoc
"""

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chemdner_bench import cem  # noqa: E402
from chemdner_bench.compare import MARGIN, boot_f1, classify, error_breakdown, overall_type  # noqa: E402
from chemdner_bench.corpus import gold_spans, load  # noqa: E402
from score_exp15 import SETTINGS, exp15_runs  # noqa: E402
from score_main import runs_of  # noqa: E402

T975_DF2 = 4.302652729911275  # t 分布（自由度 2）の 97.5% 点


def key_of(model: str, effort: str) -> str:
    return f"{model.removeprefix('gpt-')}-{effort}"


def vote(preds: list[dict], need: int) -> dict:
    out = defaultdict(set)
    for pmid in set().union(*preds):
        c = Counter(s for p in preds for s in p.get(pmid, set()))
        out[pmid] = {s for s, n in c.items() if n >= need}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10_000)
    args = p.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    docs = load("evaluation")
    gold = gold_spans(docs)
    official = (ROOT / "frozen" / "official2478.txt").read_text().split()
    cls = {(d.pmid, m.section, m.start, m.end): m.cls for d in docs for m in d.mentions}

    runs = exp15_runs()
    for model, effort in SETTINGS:
        k = key_of(model, effort)
        for r, run in enumerate(runs_of("a", model, effort)):
            runs[f"{k}/a-r{r}"] = run
        for r, run in enumerate(runs_of("b-gcp", model, effort), 1):
            runs[f"{k}/b-r{r}"] = run
    preds = {k: (cem.read_tsv(path) if path.exists() else {}) for k, (_, path) in runs.items() if not k.startswith("curve/")}
    f1 = {k: cem.score(gold, v, official, drop_unknown=True).f1 for k, v in preds.items()}
    rng = np.random.default_rng(0)
    out = {"note": "事後の探索的な解析。走らせる前に決めた判定と全体の型は変えない", "settings": []}

    for model, effort in SETTINGS:
        k = key_of(model, effort)
        b = [f"{k}/b-r{r}" for r in (1, 2, 3)]
        a = [f"{k}/a-r{r}" for r in (0, 1, 2)]
        r128 = [f"{k}/r128-s{s}" for s in (1, 2, 3)]
        sim = [f"{k}/similar"] if f"{k}/similar" in preds else []
        k0 = [f"{k}/k0"]
        others = k0 + r128 + sim + a
        row = {"setting": f"{model} {effort}"}
        # 1. 順位
        row["rank"] = {"agent_min": min(f1[x] for x in b), "others_max": max(f1[x] for x in others),
                       "all_agent_runs_above_all_others": min(f1[x] for x in b) > max(f1[x] for x in others)}
        # 2. 回のばらつきの感度（強い方の候補ごと）
        keys = b + k0 + r128 + sim
        rows = np.stack([cem.per_document(gold, preds[x], official) for x in keys])
        boot = boot_f1(rows, args.bootstrap, rng)
        col = {x: i for i, x in enumerate(keys)}
        sd_a = st.stdev(f1[x] for x in a)
        cands = {"k0": k0, "r128": r128} | ({"similar": sim} if sim else {})
        best = max(cands, key=lambda c: st.mean(f1[x] for x in cands[c]))
        bm = boot[:, [col[x] for x in b]].mean(axis=1)
        cm = boot[:, [col[x] for x in cands[best]]].mean(axis=1)
        var_doc = float(np.var(bm - cm, ddof=1))
        sd_c = st.stdev(f1[x] for x in cands[best]) if len(cands[best]) > 1 else sd_a
        var_run = st.stdev(f1[x] for x in b) ** 2 / 3 + sd_c ** 2 / len(cands[best])
        d = st.mean(f1[x] for x in b) - st.mean(f1[x] for x in cands[best])
        half = T975_DF2 * (var_doc + var_run) ** 0.5
        row["run_variance_sensitivity"] = {"best": best, "diff": d, "ci": [d - half, d + half],
                                           "sd_single_run_proxy": sd_a if len(cands[best]) == 1 else None}
        # 4. 同じ相手どうしの割合
        bk = st.mean(f1[x] for x in b) - f1[k0[0]]
        row["fraction_same_opponent"] = {"r128": (st.mean(f1[x] for x in r128) - f1[k0[0]]) / bk,
                                         "similar": (f1[sim[0]] - f1[k0[0]]) / bk if sim else None}
        # 5. 重なりで数える F1 の残りの差
        ov = lambda xs: st.mean(error_breakdown(gold, preds[x], official)["overlap_f1"] for x in xs)  # noqa: E731
        row["overlap_f1_residual"] = ov(b) - ov(cands[best])
        row["exact_f1_residual"] = d
        # 6. 多数決
        row["vote"] = {"a_3runs": cem.score(gold, vote([preds[x] for x in a], 2), official, drop_unknown=True).f1,
                       "a_mean": st.mean(f1[x] for x in a),
                       "r128_3sets": cem.score(gold, vote([preds[x] for x in r128], 2), official, drop_unknown=True).f1,
                       "r128_mean": st.mean(f1[x] for x in r128)}
        if sim:
            row["vote"]["r128_and_similar_4"] = cem.score(gold, vote([preds[x] for x in r128 + sim], 2), official,
                                                          drop_unknown=True).f1
            row["vote"]["similar"] = f1[sim[0]]
        # 7. 分類ごとの再現率（直接答える 3 回・強い方・エージェント 3 回の平均）
        def recall_by_class(xs: list[str]) -> dict:
            per = defaultdict(list)
            for x in xs:
                hit, tot = Counter(), Counter()
                for pmid in official:
                    pr = preds[x].get(pmid, set())
                    for s in gold.get(pmid, set()):
                        c = cls.get((pmid, *s), "")
                        tot[c] += 1
                        hit[c] += s in pr
                for c in tot:
                    per[c].append(hit[c] / tot[c])
            return {c: st.mean(v) for c, v in per.items()}
        totals = Counter(cls.get((pmid, *s), "") for pmid in official for s in gold.get(pmid, set()))
        row["recall_by_class"] = {"counts": dict(totals), "a": recall_by_class(a), "best": recall_by_class(cands[best]),
                                  "agent": recall_by_class(b)}
        out["settings"].append(row)

    # 3. 幅の感度（scores.json の 2 段の区間から）
    sens = {}
    for m in (0.005, MARGIN, 0.02):
        cats = [classify(r["two_stage"]["ci_low"], r["two_stage"]["ci_high"], m) for r in scores["main"]]
        cb = scores["curve_best_for_type_a"]
        sens[f"{100 * m:.1f}"] = {"categories": cats, "curve": classify(cb["ci_low"], cb["ci_high"], m),
                                  "type": overall_type(cats, classify(cb["ci_low"], cb["ci_high"], m))}
    out["margin_sensitivity"] = sens

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "posthoc.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
