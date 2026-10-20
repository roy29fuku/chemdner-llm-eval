"""実験 15（a'・b(128)・例の数の曲線）の予測を、凍結した規則で採点する。

原稿 repo の docs/experiment-protocol.md「追加の対照（実験 15）」の判定と報告をそのままコードにしたもの。手順は 2 つ。

1. 採点の門（正解を使わない）: 実験 15 のすべての回の予測の hash を frozen/exp15-predictions.sha256 に書き、commit する
       uv run python scripts/score_exp15.py hash
2. 採点: test の正解を写してから（`python -m chemdner_bench.corpus fetch --official <DIR> --with-test-gold`）
       uv run python scripts/score_exp15.py score --out runs/scores/exp15-2026-10-20
   hash の一覧が commit 済みで、予測が一覧と一致するときだけ採点する

比べる相手の a（答えを直接聞く・10-14）と b（エージェント・GCP）は、scripts/score_main.py と同じ回を使う。
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chemdner_bench import cem, direct  # noqa: E402
from chemdner_bench.compare import (boot_f1, classify, curve_rule, drift_rule, error_breakdown, group_mean,  # noqa: E402
                                    interval, overall_type)
from chemdner_bench.corpus import TEST_GOLD_STAMP, gold_spans, load, official_dir  # noqa: E402
from score_main import runs_of  # noqa: E402

FROZEN = ROOT / "frozen"
HASHES = FROZEN / "exp15-predictions.sha256"
OFF, CUR = str(FROZEN / "official2478.txt"), str(FROZEN / "curve1000.txt")
SETTINGS = [("gpt-6-luna", "medium"), ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
SIMILAR_K = {("gpt-6-luna", "medium"): 128, ("gpt-6.1-sol", "medium"): 32}  # sol low は似た例を走らせない
SEEDS = (1, 2, 3)
CURVE_K = (8, 32, 128, 256, 512, 1024)  # 128 は a'(乱数, 128) の種 1 を 1,000 件で採点し直す
B128 = [f"b-free-banonly-main-gcp-b128-gpt-6-luna-medium-r{r}" for r in (1, 2, 3)]


def direct_run(model: str, effort: str, pmids: str, **kw) -> tuple[str, Path]:
    name = direct.Config("evaluation", model, effort, pmids_file=pmids, **kw).name()
    return name, ROOT / "runs" / "direct" / name / "predictions.tsv"


def agent_run(name: str) -> tuple[str, Path]:
    return name, ROOT / "runs" / "agent" / name / "workspace" / "output" / "predictions.tsv"


def exp15_runs() -> dict[str, tuple[str, Path]]:
    """実験 15 で走らせた回。キーは役割（例: luna-medium/r128-s1）。"""
    runs = {}
    for model, effort in SETTINGS:
        key = f"{model.removeprefix('gpt-')}-{effort}"
        runs[f"{key}/k0"] = direct_run(model, effort, OFF, rep=10)
        for s in SEEDS:
            runs[f"{key}/r128-s{s}"] = direct_run(model, effort, OFF, shots=128, select="random", shot_seed=s)
        if (model, effort) in SIMILAR_K:
            runs[f"{key}/similar"] = direct_run(model, effort, OFF, shots=SIMILAR_K[(model, effort)], select="similar")
    for k in CURVE_K:
        if k != 128:
            runs[f"curve/k{k}"] = direct_run("gpt-6-luna", "medium", CUR, shots=k, select="random", shot_seed=1)
    for r, name in enumerate(B128, 1):
        runs[f"b128/r{r}"] = agent_run(name)
    frozen = set(json.loads((FROZEN / "aprime.json").read_text())["runs"])
    stray = {n for n, _ in runs.values() if not n.startswith("b-")} - frozen
    if stray or len(frozen) != len([n for n, _ in runs.values() if not n.startswith("b-")]):
        raise SystemExit(f"凍結した回の一覧と合わない: {sorted(stray)}")
    return runs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_lines(runs: dict[str, tuple[str, Path]]) -> list[str]:
    return [f"{sha256(p) if p.exists() else 'MISSING'}  {p.relative_to(ROOT)}" for _, p in sorted(runs.values())]


def cmd_hash() -> int:
    lines = hash_lines(exp15_runs())
    HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [ln for ln in lines if ln.startswith("MISSING")]
    print(f"{HASHES.relative_to(ROOT)} に {len(lines)} 回を書いた（予測なし {len(missing)} 回）。commit して push してから採点する")
    return 0


def check_gate(runs: dict[str, tuple[str, Path]]) -> None:
    rel = str(HASHES.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise SystemExit(f"採点の門: {rel} が commit されていない")
    if HASHES.read_text(encoding="utf-8").splitlines() != hash_lines(runs):
        raise SystemExit("採点の門: 予測が commit した hash と一致しない")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hash")
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--bootstrap", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--pseudo-gold", metavar="KEY",
                   help="コードの確認用: test の正解の代わりに、この役割の回の予測を正解として採点する（門と範囲の照合を飛ばす）")
    args = p.parse_args()
    return cmd_hash() if args.cmd == "hash" else score(args)


def score(args) -> int:
    runs = exp15_runs()
    gated = dict(runs)  # 採点の門で照らす回（実験 15 で走らせた回）
    official = (FROZEN / "official2478.txt").read_text().split()
    curve_docs = (FROZEN / "curve1000.txt").read_text().split()
    # 比べる相手（10-14 の a と GCP の b）を足す
    for model, effort in SETTINGS:
        key = f"{model.removeprefix('gpt-')}-{effort}"
        for r, run in enumerate(runs_of("a", model, effort)):
            runs[f"{key}/a-r{r}"] = run
        for r, run in enumerate(runs_of("b-gcp", model, effort), 1):
            runs[f"{key}/b-r{r}"] = run
    preds = {k: (cem.read_tsv(p) if p.exists() else {}) for k, (_, p) in runs.items()}

    if args.pseudo_gold:
        gold, stamp = {pmid: preds[args.pseudo_gold].get(pmid, set()) for pmid in official}, f"pseudo（{args.pseudo_gold}）"
    else:
        check_gate(gated)
        gold = gold_spans(load("evaluation"))  # 正解が無ければここで止まる
        if official != cem.official_pmids(gold):
            raise SystemExit("凍結した 2,478 件が、正解のある文書と一致しない")
        stamp = (official_dir() / TEST_GOLD_STAMP).read_text().strip()
    rng = np.random.default_rng(args.seed)

    # 範囲ごとに、文書を復元抽出した F1（同じ抽出で全部の回）
    def scope(keys: list[str], pmids: list[str], name: str) -> dict:
        rows = np.stack([cem.per_document(gold, preds[k], pmids) for k in keys])
        return {"name": name, "keys": keys, "rows": rows, "point": cem._f1(rows.sum(axis=1)),
                "boot": boot_f1(rows, args.bootstrap, rng)}

    off_keys = [k for k in runs if not k.startswith("curve/")]
    luna = "6-luna-medium"
    cur_keys = [f"curve/k{k}" for k in CURVE_K if k != 128] + [f"{luna}/r128-s1", f"{luna}/k0"] + \
        [f"{luna}/b-r{r}" for r in (1, 2, 3)]
    sc_off, sc_cur = scope(off_keys, official, "official"), scope(cur_keys, curve_docs, "curve1000")

    def col(sc: dict, keys: list[str]) -> list[int]:
        return [sc["keys"].index(k) for k in keys]

    def mean(sc: dict, keys: list[str], two_stage: bool) -> tuple[float, np.ndarray]:
        m = col(sc, keys)
        return float(sc["point"][m].mean()), group_mean(sc["boot"], m, rng if two_stage else None)

    def diff(point: float, dist: np.ndarray) -> dict:
        lo, hi = interval(dist)
        return {"diff": point, "ci_low": lo, "ci_high": hi, "category": classify(lo, hi)}

    main_rows, fractions, drift, similar = [], [], [], []
    for model, effort in SETTINGS:
        key = f"{model.removeprefix('gpt-')}-{effort}"
        row = {"setting": f"{model} {effort}"}
        for two_stage in (True, False):
            b_pt, b = mean(sc_off, [f"{key}/b-r{r}" for r in (1, 2, 3)], two_stage)
            a_pt, a = mean(sc_off, [f"{key}/a-r{r}" for r in (0, 1, 2)], two_stage)
            k0_pt, k0 = mean(sc_off, [f"{key}/k0"], two_stage)
            r_pt, r128 = mean(sc_off, [f"{key}/r128-s{s}" for s in SEEDS], two_stage)
            cands = {"k0": (k0_pt, k0), "r128": (r_pt, r128)}
            if f"{key}/similar" in runs:
                cands["similar"] = mean(sc_off, [f"{key}/similar"], two_stage)
            names = list(cands)
            stack = np.stack([cands[n][1] for n in names])
            strong, strong_pt = stack.max(axis=0), max(v[0] for v in cands.values())
            label = "two_stage" if two_stage else "documents_only"
            row[label] = {**diff(b_pt - strong_pt, b - strong),
                          "strong_choice": names[int(np.argmax([cands[n][0] for n in names]))],
                          "strong_choice_share": {n: float((stack.argmax(axis=0) == i).mean()) for i, n in enumerate(names)}}
            if two_stage:
                row["f1"] = {"b": b_pt, "a_2026_10_14": a_pt, **{n: v[0] for n, v in cands.items()}, "strong": strong_pt}
                frac = (strong - k0) / (b - k0)
                fractions.append({"setting": row["setting"], "fraction": (strong_pt - k0_pt) / (b_pt - k0_pt),
                                  "ci": interval(frac), "b_minus_k0": b_pt - k0_pt, "strong_minus_k0": strong_pt - k0_pt})
                lo, hi = interval(k0 - a)
                drift.append({"setting": row["setting"], "diff": k0_pt - a_pt, "ci_low": lo, "ci_high": hi,
                              "rule": drift_rule(lo, hi)})
                if "similar" in cands:
                    similar.append({"setting": row["setting"], "similar_k": SIMILAR_K[(model, effort)],
                                    **diff(cands["similar"][0] - r_pt, cands["similar"][1] - r128)})
        main_rows.append(row)

    # 型 A の条件: gpt-6-luna medium の b − a'(乱数, 曲線の最高点)（1,000 件）
    b_pt, b = mean(sc_cur, [f"{luna}/b-r{r}" for r in (1, 2, 3)], True)
    curve_keys = [f"curve/k{k}" if k != 128 else f"{luna}/r128-s1" for k in CURVE_K]
    cm = col(sc_cur, curve_keys)
    curve_best = {**diff(b_pt - float(sc_cur["point"][cm].max()), b - sc_cur["boot"][:, cm].max(axis=1)),
                  "best_k": CURVE_K[int(np.argmax(sc_cur["point"][cm]))]}
    overall = overall_type([r["two_stage"]["category"] for r in main_rows], curve_best["category"])

    # 例の数の曲線: F1(Kmax) − F1(128)（同じ 1,000 件で対にした bootstrap。どちらも 1 回なので文書だけを抽出する）
    kmax, k128 = col(sc_cur, ["curve/k1024"])[0], col(sc_cur, [f"{luna}/r128-s1"])[0]
    c_pt = float(sc_cur["point"][kmax] - sc_cur["point"][k128])
    c_lo, c_hi = interval(sc_cur["boot"][:, kmax] - sc_cur["boot"][:, k128])
    curve = {"points": {f"k{k}": float(sc_cur["point"][i]) for k, i in zip(CURVE_K, cm)},
             "k0": float(sc_cur["point"][col(sc_cur, [f"{luna}/k0"])[0]]),
             "b_mean": b_pt, "kmax_minus_k128": c_pt, "ci_low": c_lo, "ci_high": c_hi, "rule": curve_rule(c_pt, c_lo)}

    # b(128) − a'(乱数, 128) の種 1（予測を出さなかった回は空として採点する）
    bb_pt, bb = mean(sc_off, [f"b128/r{r}" for r in (1, 2, 3)], True)
    s1_pt, s1 = mean(sc_off, [f"{luna}/r128-s1"], True)
    b128 = {**diff(bb_pt - s1_pt, bb - s1), "b128_mean": bb_pt, "aprime_r128_s1": s1_pt,
            "runs_with_predictions": sum(runs[f"b128/r{r}"][1].exists() for r in (1, 2, 3))}

    # 回ごと: P / R / F1・誤りの内訳・費用
    per_run = []
    for k, (name, path) in runs.items():
        pmids = curve_docs if k.startswith("curve/") else official
        sc = cem.Score(*map(int, cem.per_document(gold, preds[k], pmids).sum(axis=0)))
        rec = {"key": k, "run": name, "scope": "curve1000" if k.startswith("curve/") else "official",
               "predictions": path.exists(), "precision": sc.precision, "recall": sc.recall, "f1": sc.f1,
               "errors": error_breakdown(gold, preds[k], pmids)}
        summary = path.parent / "summary.json" if "/runs/direct/" in str(path) else path.parents[2] / "summary.json"
        if summary.exists():
            d = json.loads(summary.read_text())
            rec["cost_usd"] = d.get("cost_usd", d.get("openai_cost_usd"))
            rec["documents"] = d.get("documents")
            rec["calls"] = d.get("calls")
            rec["failed"] = d.get("failed")
        per_run.append(rec)

    result = {"test_gold_fetched_at": stamp, "bootstrap": args.bootstrap, "seed": args.seed,
              "documents": {"official": len(official), "curve": len(curve_docs)},
              "overall_type": overall, "main": main_rows, "curve_best_for_type_a": curve_best,
              "fraction_closed": fractions, "k0_drift": drift, "similar_vs_random": similar, "curve": curve,
              "b128": b128, "runs": per_run}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(result)
    (args.out / "scores.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def pt(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci(lo: float, hi: float) -> str:
    return f"[{pt(lo)}, {pt(hi)}]"


def render(r: dict) -> str:
    L = [f"# 実験 15 の採点（test の正解の取得: {r['test_gold_fetched_at']}）", "",
         f"範囲: 正解のある {r['documents']['official']} 件（曲線は {r['documents']['curve']} 件）。"
         f"bootstrap {r['bootstrap']} 回・seed {r['seed']}。差はポイント（F1 × 100）", "",
         f"## 全体の型: **{r['overall_type']}**", "",
         "## 主の比較: b − a'(強い方)", "",
         "| 設定 | b | a(10-14) | 例なし | 乱数 128 | 似た例 | 強い方 | 差 | 区間（2 段） | 区分 | 区間（文書だけ） | 区分 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in r["main"]:
        f, t, d = m["f1"], m["two_stage"], m["documents_only"]
        sim = f"{100 * f['similar']:.2f}" if "similar" in f else "—"
        L.append(f"| {m['setting']} | {100 * f['b']:.2f} | {100 * f['a_2026_10_14']:.2f} | {100 * f['k0']:.2f} | "
                 f"{100 * f['r128']:.2f} | {sim} | {100 * f['strong']:.2f}（{t['strong_choice']}） | {pt(t['diff'])} | "
                 f"{ci(t['ci_low'], t['ci_high'])} | {t['category']} | {ci(d['ci_low'], d['ci_high'])} | {d['category']} |")
    cb = r["curve_best_for_type_a"]
    L += ["", f"型 A の条件（gpt-6-luna medium、1,000 件）: b − a'(乱数, 曲線の最高点 K={cb['best_k']}) = {pt(cb['diff'])} "
          f"{ci(cb['ci_low'], cb['ci_high'])}・{cb['category']}", "",
          "## 差の何割を、例を見せることで埋めたか（(強い方 − 例なし) / (b − 例なし)）", "",
          "| 設定 | 割合 | 区間 | b − 例なし | 強い方 − 例なし |", "|---|---|---|---|---|"]
    for fr in r["fraction_closed"]:
        L.append(f"| {fr['setting']} | {fr['fraction']:.2f} | [{fr['ci'][0]:.2f}, {fr['ci'][1]:.2f}] | "
                 f"{pt(fr['b_minus_k0'])} | {pt(fr['strong_minus_k0'])} |")
    L += ["", "## 副の比較", "", "| 比較 | 差 | 区間 | 判定 |", "|---|---|---|---|"]
    b = r["b128"]
    L.append(f"| b(128) − a'(乱数, 128) の種 1（gpt-6-luna medium。予測あり {b['runs_with_predictions']}/3 回） | "
             f"{pt(b['diff'])} | {ci(b['ci_low'], b['ci_high'])} | {b['category']} |")
    c = r["curve"]
    L.append(f"| 曲線: F1(1024) − F1(128) | {pt(c['kmax_minus_k128'])} | {ci(c['ci_low'], c['ci_high'])} | {c['rule']} |")
    for d in r["k0_drift"]:
        L.append(f"| 例なしの再実行 − 10-14 の a（{d['setting']}） | {pt(d['diff'])} | {ci(d['ci_low'], d['ci_high'])} | {d['rule']} |")
    for s in r["similar_vs_random"]:
        L.append(f"| 似た例（K={s['similar_k']}） − 乱数 128（{s['setting']}） | {pt(s['diff'])} | "
                 f"{ci(s['ci_low'], s['ci_high'])} | {s['category']} |")
    L += ["", "曲線（1,000 件の F1）: 例なし " + f"{100 * c['k0']:.2f}・"
          + "・".join(f"K={k[1:]} {100 * v:.2f}" for k, v in c["points"].items()) + f"・b {100 * c['b_mean']:.2f}", "",
          "## 回ごと", "",
          "| 役割 | 範囲 | 予測 | P | R | F1 | 重なりの F1 | 境界の違い（FP/FN） | 余分 | 取りこぼし | 費用 $ | 文書 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["runs"]:
        e = x["errors"]
        cost = f"{x['cost_usd']:.2f}" if x.get("cost_usd") is not None else ""
        L.append(f"| {x['key']} | {x['scope']} | {'あり' if x['predictions'] else 'なし'} | {100 * x['precision']:.2f} | "
                 f"{100 * x['recall']:.2f} | {100 * x['f1']:.2f} | {100 * e['overlap_f1']:.2f} | "
                 f"{e['fp_boundary']}/{e['fn_boundary']} | {e['fp_spurious']} | {e['fn_missed']} | {cost} | {x.get('documents') or ''} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
