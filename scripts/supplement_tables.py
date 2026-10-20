"""論文の補足資料の表を、本番の採点（score_main.py）と実験 15 の採点（score_exp15.py）の出力から Markdown で書き出す。

表の数値は採点の出力からだけ取る（補足資料に数値を手で写さない）。手法の説明など、採点の出力に無いものは書かない。

    uv run python scripts/supplement_tables.py --main runs/scores/main-2026-10-15/scores.json \\
        --exp15 runs/scores/exp15-2026-10-20/scores.json \\
        --out ../paper-bppb-2026-hands-and-yardstick/manuscript/supplement/tables-ja.md
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

SETTINGS = [("gpt-5.2", "none"), ("gpt-5.2", "medium"), ("gpt-6-luna", "none"), ("gpt-6-luna", "medium"),
            ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
NAME = {"gpt-5.2": "GPT-5.2", "gpt-6-luna": "GPT-6 luna", "gpt-6.1-sol": "GPT-6.1 sol"}
STRONG = [("gpt-6-luna", "medium"), ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]


def pc(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}"


def pt(x: float) -> str:
    return f"{100 * x:+.2f}"


def setting(model: str, effort: str) -> str:
    return f"{NAME[model]} {effort}"


def main_tables(m: dict) -> list[str]:
    runs = [r for r in m["runs"] if r["group"] in ("a", "b-gcp", "b-mac")]
    L = ["### 表 S1　二つのやり方の結果（設定ごと。正解のある評価用の要旨 2,478 件）", "",
         "| 設定 | やり方 | 回 | F1（平均 ± SD） | P（平均） | R（平均） | F1（全 3,000 件。平均） |", "|---|---|---|---|---|---|---|"]
    label = {"a": "直接答える", "b-gcp": "エージェント", "b-mac": "エージェント（1 台の計算機での最初の本番）"}
    for model, effort in SETTINGS:
        for g in ("a", "b-gcp", "b-mac"):
            rs = [r for r in runs if (r["group"], r["model"], r["effort"]) == (g, model, effort)]
            if not rs:
                continue
            f = [r["official"]["f1"] for r in rs]
            L.append(f"| {setting(model, effort)} | {label[g]} | {len(rs)} | {pc(st.mean(f))} ± {pc(st.stdev(f))} | "
                     f"{pc(st.mean(r['official']['precision'] for r in rs))} | {pc(st.mean(r['official']['recall'] for r in rs))} | "
                     f"{pc(st.mean(r['all']['f1'] for r in rs))} |")
    L += ["", "予測を出さずに終わった回は、予測が空として採点した（F1 0）。", "",
          "### 表 S2　エージェント − 直接答える（3 回の平均の差。文書を復元抽出した 95% 区間）", "",
          "| 設定 | 差（2,478 件） | 95% 区間 | 差（全 3,000 件） | 95% 区間 |", "|---|---|---|---|---|"]
    for c in m["comparisons"]:
        if c["b_group"] != "b-gcp":
            continue
        o, a = c["official"], c["all"]
        note = "（予測を出した回だけ）" if c["with_predictions_only"] else ""
        L.append(f"| {setting(c['model'], c['effort'])}{note} | {pt(o['diff'])} | [{pt(o['ci_low'])}, {pt(o['ci_high'])}] | "
                 f"{pt(a['diff'])} | [{pt(a['ci_low'])}, {pt(a['ci_high'])}] |")
    L += ["", f"bootstrap {m['bootstrap']} 回・seed {m['seed']}。差はポイント（F1 × 100）。", "",
          "### 表 S3　エージェントの回ごとの結果（2,478 件）", "",
          "| 設定 | 回 | F1 | P | R |", "|---|---|---|---|---|"]
    for model, effort in SETTINGS:
        for r in [r for r in runs if (r["group"], r["model"], r["effort"]) == ("b-gcp", model, effort)]:
            o = r["official"]
            L.append(f"| {setting(model, effort)} | {r['run'].rsplit('-', 1)[-1]} | {pc(o['f1'])} | {pc(o['precision'])} | "
                     f"{pc(o['recall'])} |")
    return L


def exp15_tables(e: dict) -> list[str]:
    runs = {r["key"]: r for r in e["runs"]}
    L = ["### 表 S4　例を見せた対照の主の比較（エージェント − 例を見せた版のうち強い方。2,478 件）", "",
         "| 設定 | 直接答える（同じ日） | 乱数の例 128 件（3 回の平均） | 似た例 | エージェント | 強い方 | 差 | 95% 区間（2 段） | 区分 | 95% 区間（文書だけ） |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    choice = {"k0": "例なし", "r128": "乱数の例", "similar": "似た例"}
    for m in e["main"]:
        f, t, d = m["f1"], m["two_stage"], m["documents_only"]
        sim = pc(f["similar"]) if "similar" in f else "走らせていない"
        L.append(f"| {m['setting']} | {pc(f['k0'])} | {pc(f['r128'])} | {sim} | {pc(f['b'])} | {choice[t['strong_choice']]} | "
                 f"{pt(t['diff'])} | [{pt(t['ci_low'])}, {pt(t['ci_high'])}] | {t['category']} | "
                 f"[{pt(d['ci_low'])}, {pt(d['ci_high'])}] |")
    cb = e["curve_best_for_type_a"]
    L += ["", f"全体の型: {e['overall_type']}。型 A の条件の 1 つ（GPT-6 luna medium、1,000 件で、エージェント − 乱数の例の曲線の"
          f"最高点 K = {cb['best_k']}）は {pt(cb['diff'])} [{pt(cb['ci_low'])}, {pt(cb['ci_high'])}]・{cb['category']}。"
          f"bootstrap {e['bootstrap']} 回・seed {e['seed']}。", "",
          "### 表 S5　差の何割を、例を見せることで埋めたか", "",
          "| 設定 | エージェント − 例なし | 強い方 − 例なし | 割合 | 95% 区間 |", "|---|---|---|---|---|"]
    for fr in e["fraction_closed"]:
        L.append(f"| {fr['setting']} | {pt(fr['b_minus_k0'])} | {pt(fr['strong_minus_k0'])} | {fr['fraction']:.2f} | "
                 f"[{fr['ci'][0]:.2f}, {fr['ci'][1]:.2f}] |")
    b, c = e["b128"], e["curve"]
    L += ["", "### 表 S6　副の比較", "", "| 比較 | 差 | 95% 区間 | 判定 |", "|---|---|---|---|",
          f"| 128 件だけのエージェント − 同じ 128 件を例に見せた版（GPT-6 luna medium） | {pt(b['diff'])} | "
          f"[{pt(b['ci_low'])}, {pt(b['ci_high'])}] | {b['category']} |",
          f"| 例の数の曲線: F1(1,024 件) − F1(128 件)（GPT-6 luna medium、1,000 件） | {pt(c['kmax_minus_k128'])} | "
          f"[{pt(c['ci_low'])}, {pt(c['ci_high'])}] | {c['rule']} |"]
    for d in e["k0_drift"]:
        L.append(f"| 同じ日の例なし − 1 週間前の直接答える（{d['setting']}） | {pt(d['diff'])} | "
                 f"[{pt(d['ci_low'])}, {pt(d['ci_high'])}] | {d['rule']} |")
    for s in e["similar_vs_random"]:
        L.append(f"| 似た例（{s['similar_k']} 件） − 乱数の例 128 件（{s['setting']}） | {pt(s['diff'])} | "
                 f"[{pt(s['ci_low'])}, {pt(s['ci_high'])}] | {s['category']} |")
    pts = " | ".join(pc(v) for v in c["points"].values())
    L += ["", "### 表 S7　例の数の曲線（GPT-6 luna medium、2,478 件から乱数で選んだ 1,000 件の F1）", "",
          "| 例なし | " + " | ".join(f"{k[1:]} 件" for k in c["points"]) + " | エージェント（参考） |",
          "|" + "---|" * (len(c["points"]) + 2),
          f"| {pc(c['k0'])} | {pts} | {pc(c['b_mean'])} |", "",
          "### 表 S8　回ごとの結果と誤りの内訳（例を見せた対照）", "",
          "| 役割 | 文書 | F1 | P | R | 重なりで数える F1 | 境界の違い（予測／正解） | 余分な予測 | 取りこぼし | API の費用（米ドル） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in runs.items():
        er = x["errors"]
        cost = f"{x['cost_usd']:.2f}" if x.get("cost_usd") is not None else ""
        L.append(f"| {k} | {'1,000' if x['scope'] == 'curve1000' else '2,478'} | {pc(x['f1'])} | {pc(x['precision'])} | "
                 f"{pc(x['recall'])} | {pc(er['overlap_f1'])} | {er['fp_boundary']:,}／{er['fn_boundary']:,} | "
                 f"{er['fp_spurious']:,} | {er['fn_missed']:,} | {cost} |")
    L += ["", "役割の名前: `<設定>/k0` は同じ日の例なし、`r128-s<種>` は乱数の例 128 件、`similar` は似た例、`a-r<回>` は 1 週間前の"
          "直接答える、`b-r<回>` はエージェント、`curve/k<件数>` は曲線、`b128/r<回>` は 128 件だけのエージェント。費用は、"
          "直接答える・例を見せる回は API の費用の合計（キャッシュの書き込みと長い文脈の割増を含む。1 週間前の回は全 3,000 件の分）、"
          "エージェントの回は API の費用だけ（GPU の費用を含まない）。境界の違い＝正解と重なるが一致しない予測（または予測と重なるが"
          "一致しない正解）、余分な予測＝どの正解とも重ならない予測、取りこぼし＝どの予測とも重ならない正解。"]
    return L


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--exp15", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    m = json.loads(args.main.read_text(encoding="utf-8"))
    e = json.loads(args.exp15.read_text(encoding="utf-8"))
    L = ["<!-- chemdner-llm-eval の scripts/supplement_tables.py が採点の出力から書き出した。手で直さない -->", ""]
    L += main_tables(m) + [""] + exp15_tables(e)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
