"""論文の補足資料の表を、本番の採点（score_main.py）と実験 15 の採点（score_exp15.py）の出力から Markdown で書き出す。

表の数値は採点の出力からだけ取る（補足資料に数値を手で写さない）。手法の説明など、採点の出力に無いものは書かない。

    uv run python scripts/supplement_tables.py --main runs/scores/main-2026-10-15/scores.json \\
        --exp15 runs/scores/exp15-2026-10-20/scores.json \\
        --exp15-addendum runs/scores/exp15-addendum-2026-10-26/scores.json \\
        --out ../paper-bppb-2026-hands-and-yardstick/manuscript/supplement/tables-ja.md

--exp15-addendum を渡すと、追加の 2 回（2026-10-26）を足した版の GPT-6.1 sol low の行を、10-20 の版の行の下に並べる
（主の結果は 10-20 の版のまま。分析計画「追加（2026-10-26）」の判定と報告の 1）
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


def show(name: str) -> str:
    """採点の出力の設定名（gpt-6-luna medium）を、表の表記（GPT-6 luna medium）に直す。"""
    model, effort = name.split(" ")
    return setting(model, effort)


KEY_SETTING = {"6-luna-medium": "GPT-6 luna medium", "6.1-sol-low": "GPT-6.1 sol low", "6.1-sol-medium": "GPT-6.1 sol medium"}


def role(key: str) -> str:
    """score_exp15.py の役割の名前を、表の表記に直す。"""
    head, tail = key.split("/")
    if head == "curve":
        return f"曲線（GPT-6 luna medium）・乱数の例 {int(tail[1:]):,} 件"
    if head == "b128":
        return f"128 件だけのエージェント（GPT-6 luna medium）・{tail}"
    name = KEY_SETTING[head]
    if tail == "k0":
        return f"{name}・同じ日の直接答える"
    if tail == "similar":
        return f"{name}・似た例" + ("（10-26 に追加）" if head == "6.1-sol-low" else "")
    if tail == "k0-r20":
        return f"{name}・同じ日の直接答える（10-26、似た例と同じ日）"
    if tail.startswith("r128-s"):
        return f"{name}・乱数の例 128 件（種 {tail[-1]}）"
    if tail.startswith("a-r"):
        return f"{name}・直接答える（10-14、{tail[2:]}）"
    return f"{name}・エージェント（{tail[2:]}）"


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
    gm = [r["official"]["f1"] for r in runs if (r["group"], r["model"], r["effort"]) == ("b-gcp", "gpt-5.2", "medium")
          and r["predictions"]]
    L += ["", f"予測を出さずに終わった回は、予測が空として採点した（F1 0）。GPT-5.2 medium のエージェントで予測を出した "
          f"{len(gm)} 回だけの平均は {pc(st.mean(gm))}。", "",
          "### 表 S2　エージェント − 直接答える（3 回の平均の差。文書を復元抽出した 95% 区間）", "",
          "| 設定 | 差（2,478 件） | 95% 区間 | 差（全 3,000 件） | 95% 区間 |", "|---|---|---|---|---|"]
    for c in m["comparisons"]:
        if c["b_group"] != "b-gcp":
            continue
        o, a = c["official"], c["all"]
        note = "（予測を出した回だけ）" if c["with_predictions_only"] else ""
        L.append(f"| {setting(c['model'], c['effort'])}{note} | {pt(o['diff'])} | [{pt(o['ci_low'])}, {pt(o['ci_high'])}] | "
                 f"{pt(a['diff'])} | [{pt(a['ci_low'])}, {pt(a['ci_high'])}] |")
    L += ["", f"bootstrap {m['bootstrap']:,} 回・seed {m['seed']}。差はポイント（F1 × 100）。区間は文書だけを復元抽出したもので、回は"
          "固定した（回のばらつき〔表 S1 の SD〕は区間に含まない）。", "",
          "### 表 S3　エージェントの回ごとの結果（2,478 件）", "",
          "| 設定 | 回 | F1 | P | R |", "|---|---|---|---|---|"]
    for model, effort in SETTINGS:
        for r in [r for r in runs if (r["group"], r["model"], r["effort"]) == ("b-gcp", model, effort)]:
            o = r["official"]
            L.append(f"| {setting(model, effort)} | {r['run'].rsplit('-', 1)[-1]} | {pc(o['f1'])} | {pc(o['precision'])} | "
                     f"{pc(o['recall'])} |")
    return L


ADDED = "GPT-6.1 sol low（似た例を足した版・10-26）"


def exp15_tables(e: dict, add: dict | None = None) -> list[str]:
    runs = {r["key"]: r for r in e["runs"]}
    if add:  # 追加の 2 回の行を、回ごとの表に足す
        runs.update({r["key"]: r for r in add["runs"] if r["key"] not in runs})
    add_main = next(m for m in add["main"] if m["setting"] == "gpt-6.1-sol low") if add else None
    L = ["### 表 S4　例を見せた対照の主の比較（エージェント − 例を見せた版のうち強い方。2,478 件）", "",
         "| 設定 | 直接答える（同じ日） | 乱数の例 128 件（3 回の平均） | 似た例 | エージェント | 強い方 | 差 | 95% 区間（2 段） | 区分 | 95% 区間（文書だけ） |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    choice = {"k0": "例なし", "r128": "乱数の例", "similar": "似た例"}
    def s4_row(label: str, m: dict) -> str:
        f, t, d = m["f1"], m["two_stage"], m["documents_only"]
        sim = pc(f["similar"]) if "similar" in f else "走らせていない"
        return (f"| {label} | {pc(f['k0'])} | {pc(f['r128'])} | {sim} | {pc(f['b'])} | {choice[t['strong_choice']]} | "
                f"{pt(t['diff'])} | [{pt(t['ci_low'])}, {pt(t['ci_high'])}] | {t['category']} | "
                f"[{pt(d['ci_low'])}, {pt(d['ci_high'])}] |")

    for m in e["main"]:
        L.append(s4_row(show(m["setting"]), m))
    if add_main:
        L.append(s4_row(ADDED, add_main))
    cb = e["curve_best_for_type_a"]
    L += ["", f"全体の型: {e['overall_type']}。型 A の条件の 1 つ（GPT-6 luna medium、1,000 件で、エージェント − 乱数の例の曲線の"
          f"最高点 K = {cb['best_k']}）は {pt(cb['diff'])} [{pt(cb['ci_low'])}, {pt(cb['ci_high'])}]・{cb['category']}。"
          f"bootstrap {e['bootstrap']:,} 回・seed {e['seed']}。"]
    if add:
        L += ["", f"最後の行は、結果を見た後に査読への対応として足した版（GPT-6.1 sol low の似た例 32 件を 2026-10-26 に走らせ、"
              f"強い方の候補に足した。比べる相手を強くする向きの追加）。主の結果は、その上の 10-20 の版である。"
              f"足した版の全体の型: {add['overall_type']}。"]
    L += ["",
          "### 表 S5　差の何割を、例を見せることで埋めたか", "",
          "| 設定 | エージェント − 例なし | 強い方 − 例なし | 割合 | 95% 区間 |", "|---|---|---|---|---|"]
    def s5_row(label: str, fr: dict) -> str:
        return (f"| {label} | {pt(fr['b_minus_k0'])} | {pt(fr['strong_minus_k0'])} | {fr['fraction']:.2f} | "
                f"[{fr['ci'][0]:.2f}, {fr['ci'][1]:.2f}] |")

    for fr in e["fraction_closed"]:
        L.append(s5_row(show(fr["setting"]), fr))
    if add:
        L.append(s5_row(ADDED, next(fr for fr in add["fraction_closed"] if fr["setting"] == "gpt-6.1-sol low")))
    b, c = e["b128"], e["curve"]
    L += ["", "### 表 S6　副の比較", "", "| 比較 | 差 | 95% 区間 | 判定 |", "|---|---|---|---|",
          f"| 128 件だけのエージェント − 同じ 128 件を例に見せた版（GPT-6 luna medium） | {pt(b['diff'])} | "
          f"[{pt(b['ci_low'])}, {pt(b['ci_high'])}] | {b['category']} |",
          f"| 例の数の曲線: F1(1,024 件) − F1(128 件)（GPT-6 luna medium、1,000 件） | {pt(c['kmax_minus_k128'])} | "
          f"[{pt(c['ci_low'])}, {pt(c['ci_high'])}] | {c['rule']} |"]
    for d in e["k0_drift"]:
        L.append(f"| 同じ日の例なし − 10-14 の直接答える（{show(d['setting'])}） | {pt(d['diff'])} | "
                 f"[{pt(d['ci_low'])}, {pt(d['ci_high'])}] | {d['rule']} |")
    for s in e["similar_vs_random"]:
        L.append(f"| 似た例（{s['similar_k']} 件） − 乱数の例 128 件（{show(s['setting'])}） | {pt(s['diff'])} | "
                 f"[{pt(s['ci_low'])}, {pt(s['ci_high'])}] | {s['category']} |")
    if add:
        s = next(s for s in add["similar_vs_random"] if s["setting"] == "gpt-6.1-sol low")
        L.append(f"| 似た例（{s['similar_k']} 件） − 乱数の例 128 件（{ADDED}） | {pt(s['diff'])} | "
                 f"[{pt(s['ci_low'])}, {pt(s['ci_high'])}] | {s['category']} |")
        d = add["addendum"]["k0_r20_minus_r10"]
        L.append(f"| 例なし: 10-26 − 10-20（GPT-6.1 sol low。モデルの変化の確かめ） | {pt(d['diff'])} | "
                 f"[{pt(d['ci_low'])}, {pt(d['ci_high'])}] | {d['rule']} |")
    pts = " | ".join(pc(v) for v in c["points"].values())
    L += ["", "### 表 S7　例の数の曲線（GPT-6 luna medium、2,478 件から乱数で選んだ 1,000 件の F1）", "",
          "| 例なし | " + " | ".join(f"{int(k[1:]):,} 件" for k in c["points"]) + " | エージェント（参考） |",
          "|" + "---|" * (len(c["points"]) + 2),
          f"| {pc(c['k0'])} | {pts} | {pc(c['b_mean'])} |", "",
          "### 表 S8　回ごとの結果と誤りの内訳（例を見せた対照）", "",
          "| 役割 | 文書 | F1 | P | R | 重なりで数える F1 | 境界の違い（予測／正解） | 余分な予測 | 取りこぼし | API の費用（米ドル） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in runs.items():
        er = x["errors"]
        cost = f"{x['cost_usd']:.2f}" if x.get("cost_usd") is not None else ""
        L.append(f"| {role(k)} | {'1,000' if x['scope'] == 'curve1000' else '2,478'} | {pc(x['f1'])} | {pc(x['precision'])} | "
                 f"{pc(x['recall'])} | {pc(er['overlap_f1'])} | {er['fp_boundary']:,}／{er['fn_boundary']:,} | "
                 f"{er['fp_spurious']:,} | {er['fn_missed']:,} | {cost} |")
    L += ["", "費用は、"
          "直接答える・例を見せる回は API の費用の合計（キャッシュの書き込みと長い文脈の割増を含む。1 週間前の回は全 3,000 件の分）、"
          "エージェントの回は API の費用だけ（GPU の費用を含まない）。境界の違い＝正解と重なるが一致しない予測（または予測と重なるが"
          "一致しない正解）、余分な予測＝どの正解とも重ならない予測、取りこぼし＝どの予測とも重ならない正解。"]
    return L


def posthoc_tables(h: dict) -> list[str]:
    L = ["### 表 S9　事後の探索的な解析: 順位・回のばらつきの感度・同じ相手どうしの割合", "",
         "| 設定 | エージェントの最も低い回 | 他の回の最も高い値 | 全回が上か | 差（強い方との） | 区間（回の分を t 分布で足す） | "
         "割合（乱数の例 128 件だけ） | 割合（似た例だけ） | 残りの差（重なりで数える F1） |", "|---|---|---|---|---|---|---|---|---|"]
    for r in h["settings"]:
        k, s_ = r["rank"], r["run_variance_sensitivity"]
        fr = r["fraction_same_opponent"]
        sim = f"{fr['similar']:.2f}" if fr["similar"] is not None else "走らせていない"
        L.append(f"| {show(r['setting'])} | {pc(k['agent_min'])} | {pc(k['others_max'])} | "
                 f"{'はい' if k['all_agent_runs_above_all_others'] else 'いいえ'} | {pt(s_['diff'])} | "
                 f"[{pt(s_['ci'][0])}, {pt(s_['ci'][1])}] | {fr['r128']:.2f} | {sim} | {pt(r['overlap_f1_residual'])} |")
    L += ["", "他の回 = 同じ設定の直接答える回（10-14 の 3 回と同じ日の 1 回）と例を見せた回すべて。回の分の分散は、エージェントと"
          "強い方の回の SD から出し（1 回だけの条件は 10-14 の直接答える 3 回の SD で代えた）、文書の抽出の分散に足して、"
          "t 分布（自由度 2）で 95% 区間にした。", "",
          "### 表 S10　事後の探索的な解析: 同等とみなす幅を変えたときの区分", "",
          "| 幅（ポイント） | GPT-6 luna medium | GPT-6.1 sol low | GPT-6.1 sol medium | 曲線の最高点（型 A の条件） | 全体の型 |",
          "|---|---|---|---|---|---|"]
    for m, v in h["margin_sensitivity"].items():
        L.append(f"| ±{m} | " + " | ".join(v["categories"]) + f" | {v['curve']} | {v['type']} |")
    L += ["", "±1.0 が実行の前に決めた幅。", "",
          "### 表 S11　事後の探索的な解析: 3 回の多数決（2 回以上に出た言及を採る）", "",
          "| 設定 | 直接答える 3 回の平均 → 多数決 | 乱数の例 3 組の平均 → 多数決 | 乱数 3 組と似た例の 4 本で 2 本以上 | 似た例（1 回） |",
          "|---|---|---|---|---|"]
    for r in h["settings"]:
        v = r["vote"]
        four = pc(v["r128_and_similar_4"]) if "r128_and_similar_4" in v else "—"
        sim = pc(v["similar"]) if "similar" in v else "—"
        L.append(f"| {show(r['setting'])} | {pc(v['a_mean'])} → {pc(v['a_3runs'])} | {pc(v['r128_mean'])} → {pc(v['r128_3sets'])} | "
                 f"{four} | {sim} |")
    classes = ["SYSTEMATIC", "TRIVIAL", "FAMILY", "FORMULA", "ABBREVIATION", "IDENTIFIER", "MULTIPLE"]
    L += ["", "### 表 S12　事後の探索的な解析: 正解の分類ごとの再現率（R。直接答える 10-14 の 3 回 ／ 強い方 ／ エージェント 3 回の平均）", "",
          "| 設定 | " + " | ".join(f"{c}（{h['settings'][0]['recall_by_class']['counts'][c]:,}）" for c in classes) + " |",
          "|---|" + "---|" * len(classes)]
    for r in h["settings"]:
        rc = r["recall_by_class"]
        L.append(f"| {show(r['setting'])} | " + " | ".join(f"{pc(rc['a'][c], 1)} ／ {pc(rc['best'][c], 1)} ／ {pc(rc['agent'][c], 1)}"
                                                       for c in classes) + " |")
    L += ["", "分類は CHEMDNER の正解に付いている 8 分類（括弧は正解のある 2,478 件での言及の数。NO CLASS 41 件は省いた）。採点には使わない。"]
    return L


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--exp15", type=Path, required=True)
    p.add_argument("--exp15-addendum", type=Path, help="score_exp15.py --addendum の scores.json（追加の 2 回を足した版）")
    p.add_argument("--posthoc", type=Path, help="scripts/posthoc_exp15.py の posthoc.json（事後の探索的な解析）")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    m = json.loads(args.main.read_text(encoding="utf-8"))
    e = json.loads(args.exp15.read_text(encoding="utf-8"))
    L = ["<!-- chemdner-llm-eval の scripts/supplement_tables.py が採点の出力から書き出した。手で直さない -->", ""]
    add = json.loads(args.exp15_addendum.read_text(encoding="utf-8")) if args.exp15_addendum else None
    L += main_tables(m) + [""] + exp15_tables(e, add)
    if args.posthoc:
        L += [""] + posthoc_tables(json.loads(args.posthoc.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
