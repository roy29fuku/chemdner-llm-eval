"""実験 15 の比較の統計（2 段の bootstrap・6 つの区分・全体の型）と、誤りの内訳。

原稿 repo の docs/experiment-protocol.md「追加の対照（実験 15）」の判定の規則をコードにしたもの。F1 は 0〜1 の値で扱い、
同等とみなす幅（±1.0 ポイント）は 0.01 で表す。

2 段の bootstrap: 文書を復元抽出し（同じ抽出で全部の回を採点し直す＝文書で対にする）、その上で群ごとに回を復元抽出して
平均を取る。文書の抽出は boot_f1 が、回の抽出は group_mean が受け持つ。
"""

import numpy as np

from chemdner_bench.cem import Span, _f1

MARGIN = 0.01  # 同等とみなす幅（±1.0 ポイント）
CATEGORIES = ("優越", "正だが幅は不明", "同等", "負だが幅は不明", "劣後", "不明")


def classify(lo: float, hi: float, margin: float = MARGIN) -> str:
    """差の 95% 区間 [lo, hi] を 6 つの区分に分ける。

    区間が ±幅 の中に収まれば、0 の片側にあっても同等とする（「幅は不明」は、0 の片側にあって ±幅 の外まで伸びるとき）。
    """
    if lo > margin:
        return "優越"
    if hi < -margin:
        return "劣後"
    if -margin <= lo and hi <= margin:
        return "同等"
    if lo > 0:
        return "正だが幅は不明"
    if hi < 0:
        return "負だが幅は不明"
    return "不明"


def overall_type(settings: list[str], curve: str) -> str:
    """全体の型（A > D > C > B の順）。settings は 3 設定の区分、curve は gpt-6-luna medium の b − a'(乱数, 曲線の最高点) の区分。"""
    if all(c == "優越" for c in settings) and curve == "優越":
        return "A"
    if "劣後" in settings:
        return "D"
    if all(c == "同等" for c in settings):
        return "C"
    return "B"


def curve_rule(point: float, lo: float, margin: float = MARGIN) -> str:
    """例の数の曲線: F1(Kmax) − F1(128) の点と下限から。"""
    if point >= margin and lo > 0:
        return "伸び続けた"
    if point <= -margin:
        return "下がった"
    return "頭打ちに近い"


def drift_rule(lo: float, hi: float, margin: float = MARGIN) -> str:
    """例なしの再実行 − 10-14 の a の区間から。"""
    return "モデルの変化は見えない" if -margin <= lo and hi <= margin else "外れた"


def doc_weights(idx: np.ndarray, docs: int) -> np.ndarray:
    """復元抽出の添字 (抽出, docs) を、文書ごとの抽出の回数 (抽出, docs) にする。"""
    m = idx.shape[0]
    flat = (idx + np.arange(m)[:, None] * docs).ravel()
    return np.bincount(flat, minlength=m * docs).reshape(m, docs)


def f1_from_weights(weights: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """weights: (抽出, 文書)、rows: (回, 文書, 3) → 抽出ごと・回ごとの micro F1 (抽出, 回)。"""
    runs, docs, _ = rows.shape
    flat = rows.transpose(1, 0, 2).reshape(docs, runs * 3).astype(np.float64)
    counts = (weights.astype(np.float64) @ flat).reshape(-1, runs, 3)
    return _f1(counts)


def boot_f1(rows: np.ndarray, n: int, rng: np.random.Generator, chunk: int = 500) -> np.ndarray:
    """文書を復元抽出した n 回それぞれで、全部の回を同じ抽出で採点した F1 (n, 回)。"""
    docs = rows.shape[1]
    out = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        out.append(f1_from_weights(doc_weights(rng.integers(0, docs, size=(m, docs)), docs), rows))
    return np.concatenate(out)


def group_mean(f1: np.ndarray, members: list[int], rng: np.random.Generator | None) -> np.ndarray:
    """群（同じ条件の回）の平均 F1 の分布 (抽出,)。rng があれば、抽出ごとに群の回を復元抽出する（2 段の bootstrap）。"""
    cols = f1[:, members]
    if rng is None or len(members) == 1:
        return cols.mean(axis=1)
    pick = rng.integers(0, len(members), size=(cols.shape[0], len(members)))
    return np.take_along_axis(cols, pick, axis=1).mean(axis=1)


def interval(dist: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _overlaps(a: Span, b: Span) -> bool:
    """同じ節で、長さのある重なりを持つ（長さ 0 の言及はどれとも重ならない）。"""
    return a[0] == b[0] and max(a[1], b[1]) < min(a[2], b[2])


def error_breakdown(gold: dict[str, set[Span]], pred: dict[str, set[Span]], pmids: list[str]) -> dict:
    """誤りの内訳と、重なりで一致を数える P / R / F1。

    - 予測の誤り（FP）: 正解と重なる＝境界の違い（fp_boundary）、どの正解とも重ならない＝余分（fp_spurious）
    - 正解の取りこぼし（FN）: 予測と重なる＝境界の違い（fn_boundary）、どの予測とも重ならない＝取りこぼし（fn_missed）
    - 重なりの P = 正解と重なる予測の数 / 予測の数、重なりの R = 予測と重なる正解の数 / 正解の数
    """
    c = dict.fromkeys(("tp", "fp_boundary", "fp_spurious", "fn_boundary", "fn_missed",
                       "pred", "gold", "pred_overlapping", "gold_overlapping"), 0)
    for pmid in pmids:
        g, p = gold.get(pmid, set()), pred.get(pmid, set())
        c["pred"] += len(p)
        c["gold"] += len(g)
        c["tp"] += len(g & p)
        for s in p:
            hit = any(_overlaps(s, t) for t in g)
            c["pred_overlapping"] += hit
            if s not in g:
                c["fp_boundary" if hit else "fp_spurious"] += 1
        for t in g:
            hit = any(_overlaps(t, s) for s in p)
            c["gold_overlapping"] += hit
            if t not in p:
                c["fn_boundary" if hit else "fn_missed"] += 1
    prec = c["pred_overlapping"] / c["pred"] if c["pred"] else 0.0
    rec = c["gold_overlapping"] / c["gold"] if c["gold"] else 0.0
    c["overlap_precision"], c["overlap_recall"] = prec, rec
    c["overlap_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return c
