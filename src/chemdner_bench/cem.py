"""CEM（chemical entity mention）の採点器。全条件をこの 1 本で採点する。

定義は BioCreative IV CHEMDNER の CEM と同じ: 言及は (PMID, 節, 開始, 終了) の組で表し、4 つすべてが正解と
完全に一致したものだけを正解とする。予測の重複は 1 つに数える。集計は全文書を合算した micro precision / recall / F1。
区間は採点対象の文書を単位にした bootstrap で出す。

採点範囲は 2 つ。
- official（主）: 2013 年の公式の評価（bc-evaluate）と同じく、正解のある文書だけを採点し、それ以外の文書の予測は
  捨てる（test では 2,478 件）。公式の評価の見本（CHEMDNER_SAMPLE_JUNE25: TP 226・FP 302・FN 334）を再現する
- all（副）: 指定した全文書（test では 3,000 件）を採点し、正解の無い文書への予測も FP に数える

ファイルの形式は公式の CEM と同じ「PMID<TAB>節:開始:終了」（3 列目以降は無視する。公式の rank・confidence が付いていてよい）。
公式の評価と同じく、PMID の文字列は加工しない（先頭の BOM なども取り除かない）。

    uv run python -m chemdner_bench.cem --gold gold.tsv --pred pred.tsv                          # official
    uv run python -m chemdner_bench.cem --gold gold.tsv --pred pred.tsv --scope all --pmids pmids.txt
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Span = tuple[str, int, int]  # (節, 開始, 終了)


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def read_tsv(path: Path) -> dict[str, set[Span]]:
    spans: dict[str, set[Span]] = defaultdict(set)
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cols = line.split("\t")
        try:
            section, start, end = cols[1].split(":")
            spans[cols[0]].add((section, int(start), int(end)))
        except (IndexError, ValueError) as e:
            raise ValueError(f"{path}:{n}: 「PMID<TAB>節:開始:終了」の形式ではない: {line!r}") from e
    return dict(spans)


def write_tsv(path: Path, spans: dict[str, set[Span]]) -> None:
    lines = [f"{pmid}\t{s}:{b}:{e}" for pmid in sorted(spans) for s, b, e in sorted(spans[pmid])]
    Path(path).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def per_document(gold: dict[str, set[Span]], pred: dict[str, set[Span]], pmids: list[str]) -> np.ndarray:
    """文書ごとの (TP, FP, FN)。pmids に無い文書の予測は採点しない（呼び出し側で検査する）。"""
    rows = []
    for pmid in pmids:
        g, p = gold.get(pmid, set()), pred.get(pmid, set())
        rows.append((len(g & p), len(p - g), len(g - p)))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def official_pmids(gold: dict[str, set[Span]]) -> list[str]:
    """公式の採点範囲: 正解の言及が 1 件以上ある文書。"""
    return sorted(pmid for pmid, spans in gold.items() if spans)


def score(gold: dict[str, set[Span]], pred: dict[str, set[Span]], pmids: list[str], drop_unknown: bool = False) -> Score:
    """pmids の文書を採点する。pmids に無い文書の予測は、drop_unknown なら捨て（公式の評価と同じ）、そうでなければエラーにする。"""
    unknown = {pmid for pmid, spans in pred.items() if spans} - set(pmids)  # 予測の無い文書は数えない
    if unknown and not drop_unknown:
        raise ValueError(f"評価対象に無い文書の予測がある: {sorted(unknown)[:5]} など {len(unknown)} 件")
    tp, fp, fn = per_document(gold, pred, pmids).sum(axis=0)
    return Score(int(tp), int(fp), int(fn))


def _f1(counts: np.ndarray) -> np.ndarray:
    """counts: (..., 3) の TP/FP/FN から micro F1（= 2TP / (2TP + FP + FN)）。"""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def bootstrap(rows: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """文書を復元抽出して micro F1 の分布を作り、(点推定, 下限, 上限) を返す。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n, len(rows)))
    dist = _f1(rows[idx].sum(axis=1))
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return float(_f1(rows.sum(axis=0))), float(lo), float(hi)


def paired_bootstrap(rows_a: np.ndarray, rows_b: np.ndarray, n: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float, float, float]:
    """同じ文書の復元抽出で F1(b) − F1(a) の分布を作り、(点推定, 下限, 上限, 差が 0 以下になった割合) を返す。"""
    if rows_a.shape != rows_b.shape:
        raise ValueError("2 つの条件は同じ文書を同じ順で採点したものでなければならない")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows_a), size=(n, len(rows_a)))
    diff = _f1(rows_b[idx].sum(axis=1)) - _f1(rows_a[idx].sum(axis=1))
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    point = float(_f1(rows_b.sum(axis=0)) - _f1(rows_a.sum(axis=0)))
    return point, float(lo), float(hi), float((diff <= 0).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chemdner_bench.cem")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--scope", choices=("official", "all"), default="official",
                        help="official: 正解のある文書だけを採点（公式と同じ）/ all: --pmids の全文書を採点")
    parser.add_argument("--pmids", type=Path, help="--scope all で採点する全文書の PMID の一覧（1 行 1 件）")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    gold, pred = read_tsv(args.gold), read_tsv(args.pred)
    if args.scope == "official":
        pmids = official_pmids(gold)
        dropped = sum(len(v) for k, v in pred.items() if k not in set(pmids))
        s = score(gold, pred, pmids, drop_unknown=True)
    else:
        if not args.pmids:
            parser.error("--scope all には --pmids が要る")
        pmids = args.pmids.read_text(encoding="utf-8").split()
        dropped = 0
        s = score(gold, pred, pmids)
    f1, lo, hi = bootstrap(per_document(gold, pred, pmids), args.bootstrap, args.seed)
    print(f"scope {args.scope}  documents {len(pmids)}  TP {s.tp}  FP {s.fp}  FN {s.fn}  (採点範囲外で捨てた予測 {dropped})")
    print(f"precision {s.precision:.4f}  recall {s.recall:.4f}  F1 {f1:.4f}  (95% CI {lo:.4f}–{hi:.4f}, bootstrap {args.bootstrap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
