"""条件 a'（a ＋ 正解付きの例を見せる）の部品: 例の候補・例の選び方・例の見せ方・test の部分の選び方。

a と同じ指示（凍結した frozen/direct.json）に、正解付きの例を「入力（要旨の JSON）→ 正しい出力（化学物質名を
`<chem>` で囲んだ JSON）」の会話の組で並べ、最後に対象の文書を置く。変えるのは例を見せることだけにする。

- 例の候補（standard_pool）: training と development（b が使えたのと同じ 7,000 件）から、(1) 重なりのある言及・
  長さ 0 の言及を含む文書（タグで正しく表せない）と、(2) development での確認に使う先頭 100 件（確認の対象が自分の
  例に入らないように）を除いた集合。development での確認・test・b(128) で同じ候補を使う
- 乱数で選ぶ例: 候補を種で並べ替え、先頭から K 件。K が小さい例は大きい例の先頭の部分になる（入れ子）。
  例はその回のすべての文書で同じなので、最後の例に置いたキャッシュの区切りから先頭がキャッシュされる
- 似た例を選ぶ例: 対象の文書との TF-IDF の cosine 類似度が高い K 件を、似ている順が対象に近くなるよう
  （最も似た例を対象の直前に）並べる
- test の部分: 正解のある文書（公式の採点範囲）から、種で n 件を選ぶ
"""

import json
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

from chemdner_bench.corpus import Document
from chemdner_bench.inline_tags import tag

_TOKEN = re.compile(r"\w+")  # Unicode の語（α・β なども拾う）


def usable(doc: Document) -> bool:
    """例として正しくタグで表せる文書か（言及に重なりも長さ 0 も無い）。"""
    for section in ("T", "A"):
        spans = sorted((m.start, m.end) for m in doc.mentions if m.section == section)
        if any(e <= s for s, e in spans):
            return False
        if any(s2 < e1 for (_, e1), (s2, _) in zip(spans, spans[1:])):
            return False
    return True


def build_pool(docs: list[Document]) -> tuple[list[Document], int]:
    """例の候補と、除いた文書の数。PMID の順に並べる（種を同じにすれば同じ例になる）。"""
    ok = sorted((d for d in docs if usable(d)), key=lambda d: d.pmid)
    return ok, len(docs) - len(ok)


DEV_CHECK = 100  # development での確認に使う先頭（PMID の順）の件数


def dev_check_docs(development: list[Document]) -> list[Document]:
    """development での確認に使う文書（PMID の順の先頭 100 件。a の試行と同じ文書）。"""
    return sorted(development, key=lambda d: d.pmid)[:DEV_CHECK]


def standard_pool(training: list[Document], development: list[Document]) -> tuple[list[Document], dict]:
    """例の候補と、除いた文書の内訳。"""
    check = {d.pmid for d in dev_check_docs(development)}
    candidates = [d for d in training + development if d.pmid not in check]
    pool, unusable = build_pool(candidates)
    return pool, {"training": len(training), "development": len(development), "dev_check_excluded": len(check),
                  "unusable_excluded": unusable, "pool": len(pool)}


def similar_examples(index: "TfidfIndex", doc: Document, k: int) -> list[Document]:
    """似た例 k 件を、似ている順が対象に近くなるよう並べる（最も似た例が最後＝対象の直前）。"""
    return list(reversed(index.nearest(doc, k)))


def random_examples(pool: list[Document], k: int, seed: int) -> list[Document]:
    """候補を種で並べ替えた先頭の k 件（k が小さい例は、大きい例の先頭の部分）。"""
    order = list(pool)
    random.Random(seed).shuffle(order)
    return order[:k]


def tagged(doc: Document) -> dict:
    """例の正しい出力（a の出力の形式と同じ JSON）。"""
    by = {"T": [], "A": []}
    for m in doc.mentions:
        by[m.section].append((m.start, m.end))
    return {"title": tag(doc.title, by["T"]), "abstract": tag(doc.abstract, by["A"])}


def example_turns(examples: list[Document], breakpoint: bool = False) -> list[dict]:
    """例を、入力と正しい出力の会話の組にする（対象の文書の前に置く）。

    breakpoint が真なら、最後の例の入力にキャッシュの区切りを置く（明示のキャッシュ。区切りまでの先頭がキャッシュされ、
    最後の例の出力と対象の文書は通常の入力として扱われる）。
    """
    turns = []
    for d in examples:
        turns.append({"role": "user", "content": json.dumps({"title": d.title, "abstract": d.abstract}, ensure_ascii=False)})
        turns.append({"role": "assistant", "content": json.dumps(tagged(d), ensure_ascii=False)})
    if breakpoint and turns:
        last = max(i for i, t in enumerate(turns) if t["role"] == "user")
        turns[last] = {"role": "user", "content": [{"type": "input_text", "text": turns[last]["content"],
                                                     "prompt_cache_breakpoint": {"mode": "explicit"}}]}
    return turns


def subset_pmids(pmids: list[str], n: int, seed: int) -> list[str]:
    """test の部分: pmids（公式の採点範囲）から種で n 件を選び、PMID の順に返す。"""
    if n > len(pmids):
        raise ValueError(f"n={n} は文書の数 {len(pmids)} より多い")
    return sorted(random.Random(seed).sample(sorted(pmids), n))


class TfidfIndex:
    """似た例を選ぶ版のための TF-IDF（英数字の語・小文字・L2 正規化した cosine）。語ごとの転置索引で疎に持つ。"""

    def __init__(self, pool: list[Document]):
        self.pool = pool
        counts = [Counter(_tokens(d)) for d in pool]
        df = Counter(t for c in counts for t in c)
        n = len(pool)
        self.idf = {t: math.log((1 + n) / (1 + v)) + 1 for t, v in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, c in enumerate(counts):
            for t, w in self._weights(c).items():
                self.postings[t].append((i, w))

    def _weights(self, counts: Counter) -> dict[str, float]:
        w = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in w.values()))
        return {t: v / norm for t, v in w.items()} if norm else {}

    def nearest(self, doc: Document, k: int) -> list[Document]:
        """類似度の高い順に k 件（同点は PMID の順）。test の文書は候補に含めない前提。"""
        scores = np.zeros(len(self.pool))
        for t, qw in self._weights(Counter(_tokens(doc))).items():
            for i, w in self.postings.get(t, ()):
                scores[i] += qw * w
        order = sorted(range(len(self.pool)), key=lambda i: (-scores[i], self.pool[i].pmid))
        return [self.pool[i] for i in order[:k]]


def _tokens(doc: Document) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(doc.title + " " + doc.abstract)]
