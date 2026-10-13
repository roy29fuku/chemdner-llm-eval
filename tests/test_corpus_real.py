"""実際のコーパスを使うテスト。data/ に配布ファイルが無ければ飛ばす（`python -m chemdner_bench.corpus fetch` で取得）。

件数の期待値は NCBI 配布版を数えたもの。公式（Krallinger et al. 2015, Table 1）より言及が
training で 16 件・development で 3 件・evaluation で 5 件少なく、evaluation ではさらに長さ 0 の注釈 4 件を除く。
"""

import pytest

from chemdner_bench.cem import score
from chemdner_bench.corpus import corpus_path, gold_spans, load

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not corpus_path().exists(), reason="コーパスが data/ に無い"),
]

EXPECTED = {  # split: (abstracts, mentions, 公式の mentions)
    "training": (3500, 29462, 29478),
    "development": (3500, 29523, 29526),
    "evaluation": (3000, 25342, 25351),
}


@pytest.mark.parametrize("split", EXPECTED)
def test_counts(split):
    docs = load(split)
    n_docs, n_mentions, _ = EXPECTED[split]
    assert len(docs) == n_docs
    assert len({d.pmid for d in docs}) == n_docs
    assert sum(len(d.mentions) for d in docs) == n_mentions


def test_splits_are_disjoint():
    pmids = [{d.pmid for d in load(s)} for s in EXPECTED]
    assert not (pmids[0] & pmids[1]) and not (pmids[0] & pmids[2]) and not (pmids[1] & pmids[2])


def test_gold_scores_one_against_itself():
    docs = load("evaluation")
    gold = gold_spans(docs)
    s = score(gold, gold, [d.pmid for d in docs])
    assert s.f1 == 1.0
