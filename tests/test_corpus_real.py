"""実際のコーパスを使うテスト。手元にその版が無ければ飛ばす（`python -m chemdner_bench.corpus fetch ...`）。

official（2013 年の公式配布物）の件数は公式の表（Krallinger et al. 2015, Table 1）と一致する。
ncbi（NCBI の BioC 版）は、各分割で 1 文書ずつ、その文書の言及がすべて欠けている。
"""

import pytest

from chemdner_bench.cem import official_pmids, score
from chemdner_bench.corpus import available, gold_spans, has_test_gold, load

pytestmark = pytest.mark.corpus

EXPECTED = {  # (source, split): (abstracts, abstracts with mentions, mentions)
    ("official", "training"): (3500, 2916, 29478),
    ("official", "development"): (3500, 2907, 29526),
    ("official", "evaluation"): (3000, 2478, 25351),
    # NCBI 版は欠けている文書の言及がすべて無いので、言及のある文書も 1 件ずつ少ない
    ("ncbi", "training"): (3500, 2915, 29462),
    ("ncbi", "development"): (3500, 2906, 29523),
    ("ncbi", "evaluation"): (3000, 2477, 25346),
}
MISSING_IN_NCBI = {"training": ("23645249", 16), "development": ("23646356", 3), "evaluation": ("23645677", 5)}


def needs(source):
    return pytest.mark.skipif(not available(source), reason=f"{source} のコーパスが手元に無い")


def needs_test_gold(source):
    return pytest.mark.skipif(not (available(source) and has_test_gold(source)), reason=f"{source} の test の正解は隔離中")


def marks(source, split):
    return needs_test_gold(source) if split == "evaluation" else needs(source)


@pytest.mark.parametrize("source,split", [pytest.param(s, sp, marks=marks(s, sp)) for s, sp in EXPECTED])
def test_counts(source, split):
    docs = load(split, source)
    n_docs, n_with, n_mentions = EXPECTED[(source, split)]
    assert len(docs) == n_docs == len({d.pmid for d in docs})
    assert sum(1 for d in docs if d.mentions) == n_with
    assert sum(len(d.mentions) for d in docs) == n_mentions


@needs_test_gold("official")
def test_zero_length_gold_in_official_evaluation():
    zero = sorted((d.pmid, m.section, m.start) for d in load("evaluation") for m in d.mentions if m.start == m.end)
    assert zero == [("23265513", "T", 179), ("23360143", "T", 55), ("23420583", "A", 1519), ("23454135", "A", 58)]


@pytest.mark.parametrize("split", [pytest.param(sp, marks=[marks("official", sp), marks("ncbi", sp)]) for sp in MISSING_IN_NCBI])
def test_ncbi_differs_from_official_by_one_document(split):
    official, ncbi = gold_spans(load(split, "official")), gold_spans(load(split, "ncbi"))
    assert official.keys() == ncbi.keys()
    diff = {p: official[p] - ncbi[p] for p in official if official[p] != ncbi[p]}
    pmid, n = MISSING_IN_NCBI[split]
    assert list(diff) == [pmid] and len(diff[pmid]) == n
    assert not ncbi[pmid]  # その文書の言及がすべて欠けている
    assert all(ncbi[p] <= official[p] for p in official)  # NCBI 版にだけある言及は無い


@needs("official")
def test_splits_are_disjoint():
    pmids = [{d.pmid for d in load(s, gold=False)} for s in ("training", "development", "evaluation")]
    assert not (pmids[0] & pmids[1]) and not (pmids[0] & pmids[2]) and not (pmids[1] & pmids[2])


@needs_test_gold("official")
def test_gold_scores_one_against_itself_in_official_scope():
    gold = gold_spans(load("evaluation"))
    pmids = official_pmids(gold)
    assert len(pmids) == 2478
    assert score(gold, gold, pmids).f1 == 1.0


@needs("official")
def test_test_inputs_are_readable_without_gold():
    docs = load("evaluation", gold=False)
    assert len(docs) == 3000 and all(not d.mentions for d in docs)
