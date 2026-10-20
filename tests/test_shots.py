"""条件 a' の部品（chemdner_bench.shots）のテスト。"""

import json

import pytest

from chemdner_bench.corpus import Document, Mention
from chemdner_bench.inline_tags import parse
from chemdner_bench.shots import (TfidfIndex, build_pool, example_turns, random_examples, subset_pmids, tagged,
                                  usable)


def doc(pmid, title, abstract, mentions):
    return Document(pmid, title, abstract, [Mention(s, b, e, (title if s == "T" else abstract)[b:e]) for s, b, e in mentions])


A = doc("1", "Aspirin and you", "We gave aspirin and ibuprofen to mice.", [("T", 0, 7), ("A", 8, 15), ("A", 20, 29)])
B = doc("2", "Water", "Sodium chloride dissolves in water.", [("A", 0, 15), ("A", 29, 34)])
OVERLAP = doc("3", "X", "sodium chloride", [("A", 0, 6), ("A", 0, 15)])
ZERO = doc("4", "Y", "nothing", [("A", 3, 3)])


def test_usable_rejects_overlap_and_zero_length():
    assert usable(A) and usable(B)
    assert not usable(OVERLAP)
    assert not usable(ZERO)


def test_build_pool_drops_unusable_and_sorts_by_pmid():
    pool, dropped = build_pool([B, OVERLAP, A, ZERO])
    assert [d.pmid for d in pool] == ["1", "2"]
    assert dropped == 2


def test_tagged_round_trips_to_gold_spans():
    out = tagged(A)
    assert out["title"] == "<chem>Aspirin</chem> and you"
    parsed = parse(out["abstract"])
    assert parsed.plain == A.abstract
    assert parsed.spans == [(8, 15), (20, 29)]


def test_example_turns_alternate_user_and_assistant():
    turns = example_turns([A, B])
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert json.loads(turns[0]["content"]) == {"title": A.title, "abstract": A.abstract}
    assert json.loads(turns[1]["content"]) == tagged(A)


def test_random_examples_are_nested_and_reproducible():
    pool = [doc(str(i), f"t{i}", f"a{i}", []) for i in range(50)]
    big = random_examples(pool, 32, seed=7)
    small = random_examples(pool, 8, seed=7)
    assert small == big[:8]
    assert random_examples(pool, 32, seed=7) == big
    assert random_examples(pool, 32, seed=8) != big


def test_subset_pmids_is_reproducible_sorted_and_bounded():
    pmids = [str(i) for i in range(100)]
    s = subset_pmids(pmids, 10, seed=1)
    assert s == sorted(s) and len(set(s)) == 10
    assert subset_pmids(list(reversed(pmids)), 10, seed=1) == s  # 入力の順に依らない
    with pytest.raises(ValueError):
        subset_pmids(pmids, 101, seed=1)


def test_tfidf_nearest_prefers_shared_rare_terms():
    pool = [A, B, doc("5", "Mice", "Mice were fed.", [])]
    index = TfidfIndex(pool)
    query = doc("9", "Ibuprofen in mice", "ibuprofen dose", [])
    assert index.nearest(query, 1)[0].pmid == "1"
    assert [d.pmid for d in index.nearest(query, 3)][0] == "1"


def test_example_turns_put_breakpoint_on_last_example_input():
    turns = example_turns([A, B], breakpoint=True)
    assert turns[2]["role"] == "user" and isinstance(turns[2]["content"], list)
    block = turns[2]["content"][0]
    assert block["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert json.loads(block["text"]) == {"title": B.title, "abstract": B.abstract}
    assert isinstance(turns[0]["content"], str)  # 区切りは最後の例の入力だけ


def test_similar_examples_put_most_similar_last():
    pool = [A, B, doc("5", "Mice", "Mice were fed.", [])]
    index = TfidfIndex(pool)
    query = doc("9", "Ibuprofen in mice", "ibuprofen dose", [])
    from chemdner_bench.shots import similar_examples
    assert similar_examples(index, query, 2)[-1].pmid == "1"


def test_standard_pool_excludes_dev_check_and_unusable():
    from chemdner_bench.shots import standard_pool
    dev = [doc(str(100 + i), "t", "a", []) for i in range(150)] + [OVERLAP]
    train = [A, B, ZERO]
    pool, info = standard_pool(train, dev)
    pmids = {d.pmid for d in pool}
    assert not ({str(100 + i) for i in range(100)} & pmids)  # 確認に使う先頭 100 件は入らない
    assert "4" not in pmids
    assert info["dev_check_excluded"] == 100 and info["pool"] == len(pool)
