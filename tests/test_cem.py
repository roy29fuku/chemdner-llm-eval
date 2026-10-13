"""採点器のテスト。正解を入れたら満点、既知の誤りを入れたら期待どおりに下がることを確かめる。"""

import numpy as np
import pytest

from chemdner_bench.cem import Score, bootstrap, paired_bootstrap, per_document, read_tsv, score, write_tsv

GOLD = {
    "1": {("T", 0, 7), ("A", 10, 18), ("A", 30, 35)},
    "2": {("A", 5, 12)},
    "3": set(),  # 言及の無い文書
}
PMIDS = ["1", "2", "3"]


def test_perfect_prediction_scores_one():
    s = score(GOLD, GOLD, PMIDS)
    assert (s.tp, s.fp, s.fn) == (4, 0, 0)
    assert s.precision == s.recall == s.f1 == 1.0


def test_empty_prediction_scores_zero():
    s = score(GOLD, {}, PMIDS)
    assert (s.tp, s.fp, s.fn) == (0, 0, 4)
    assert s.precision == s.recall == s.f1 == 0.0


def test_boundary_shift_is_one_false_positive_and_one_false_negative():
    pred = {**GOLD, "2": {("A", 5, 13)}}  # 終了位置が 1 文字ずれた
    s = score(GOLD, pred, PMIDS)
    assert (s.tp, s.fp, s.fn) == (3, 1, 1)


def test_same_offsets_in_other_section_do_not_match():
    pred = {**GOLD, "2": {("T", 5, 12)}}  # 同じ位置でも節が違う
    s = score(GOLD, pred, PMIDS)
    assert (s.tp, s.fp, s.fn) == (3, 1, 1)


def test_extra_and_missing_mentions():
    pred = {"1": {("T", 0, 7), ("A", 40, 44)}, "2": {("A", 5, 12)}, "3": {("A", 0, 3)}}
    s = score(GOLD, pred, PMIDS)
    assert (s.tp, s.fp, s.fn) == (2, 2, 2)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)


def test_prediction_for_unknown_document_is_rejected():
    with pytest.raises(ValueError, match="評価対象に無い文書"):
        score(GOLD, {"999": {("A", 0, 1)}}, PMIDS)


def test_tsv_roundtrip_dedupes_and_ignores_extra_columns(tmp_path):
    path = tmp_path / "pred.tsv"
    # 公式の予測ファイルの形式（3・4 列目に rank と confidence）でも読める。重複は 1 つに数える
    path.write_text("1\tT:0:7\t1\t0.9\n1\tT:0:7\t2\t0.8\n2\tA:5:12\n\n", encoding="utf-8")
    assert read_tsv(path) == {"1": {("T", 0, 7)}, "2": {("A", 5, 12)}}
    out = tmp_path / "out.tsv"
    write_tsv(out, GOLD)
    assert read_tsv(out) == {k: v for k, v in GOLD.items() if v}


def test_malformed_line_is_reported_with_line_number(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("1\tT:0:7\n2\tA-5-12\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.tsv:2"):
        read_tsv(path)


def test_score_f1_matches_definition():
    s = Score(tp=3, fp=1, fn=2)
    assert s.f1 == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))


def test_bootstrap_of_perfect_prediction_is_degenerate_at_one():
    # 言及の無い文書だけを引いた標本は 0/0 になるので、言及のある文書だけで確かめる
    rows = per_document(GOLD, GOLD, ["1", "2"])
    assert bootstrap(rows, n=200, seed=1) == (1.0, 1.0, 1.0)


def test_bootstrap_is_deterministic_and_brackets_point_estimate():
    rng = np.random.default_rng(42)
    rows = rng.integers(0, 5, size=(300, 3))
    a = bootstrap(rows, n=500, seed=7)
    assert a == bootstrap(rows, n=500, seed=7)
    point, lo, hi = a
    assert lo <= point <= hi
    assert point == pytest.approx(2 * rows[:, 0].sum() / (2 * rows[:, 0].sum() + rows[:, 1].sum() + rows[:, 2].sum()))


def test_paired_bootstrap_of_identical_systems_is_zero():
    rows = np.random.default_rng(3).integers(0, 5, size=(100, 3))
    point, lo, hi, p_le_zero = paired_bootstrap(rows, rows.copy(), n=300, seed=0)
    assert (point, lo, hi) == (0.0, 0.0, 0.0)
    assert p_le_zero == 1.0


def test_paired_bootstrap_detects_uniformly_better_system():
    rng = np.random.default_rng(5)
    tp = rng.integers(5, 10, size=200)
    rows_a = np.stack([tp, np.full(200, 3), np.full(200, 3)], axis=1)  # 誤りが多い
    rows_b = np.stack([tp, np.full(200, 1), np.full(200, 1)], axis=1)  # 同じ TP で誤りが少ない
    point, lo, hi, p_le_zero = paired_bootstrap(rows_a, rows_b, n=500, seed=0)
    assert point > 0 and lo > 0
    assert p_le_zero == 0.0


def test_paired_bootstrap_requires_same_documents():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros((3, 3)), np.zeros((4, 3)))
