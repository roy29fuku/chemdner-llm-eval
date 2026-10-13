"""インラインタグの解析と位置合わせのテスト。"""

import pytest

from chemdner_bench.inline_tags import align, parse, tag

TEXT = "We gave ATP and NaCl to rats."


def test_parse_simple():
    p = parse("We gave <chem>ATP</chem> and <chem>NaCl</chem> to rats.")
    assert p.plain == TEXT
    assert p.spans == [(8, 11), (16, 20)]
    assert p.violations == []


def test_parse_records_violations_instead_of_guessing():
    p = parse("<chem>A<chem>B</chem>C</chem> x</chem> <chem></chem> <chem>tail")
    assert p.plain == "ABC x  tail"
    assert p.spans == [(0, 2)]  # 入れ子の内側の開きタグは無視し、最初の閉じタグで閉じる
    assert any(v.startswith("nested-open") for v in p.violations)
    assert sum(v.startswith("unmatched-close") for v in p.violations) == 2
    assert any(v.startswith("empty") for v in p.violations)
    assert any(v.startswith("unclosed") for v in p.violations)


def test_align_exact_when_text_is_unchanged():
    a = align(parse("We gave <chem>ATP</chem> and <chem>NaCl</chem> to rats."), TEXT)
    assert a.text_matches and a.spans == [(8, 11), (16, 20)] and a.dropped == 0


def test_align_recovers_positions_when_other_text_changed():
    # 前の空白が 1 つ増え、語が 1 つ消えた。言及そのものは変わっていないので位置を戻せる
    a = align(parse("We  gave <chem>ATP</chem> and <chem>NaCl</chem> rats."), TEXT)
    assert not a.text_matches
    assert a.spans == [(8, 11), (16, 20)] and a.dropped == 0


def test_align_drops_mention_whose_text_was_altered():
    a = align(parse("We gave <chem>ADP</chem> and <chem>NaCl</chem> to rats."), TEXT)
    assert a.spans == [(16, 20)] and a.dropped == 1


def test_tag_roundtrips_with_parse():
    spans = [(8, 11), (16, 20)]
    p = parse(tag(TEXT, spans))
    assert p.plain == TEXT and p.spans == spans


def test_tag_rejects_overlapping_spans():
    with pytest.raises(ValueError):
        tag(TEXT, [(8, 11), (10, 14)])
