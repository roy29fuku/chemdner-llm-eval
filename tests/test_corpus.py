"""コーパスの読み込みのテスト。小さな BioC を合成して、注釈の取り出し方を確かめる。"""

import pytest

from chemdner_bench.corpus import Mention, gold_spans, parse_bioc

BIOC = """<?xml version="1.0" encoding="utf8" standalone="yes"?>
<collection><source/><date>2021-05-29</date><key/>
  <document><id>12345</id>
    <passage><infon key="type">title</infon><offset>0</offset><text>Effect of aspirin on χ2 levels</text>
      <annotation id="T1"><infon key="type">Chemical</infon><infon key="class">TRIVIAL</infon>
        <location offset="10" length="7"/><text>aspirin</text></annotation>
      <annotation id="MIC1"><infon key="type">MeSH_Indexing_Chemical</infon><infon key="identifier">MESH:D001241</infon>
        <infon key="entry_term">Aspirin</infon><text/></annotation>
    </passage>
    <passage><infon key="type">abstract</infon><offset>0</offset><text>We gave ATP and NaCl.</text>
      <annotation id="T2"><infon key="type">Chemical</infon><infon key="class">ABBREVIATION</infon>
        <location offset="8" length="3"/><text>ATP</text></annotation>
      <annotation id="T3"><infon key="type">Chemical</infon><infon key="class">FORMULA</infon>
        <location offset="16" length="4"/><text>NaCl</text></annotation>
    </passage>
  </document>
  <document><id>67890</id>
    <passage><infon key="type">title</infon><offset>0</offset><text>No chemicals here</text></passage>
    <passage><infon key="type">abstract</infon><offset>0</offset><text>Nothing.</text></passage>
  </document>
</collection>
""".encode("utf-8")


def test_parse_keeps_only_chemdner_annotations_with_section_relative_offsets():
    docs = parse_bioc(BIOC)
    assert [d.pmid for d in docs] == ["12345", "67890"]
    d = docs[0]
    assert d.title == "Effect of aspirin on χ2 levels"
    assert d.mentions == [
        Mention("T", 10, 17, "aspirin", "TRIVIAL"),
        Mention("A", 8, 11, "ATP", "ABBREVIATION"),
        Mention("A", 16, 20, "NaCl", "FORMULA"),
    ]
    for m in d.mentions:
        assert d.text(m.section)[m.start:m.end] == m.text


def test_gold_spans_include_documents_without_mentions():
    gold = gold_spans(parse_bioc(BIOC))
    assert gold == {"12345": {("T", 10, 17), ("A", 8, 11), ("A", 16, 20)}, "67890": set()}


def test_passage_offset_is_subtracted():
    raw = BIOC.replace(b'<offset>0</offset><text>We gave', b'<offset>31</offset><text>We gave')
    raw = raw.replace(b'offset="8" length="3"', b'offset="39" length="3"').replace(b'offset="16" length="4"', b'offset="47" length="4"')
    assert parse_bioc(raw)[0].mentions[1:] == [Mention("A", 8, 11, "ATP", "ABBREVIATION"), Mention("A", 16, 20, "NaCl", "FORMULA")]


def test_span_text_mismatch_is_an_error():
    raw = BIOC.replace(b"<text>ATP</text>", b"<text>ADP</text>")
    with pytest.raises(ValueError, match="一致しない"):
        parse_bioc(raw)
