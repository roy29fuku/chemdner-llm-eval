"""条件 a-A の実行のテスト。API は呼ばず、偽のクライアントで応答を返す。"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from chemdner_bench import direct
from chemdner_bench.corpus import Document, Mention
from chemdner_bench.llm import CallRecord

TRAIN = [
    Document("1", "Aspirin study", "We gave aspirin.", [Mention("T", 0, 7, "Aspirin"), Mention("A", 8, 15, "aspirin")]),
    Document("2", "No chemicals", "None here.", []),
    Document("3", "Overlap", "methyl ether test", [Mention("A", 0, 6, "methyl"), Mention("A", 0, 12, "methyl ether")]),
    Document("4", "ATP", "ATP and NaCl.", [Mention("T", 0, 3, "ATP"), Mention("A", 0, 3, "ATP"), Mention("A", 8, 12, "NaCl")]),
]
DEV = [
    Document("10", "Benzene in water", "Toluene was found.", [Mention("T", 0, 7, "Benzene"), Mention("A", 0, 7, "Toluene")]),
    Document("11", "Plain", "Nothing.", []),
    Document("12", "Iron", "Iron levels.", [Mention("T", 0, 4, "Iron"), Mention("A", 0, 4, "Iron")]),
]


@pytest.fixture
def corpus(monkeypatch):
    def fake_load(split, source="official", gold=True):
        docs = {"training": TRAIN, "development": DEV, "evaluation": DEV}[split]
        if not gold:
            return [Document(d.pmid, d.title, d.abstract, []) for d in docs]
        return docs
    monkeypatch.setattr(direct, "load", fake_load)


class FakeResponses:
    """対象の文書の正解をタグにして返す。fail に入れた PMID は例外、incomplete に入れた PMID は未完了を返す。"""

    def __init__(self, fail=(), incomplete=()):
        self.fail, self.incomplete, self.calls = set(fail), set(incomplete), []

    async def create(self, **kw):
        self.calls.append(kw)
        target = json.loads(kw["input"][-1]["content"])
        doc = next(d for d in DEV if d.title == target["title"])
        if doc.pmid in self.fail:
            raise TimeoutError("boom")
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100,
                                input_tokens_details=SimpleNamespace(cached_tokens=800),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=40))
        status = "incomplete" if doc.pmid in self.incomplete else "completed"
        return SimpleNamespace(id=f"resp-{doc.pmid}", model="gpt-6-luna-2026", status=status,
                               incomplete_details=SimpleNamespace(reason="max_output_tokens") if status == "incomplete" else None,
                               usage=usage, output_text=direct.as_answer(doc))


def client(**kw):
    return SimpleNamespace(responses=FakeResponses(**kw))


def test_select_examples_is_deterministic_and_skips_overlapping_documents():
    a = direct.select_examples(TRAIN, 3, seed=1)
    assert [d.pmid for d in a] == [d.pmid for d in direct.select_examples(TRAIN, 3, seed=1)]
    assert "3" not in {d.pmid for d in a}
    with pytest.raises(ValueError):
        direct.select_examples(TRAIN, 4, seed=0)  # 重なりのある 1 件を除くと 3 件しかない


def test_messages_contain_examples_then_target_only():
    ex = direct.select_examples(TRAIN, 2, seed=0)
    msgs = direct.messages(ex, DEV[0])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert any("<chem>" in m["content"] for m in msgs if m["role"] == "assistant")  # 例には正解のタグが入る
    assert json.loads(msgs[-1]["content"])["title"] == DEV[0].title
    assert "<chem>" not in msgs[-1]["content"]  # 対象の文書には正解を入れない


def test_instructions_include_guidelines_without_attribution_comment():
    text = direct.instructions()
    assert "{guidelines}" not in text and "chemical entity mention" in text
    assert "<!--" not in text


def test_run_on_development_reproduces_gold_and_records_calls(corpus, tmp_path):
    cfg = direct.Config("development", "gpt-6-luna", "medium", k=2)
    c = client()
    out = asyncio.run(direct.run(cfg, c, out_root=tmp_path))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["documents"] == 3 and summary["failed"] == 0
    assert summary["models_in_responses"] == ["gpt-6-luna-2026"]
    assert summary["cost_usd"] > 0
    kw = c.responses.calls[0]
    assert kw["reasoning"] == {"effort": "medium"} and kw["text"]["format"]["strict"] is True
    assert kw["store"] is False
    pred = (out / "predictions.tsv").read_text()
    assert "10\tT:0:7" in pred and "12\tA:0:4" in pred
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert all("output_text" not in r for r in calls)  # 本文は公開用の記録に入れない
    assert (out / "private" / "raw.jsonl").exists()


def test_failures_are_recorded_not_silently_empty(corpus, tmp_path):
    cfg = direct.Config("development", "gpt-6-luna", "medium", k=2)
    out = asyncio.run(direct.run(cfg, client(fail={"11"}, incomplete={"12"}), out_root=tmp_path))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failed"] == 2
    assert summary["failed_reasons"] == {"TimeoutError": 1, "max_output_tokens": 1}
    with pytest.raises(RuntimeError, match="失敗した文書"):
        direct.score_run(out)
    # もう一度走らせると、失敗した文書だけを呼び直して埋める
    c = client()
    asyncio.run(direct.run(cfg, c, out_root=tmp_path))
    assert len(c.responses.calls) == 2
    assert json.loads((out / "summary.json").read_text())["failed"] == 0


def test_max_cost_stops_new_calls(corpus, tmp_path):
    cfg = direct.Config("development", "gpt-6-luna", "medium", k=2)
    c = client()
    out = asyncio.run(direct.run(cfg, c, concurrency=1, max_cost=0.0, out_root=tmp_path))
    assert len(c.responses.calls) == 1
    assert json.loads((out / "summary.json").read_text())["failed_reasons"] == {"not-run": 2}


def test_evaluation_requires_frozen_config(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "FROZEN", tmp_path / "frozen" / "direct.json")
    cfg = direct.Config("evaluation", "gpt-6-luna", "medium", k=2)
    with pytest.raises(RuntimeError, match="凍結"):
        asyncio.run(direct.run(cfg, client(), out_root=tmp_path))


def test_frozen_config_must_match(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "FROZEN", tmp_path / "frozen" / "direct.json")
    direct.freeze(k=2, seed=0)
    cfg = direct.Config("evaluation", "gpt-6-luna", "medium", k=2, seed=1)  # 凍結と違う seed
    with pytest.raises(RuntimeError, match="凍結した内容と違う"):
        direct.check_frozen(cfg, direct.select_examples(TRAIN, 2, 1))


def test_unsupported_effort_is_rejected(corpus, tmp_path):
    cfg = direct.Config("development", "gpt-6.1-sol", "none", k=2)
    with pytest.raises(ValueError, match="effort"):
        asyncio.run(direct.run(cfg, client(), out_root=tmp_path))


def test_call_record_cost():
    r = CallRecord("x", "gpt-6.1-sol", "medium", status="completed", input_tokens=1_000_000, cached_tokens=500_000,
                   output_tokens=100_000)
    assert r.cost_usd() == pytest.approx(0.5 * 2.0 + 0.5 * 0.1 + 0.1 * 10.0)
