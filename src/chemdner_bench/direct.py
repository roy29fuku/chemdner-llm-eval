"""条件 a: 答えそのものを LLM に聞く（zero-shot・インラインタグ）。

最もナイーブな使い方として、アブストラクト 1 件ごとに、固定の指示（課題の定義と出力の形式）だけを与えて解かせる。
例は与えず、プロンプトは調整しない。原文に `<chem>` タグを入れた全文を JSON で返させ、位置に直して、採点器（cem.py）の
形式で予測を書き出す。

test の隔離（docs/experiment-protocol.md）:
- development は出力の形式が崩れないことの確認だけに使う
- test で走らせる前に `freeze` で指示・課題の定義・形式の hash を frozen/direct.json に書き、commit する。
  test の実行は、凍結した内容と今のファイルが一致し、凍結のファイルが commit 済みのときだけ通す
- test の入力は本文だけを読む（corpus.load(..., gold=False)）

    uv run python -m chemdner_bench.direct run --split development --limit 100
    uv run python -m chemdner_bench.direct freeze
    uv run python -m chemdner_bench.direct run --split evaluation --rep 0
    uv run python -m chemdner_bench.direct score runs/direct/<run>
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from chemdner_bench import cem, inline_tags, llm
from chemdner_bench.corpus import Document, gold_spans, load

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTIONS = ROOT / "prompts" / "direct" / "instructions.md"
GUIDELINES = ROOT / "task" / "guidelines.md"
FROZEN = ROOT / "frozen" / "direct.json"
RUNS = ROOT / "runs" / "direct"

SCHEMA = {
    "title": "tagged_abstract",
    "type": "object",
    "properties": {"title": {"type": "string"}, "abstract": {"type": "string"}},
    "required": ["title", "abstract"],
    "additionalProperties": False,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instructions() -> str:
    guidelines = GUIDELINES.read_text(encoding="utf-8")
    if guidelines.startswith("<!--"):
        guidelines = guidelines[guidelines.index("-->") + 3:].lstrip()  # 出典の注記はモデルに見せない
    return INSTRUCTIONS.read_text(encoding="utf-8").replace("{guidelines}", guidelines.strip())


def messages(doc: Document) -> list[dict]:
    """対象の文書だけを渡す（zero-shot）。"""
    return [{"role": "user", "content": json.dumps({"title": doc.title, "abstract": doc.abstract}, ensure_ascii=False)}]


def predict(doc: Document, output_text: str) -> tuple[set[tuple[str, int, int]], dict]:
    """モデルの出力（JSON）を位置に直す。読めないときは例外を投げる（呼び出し側で失敗として記録する）。"""
    data = json.loads(output_text)
    spans: set[tuple[str, int, int]] = set()
    stats = {"violations": [], "text_matches": True, "dropped": 0, "tagged": 0}
    for section, field in (("T", "title"), ("A", "abstract")):
        parsed = inline_tags.parse(data[field])
        aligned = inline_tags.align(parsed, doc.text(section))
        stats["violations"] += [f"{section}:{v}" for v in parsed.violations]
        stats["text_matches"] &= aligned.text_matches
        stats["dropped"] += aligned.dropped
        stats["tagged"] += len(parsed.spans)
        spans |= {(section, s, e) for s, e in aligned.spans}
    return spans, stats


@dataclass
class Config:
    split: str
    model: str = "gpt-6-luna"
    effort: str = "medium"
    rep: int = 0
    limit: int | None = None
    max_output_tokens: int = 16000

    def name(self) -> str:
        limit = f"-n{self.limit}" if self.limit else ""
        return f"{self.split}-{self.model}-{self.effort}-r{self.rep}{limit}"


def fingerprint() -> dict:
    return {
        "instructions_sha256": sha256_text(INSTRUCTIONS.read_text(encoding="utf-8")),
        "guidelines_sha256": sha256_text(GUIDELINES.read_text(encoding="utf-8")),
        "schema_sha256": sha256_text(json.dumps(SCHEMA, sort_keys=True)),
    }


def freeze() -> Path:
    data = {**fingerprint(), "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return FROZEN


def check_frozen() -> None:
    """test で走らせてよいか。凍結の内容と今の状態が一致し、凍結のファイルが commit 済みであること。"""
    if not FROZEN.exists():
        raise RuntimeError("test の前に `direct freeze` で指示を凍結し、frozen/direct.json を commit する")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    current = fingerprint()
    diff = [key for key in current if frozen.get(key) != current[key]]
    if diff:
        raise RuntimeError(f"凍結した内容と違う: {diff}")
    rel = FROZEN.relative_to(ROOT)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel), "prompts", "task"], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise RuntimeError("frozen/direct.json・prompts/・task/ に commit していない変更がある")


async def run(cfg: Config, client, concurrency: int = 16, max_cost: float | None = None, out_root: Path = RUNS) -> Path:
    llm.check_model(cfg.model, cfg.effort)
    if cfg.split == "evaluation":
        check_frozen()
    elif cfg.split != "development":
        raise ValueError("a を走らせる分割は development か evaluation")
    docs = load(cfg.split, gold=False)  # 入力は本文だけ
    if cfg.limit:
        docs = sorted(docs, key=lambda d: d.pmid)[:cfg.limit]
    out = out_root / cfg.name()
    (out / "private").mkdir(parents=True, exist_ok=True)
    calls_path = out / "calls.jsonl"
    done = {}
    if calls_path.exists():  # 続きから走らせる。成功した文書は飛ばす
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["ok"] and r.get("parse_error") is None:
                done[r["key"]] = r
    (out / "config.json").write_text(json.dumps({
        **cfg.__dict__, **fingerprint(), "sdk": _sdk_version(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    system = instructions()
    cache_key = sha256_text(system)[:32]
    sem = asyncio.Semaphore(concurrency)
    spent = sum(r["cost_usd"] for r in done.values())
    stop = asyncio.Event()

    async def one(doc: Document) -> None:
        nonlocal spent
        if doc.pmid in done or stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            text, rec = await llm.respond(client, key=doc.pmid, model=cfg.model, effort=cfg.effort,
                                          instructions=system, input=messages(doc), json_schema=SCHEMA,
                                          max_output_tokens=cfg.max_output_tokens, prompt_cache_key=cache_key)
        row = rec.to_dict()
        row["parse_error"] = None
        if text is not None:
            try:
                spans, stats = predict(doc, text)
                row.update(stats)
                row["spans"] = sorted(spans)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                row["parse_error"] = f"{type(e).__name__}: {e}"[:300]
        with open(calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out / "private" / "raw.jsonl", "a", encoding="utf-8") as f:  # 本文を含むので公開しない
            f.write(json.dumps({"key": doc.pmid, "output_text": text}, ensure_ascii=False) + "\n")
        spent += row["cost_usd"]
        if max_cost is not None and spent > max_cost:
            stop.set()

    await asyncio.gather(*(one(d) for d in docs))
    write_outputs(out, [d.pmid for d in docs])
    return out


def write_outputs(out: Path, pmids: list[str]) -> dict:
    """calls.jsonl から予測と集計を作る。同じ文書は最後の記録を使う。"""
    latest = {}
    for line in (out / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        latest[r["key"]] = r
    pred = {k: {tuple(s) for s in r.get("spans", [])} for k, r in latest.items() if r["ok"] and r["parse_error"] is None}
    cem.write_tsv(out / "predictions.tsv", pred)
    failed = sorted(set(pmids) - set(pred))
    summary = {
        "documents": len(pmids),
        "succeeded": len(pred),
        "failed": len(failed),
        "failed_reasons": _count(_reason(latest.get(k)) for k in failed),
        "text_mismatch_documents": sum(1 for k in pred if not latest[k].get("text_matches", True)),
        "dropped_mentions": sum(latest[k].get("dropped", 0) for k in pred),
        "format_violations": sum(len(latest[k].get("violations", [])) for k in pred),
        "models_in_responses": sorted({r["model"] for r in latest.values() if r.get("model")}),
        "input_tokens": sum(r["input_tokens"] for r in latest.values()),
        "cached_tokens": sum(r["cached_tokens"] for r in latest.values()),
        "output_tokens": sum(r["output_tokens"] for r in latest.values()),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in latest.values()),
        "cost_usd": round(sum(r["cost_usd"] for r in latest.values()), 4),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _reason(r: dict | None) -> str:
    if r is None:
        return "not-run"  # --max-cost で止めた・途中で中断した
    for key in ("error", "incomplete_reason", "parse_error"):
        if r.get(key):
            return str(r[key]).split(":")[0]
    return str(r.get("status"))


def _count(items) -> dict:
    counts: dict = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return counts


def _sdk_version() -> str:
    try:
        import openai
        return openai.__version__
    except ImportError:
        return "unknown"


def score_run(out: Path) -> dict:
    """予測を採点する（主: official、副: all）。失敗した文書が残っていれば採点しない。"""
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary["failed"]:
        raise RuntimeError(f"失敗した文書が {summary['failed']} 件ある。`run` をもう一度実行して埋めてから採点する")
    docs = load(cfg["split"])
    if cfg.get("limit"):
        docs = sorted(docs, key=lambda d: d.pmid)[:cfg["limit"]]
    gold = gold_spans(docs)
    pred = cem.read_tsv(out / "predictions.tsv")
    official = cem.official_pmids(gold)
    all_pmids = [d.pmid for d in docs]
    result = {}
    for scope, pmids, drop in (("official", official, True), ("all", all_pmids, False)):
        s = cem.score(gold, pred, pmids, drop_unknown=drop)
        f1, lo, hi = cem.bootstrap(cem.per_document(gold, pred, pmids))
        result[scope] = {"documents": len(pmids), "tp": s.tp, "fp": s.fp, "fn": s.fn,
                         "precision": s.precision, "recall": s.recall, "f1": f1, "f1_ci95": [lo, hi]}
    (out / "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chemdner_bench.direct")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="条件 a を走らせる")
    p_run.add_argument("--split", choices=("development", "evaluation"), required=True)
    p_run.add_argument("--model", choices=llm.MODELS, default="gpt-6-luna")
    p_run.add_argument("--effort", default="medium")
    p_run.add_argument("--rep", type=int, default=0, help="反復の番号（出力のディレクトリ名に入る）")
    p_run.add_argument("--limit", type=int, help="先頭（PMID の順）から n 件だけ走らせる（試行用）")
    p_run.add_argument("--concurrency", type=int, default=16)
    p_run.add_argument("--max-cost", type=float, help="この額（米ドル）を超えたら新しい呼び出しを止める")
    p_run.add_argument("--max-output-tokens", type=int, default=16000)
    sub.add_parser("freeze", help="test の前に指示を凍結する")
    p_score = sub.add_parser("score", help="走らせた結果を採点する")
    p_score.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(f"wrote {freeze()}（commit してから test を走らせる）")
    elif args.command == "score":
        print(json.dumps(score_run(args.run_dir), indent=2))
    else:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        load_dotenv(ROOT / ".env")
        cfg = Config(args.split, args.model, args.effort, args.rep, args.limit, args.max_output_tokens)
        out = asyncio.run(run(cfg, AsyncOpenAI(max_retries=6), args.concurrency, args.max_cost))
        print(json.dumps(json.loads((out / "summary.json").read_text()), indent=2, ensure_ascii=False))
        print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
