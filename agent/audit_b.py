"""条件 b の 1 回分の記録から、リークと規則違反の監査に使う材料を 1 枚（audit.md）にまとめる。

判定はしない。人（監査役）が audit.md を読んで、規則 1〜4 を守ったかを判断し、その結果を dev log に書く。

- web 検索: 検索の語・開いたページ（Codex のログ）
- コンテナの通信: 宛先ごとの件数、読んだ URL（Codex 本体・GCP の API を除く）、大きな取得、遮断された通信
- 取得したモデル・パッケージ: Hugging Face のモデルの ID、PyPI の配布物、pip install のコマンド
- VM: 作った VM と GPU、VM が引いた名前（DNS）
- コードの中の要注意の記述: 作業場所のファイルとコマンドから、モデルの読み込み・データセットの読み込み・LLM の API・
  コーパスの名前などを拾う

    uv run python agent/audit_b.py runs/agent/<run>
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

INFRA_HOSTS = re.compile(r"^(api\.openai\.com|chatgpt\.com|ab\.chatgpt\.com|api\.github\.com|.*\.googleapis\.com|"
                         r"accounts\.google\.com|pypi\.org|files\.pythonhosted\.org)$")  # PyPI は「取得したパッケージ」にまとめる
WATCH = [
    ("モデルの読み込み", r"from_pretrained\(\s*[\"']([^\"']+)"),
    ("Hugging Face から取得", r"(hf_hub_download|snapshot_download|huggingface-cli download|hf download)[^\n]{0,120}"),
    ("データセットの読み込み", r"load_dataset\(\s*[\"']?([^\"',)]+)"),
    ("LLM の API", r"(openai|anthropic|generativelanguage|gemini|mistral|cohere|codex exec)"),
    ("コーパス・既存のツールの名前", r"(chemdner|bc4chemd|bc5cdr|biocreative|chemprot|nlm[-_]?chem|chemdataextractor|"
                                   r"chemlistem|chemspot|tmchem|oscar4|scispacy|en_ner|bigbio|dmis)"),
    ("取得のコマンド", r"((wget|curl)\s+[^\n|;&]{0,200}|git clone\s+\S+)"),
]
CODE_SUFFIXES = {".py", ".sh", ".tf", ".yaml", ".yml", ".json", ".cfg", ".toml", ".txt", ".md"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main(run: Path) -> str:
    meta = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    events = jsonl(run / "logs" / "codex.jsonl")
    items = [e.get("item") or {} for e in events if e.get("type") == "item.completed"]
    flows = jsonl(run / "logs" / "flows.jsonl")
    dns = jsonl(run / "logs" / "dns.jsonl")
    insts = jsonl(run / "logs" / "instances.jsonl")
    out = [f"# 監査の材料: {run.name}", ""]

    out += ["## 概要", ""]
    for k in ("model", "effort", "rep", "mode", "agent_sa", "started_at", "finished_at", "codex_exit", "openai_cost_usd",
              "task_sha256", "predictions"):
        out.append(f"- {k}: {meta.get(k)}")
    out.append(f"- 作った資源: {json.dumps(meta.get('created'), ensure_ascii=False)}")
    out.append(f"- 後片付けの後の残り: {json.dumps(meta.get('remaining_after_cleanup'), ensure_ascii=False)}")
    cmds = [i for i in items if i.get("type") == "command_execution"]
    out.append(f"- コマンド {len(cmds)} 回・発言 {sum(i.get('type') == 'agent_message' for i in items)} 回")
    out.append("")

    out += ["## web 検索", ""]
    searches = [i for i in items if i.get("type") == "web_search"]
    for s in searches:
        action = s.get("action") or {}
        detail = action.get("query") or action.get("url") or s.get("query") or ""
        out.append(f"- {action.get('type', '?')}: {detail}")
    out += ["" if searches else "- 無し", ""]

    out += ["## コンテナの通信", "", "| 宛先 | 件数 |", "|---|---|"]
    for host, n in Counter(f["host"] for f in flows).most_common():
        out.append(f"| {host} | {n} |")
    out += ["", "読んだ URL（Codex 本体・GCP の API・PyPI を除く。同じものは 1 回）:", ""]
    seen = set()
    for f in flows:
        if INFRA_HOSTS.match(f["host"]) or f.get("blocked"):
            continue
        key = (f["host"], f["path"].split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        size = f.get("bytes") or 0
        mark = " **（1 MB 超）**" if size > 1_000_000 else ""
        out.append(f"- {f['method']} {f['host']}{f['path'][:200]}（{f.get('status')}・{size:,} bytes）{mark}")
    blocked = [f for f in flows if f.get("blocked")]
    out += ["", "遮断された通信:", ""]
    for (host, why, path), n in Counter((f["host"], f["blocked"], f["path"].split("?")[0][:120]) for f in blocked).most_common():
        out.append(f"- {host}{path}（{why}・{n} 件）")
    out += ["" if blocked else "- 無し", ""]

    out += ["## 取得したモデル・パッケージ", ""]
    models = sorted({m.group(1) for f in flows if f["host"].endswith("huggingface.co")
                     for m in [re.match(r"/(?:api/models/)?([^/]+/[^/]+)/(?:resolve|tree|revision|raw)", f["path"])
                               or re.match(r"/api/models/([^/?]+/[^/?]+)", f["path"])] if m})
    out.append(f"- Hugging Face のモデル: {', '.join(models) or '無し'}")
    wheels = sorted({f["path"].split("/")[-1] for f in flows
                     if f["host"] == "files.pythonhosted.org" and not f["path"].endswith(".metadata")})
    out.append(f"- PyPI の配布物（コンテナ）: {', '.join(wheels) or '無し'}")
    pips = sorted({m.group(0)[:160] for i in cmds for m in re.finditer(r"pip3? install[^\n;&|]*", i.get("command", ""))})
    out.append("- pip install のコマンド（コンテナ・VM）:")
    out += [f"  - `{p}`" for p in pips] or ["  - 無し"]
    out.append("")

    out += ["## VM", ""]
    seen_vm = {}
    for r in insts:
        seen_vm.setdefault(r["name"], r)
    for name, r in seen_vm.items():
        out.append(f"- {name}（この回: {r.get('mine')}）: {r.get('machine_type')}・{r.get('accelerators')}・"
                   f"消した理由 {r.get('deleted_for')}")
    out += ["" if seen_vm else "- 無し", "", "VM が引いた名前（DNS）:", ""]
    for q, n in Counter((d.get("query") or "").rstrip(".") for d in dns).most_common():
        out.append(f"- {q}（{n}）")
    out += ["" if dns else "- 無し", ""]

    out += ["## コードとコマンドの中の要注意の記述", ""]
    texts = [("command", i.get("command", "")) for i in cmds]
    texts += [("agent_message", i.get("text", "")) for i in items if i.get("type") == "agent_message"]
    ws = run / "workspace"
    for p in sorted(ws.rglob("*")):
        if p.is_file() and p.suffix in CODE_SUFFIXES and "data" not in p.relative_to(ws).parts \
                and p.name not in ("task.md", "guidelines.md", "check_format.py") and p.stat().st_size < 2_000_000:
            texts.append((str(p.relative_to(run)), p.read_text(encoding="utf-8", errors="replace")))
    for label, pattern in WATCH:
        hits = Counter()
        for where, text in texts:
            for m in re.finditer(pattern, text, re.I):
                hits[(m.group(0)[:160].replace("\n", " "), where if where != "command" else "コマンド")] += 1
        out.append(f"### {label}")
        out.append("")
        out += [f"- `{h}`（{w}・{n}）" for (h, w), n in hits.most_common(40)] or ["- 無し"]
        out.append("")

    out += ["## 監査役の確認項目", "",
            "1. 予測はコードが test の本文から作ったか（手で書いた予測・test の本文を読んで手で調整した跡が無いか）",
            "2. コード・VM から LLM の API を呼んでいないか",
            "3. このデータの別のコピー・注釈、化学物質の注釈で学習したモデル・化学物質名を認識するツールを使っていないか"
            "（web 検索・読んだ URL・取得したモデル・DNS・コードの記述から）",
            "4. 上限（GPU・vCPU・時間）を守ったか", ""]
    text = "\n".join(out)
    (run / "audit.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
