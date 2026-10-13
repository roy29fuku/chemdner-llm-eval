"""条件 b のエージェントの通信を記録し、絞る mitmproxy の追加スクリプト。

- 全通信を /logs/flows.jsonl に記録する（時刻・方法・宛先・経路・状態・大きさ・遮断の理由）。本文は記録しない
- 許可リストに無い宛先は遮断する
- 許可した宛先でも、正解の置き場（CHEMDNER の公開ミラーなど）・データセット・既存の化学物質 NER のモデルやツールは
  経路（パス）で遮断する
- OpenAI の API（Codex 本体）の使用量を応答から数え、費用が上限（環境変数 MAX_COST_USD）を超えたら以後の API の
  呼び出しを止める

SSH（22 番）への CONNECT は中身を見ずに通す（mitmproxy の ignore_hosts で設定する）。
Codex は Responses API を WebSocket で使うので、使用量は WebSocket のメッセージからも数える。
実験の GCP プロジェクトの ID（環境変数 PROJECT_ID。名前に chemdner を含む）は、照合の前に URL から除く。
"""

import json
import os
import re
import time

from mitmproxy import http

LOG = os.environ.get("FLOW_LOG", "/logs/flows.jsonl")
PROJECT_ID = os.environ.get("PROJECT_ID", "")
USAGE_LOG = os.environ.get("USAGE_LOG", "/logs/openai_usage.jsonl")
MAX_COST_USD = float(os.environ.get("MAX_COST_USD", "10"))
# 入力・キャッシュされた入力・出力（100 万トークンあたりの米ドル）。chemdner_bench/llm.py の PRICING と同じ
PRICES = {"gpt-6-luna": (0.1, 0.01, 0.5), "gpt-6.1-sol": (2.0, 0.1, 10.0), "gpt-6-astra": (10.0, 1.0, 50.0),
          "gpt-5.2": (1.75, 0.175, 14.0)}
UNKNOWN_PRICE = (10.0, 1.0, 50.0)  # 表に無いモデルは高い側で数える

ALLOW_HOSTS = [
    r"api\.openai\.com",                        # Codex 本体
    r"(.+\.)?googleapis\.com", r"accounts\.google\.com", r"(.+\.)?gstatic\.com",  # gcloud・Terraform・GCS
    r"cloud\.google\.com", r"dl\.google\.com", r"packages\.cloud\.google\.com",
    r"releases\.hashicorp\.com", r"registry\.terraform\.io", r"checkpoint-api\.hashicorp\.com",
    r"pypi\.org", r"files\.pythonhosted\.org", r"download\.pytorch\.org",
    r"huggingface\.co", r"(.+\.)?hf\.co", r"cdn-lfs.*\.huggingface\.co", r"(.+\.)?xethub\.hf\.co",
    r"github\.com", r"raw\.githubusercontent\.com", r"objects\.githubusercontent\.com", r"codeload\.github\.com",
    r"(deb|security)\.debian\.org", r"(.+\.)?ubuntu\.com", r"developer\.download\.nvidia\.com",
    r"registry\.npmjs\.org",
]
# 許可した宛先の中で遮断するもの（宛先と経路を合わせた URL に当てる）
DENY_URL = [
    (r"generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com", "他の LLM の API"),
    (r"huggingface\.co/(api/)?datasets", "Hugging Face のデータセット"),
    (r"datasets-server\.huggingface\.co", "Hugging Face のデータセット"),
    (r"chemdner|chem[-_]?dner|bc4chemd|bc5cdr|biocreative|chemprot|nlm[-_]?chem", "CHEMDNER や関連コーパス・それで学習したモデル"),
    (r"chem[^/]*[-_]ner|ner[-_][^/]*chem|chemical[-_]?ner|chem[-_]?ner", "化学物質の NER のモデル"),
    (r"chemdataextractor|chemlistem|chemspot|oscar4|tmchem|chemtagger|en[-_]ner[-_]bc5cdr", "化学物質の NER のツール"),
    (r"lingluodlut|spyysalo/chemdner|roy29fuku", "正解の置き場・本実験の repo"),
]

_allow = [re.compile(rf"^{p}$", re.I) for p in ALLOW_HOSTS]
_deny = [(re.compile(p, re.I), why) for p, why in DENY_URL]
_spent = 0.0


def _write(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _blocked(flow: http.HTTPFlow) -> str | None:
    host = flow.request.pretty_host
    if not any(p.match(host) for p in _allow):
        return "許可リストに無い宛先"
    url = f"{host}{flow.request.path}"
    if PROJECT_ID:
        url = url.replace(PROJECT_ID, "<project>")  # 実験のプロジェクト自体は遮断しない
    for pattern, why in _deny:
        if pattern.search(url):
            return why
    if host == "api.openai.com" and _spent > MAX_COST_USD:
        return f"API の費用の上限（${MAX_COST_USD}）を超えた"
    return None


def request(flow: http.HTTPFlow) -> None:
    why = _blocked(flow)
    flow.metadata["blocked"] = why
    if why:
        flow.response = http.Response.make(403, f"blocked by experiment proxy: {why}\n".encode(), {"content-type": "text/plain"})


def _usage_from(body: bytes) -> list[dict]:
    """Responses API の応答（JSON か SSE）から usage を取り出す。"""
    text = body.decode("utf-8", "replace")
    found = []
    for line in text.splitlines():
        line = line[5:].strip() if line.startswith("data:") else line.strip()
        if '"usage"' not in line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = data.get("response", data)
        if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
            found.append({"model": resp.get("model"), "usage": resp["usage"]})
    return found


def _account(found: list[dict], flow: http.HTTPFlow, channel: str) -> None:
    global _spent
    for u in found:
        usage = u["usage"]
        inp = usage.get("input_tokens", 0) or 0
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        model = u["model"] or ""
        price = next((v for k, v in PRICES.items() if model.startswith(k)), UNKNOWN_PRICE)
        cost = ((inp - cached) * price[0] + cached * price[1] + out * price[2]) / 1e6
        _spent += cost
        _write(USAGE_LOG, {"t": time.time(), "channel": channel, "model": u["model"], "input_tokens": inp,
                           "cached_tokens": cached, "output_tokens": out, "cost_usd": round(cost, 6),
                           "total_usd": round(_spent, 4), "user_agent": flow.request.headers.get("user-agent", "")})


def websocket_message(flow: http.HTTPFlow) -> None:
    """Codex の WebSocket。サーバーからのメッセージの usage を数え、上限を超えたら接続を切る。"""
    if flow.request.pretty_host != "api.openai.com" or flow.websocket is None:
        return
    msg = flow.websocket.messages[-1]
    if not msg.from_client:
        _account(_usage_from(msg.content if isinstance(msg.content, bytes) else msg.content.encode()), flow, "websocket")
        if _spent > MAX_COST_USD:
            _write(LOG, {"t": time.time(), "method": "WS", "host": "api.openai.com", "path": flow.request.path[:300],
                         "status": None, "bytes": 0, "user_agent": "", "blocked": f"API の費用の上限（${MAX_COST_USD}）を超えた"})
            flow.kill()


def response(flow: http.HTTPFlow) -> None:
    if flow.request.pretty_host == "api.openai.com" and flow.response is not None and not flow.metadata.get("blocked"):
        _account(_usage_from(flow.response.content or b""), flow, "http")
    _write(LOG, {
        "t": time.time(),
        "method": flow.request.method,
        "host": flow.request.pretty_host,
        "path": flow.request.path[:300],
        "status": flow.response.status_code if flow.response else None,
        "bytes": len(flow.response.content or b"") if flow.response else 0,
        "user_agent": flow.request.headers.get("user-agent", "")[:120],
        "blocked": flow.metadata.get("blocked"),
    })
