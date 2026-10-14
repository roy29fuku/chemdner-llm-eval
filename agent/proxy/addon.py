"""条件 b のエージェントの通信を記録し、絞る mitmproxy の追加スクリプト。

- 全通信を /logs/flows.jsonl に記録する（時刻・方法・宛先・経路・状態・大きさ・遮断の理由）。本文は記録しない
- 読むだけの通信（GET・HEAD・OPTIONS）はどの宛先にも通す（web 調査・パッケージ・事前学習済みモデルの取得のため）。
  書き込みを伴う通信（POST など）は許可リストの宛先にだけ通す
- 他の LLM の API の宛先は、方法を問わず遮断する
- 正解の置き場（CHEMDNER の公開ミラーなど）・データセット・既存の化学物質 NER のモデルやツールは、宛先と経路
  （クエリを除く）で遮断する。GCP の API（*.googleapis.com）には当てない（エージェントが自分の VM やバケットに
  chem-ner などの名前を付けると、その操作まで遮断してしまうため。2026-10-14 の試行 4 で起きた）。遮断できなかった
  取得は、記録から事後に監査する
- 費用の累計は使用量の記録から復元する（mitmproxy はこのファイルが変わると読み込み直し、変数が初期化されるため）
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

READ_METHODS = {"GET", "HEAD", "OPTIONS"}
# 書き込みを伴う通信を通す宛先
ALLOW_HOSTS = [
    r"api\.openai\.com",                        # Codex 本体
    r"chatgpt\.com",                             # Codex のモデル情報の取得だけを通す（_blocked で経路を絞る）
    r"(.+\.)?googleapis\.com", r"accounts\.google\.com", r"(.+\.)?gstatic\.com",  # gcloud・Terraform・GCS
    r"cloud\.google\.com", r"dl\.google\.com", r"packages\.cloud\.google\.com",
    r"releases\.hashicorp\.com", r"registry\.terraform\.io", r"checkpoint-api\.hashicorp\.com",
    r"pypi\.org", r"files\.pythonhosted\.org", r"download\.pytorch\.org",
    r"huggingface\.co", r"(.+\.)?hf\.co", r"cdn-lfs.*\.huggingface\.co", r"(.+\.)?xethub\.hf\.co",
    r"github\.com", r"raw\.githubusercontent\.com", r"objects\.githubusercontent\.com", r"codeload\.github\.com",
    r"(deb|security)\.debian\.org", r"(.+\.)?ubuntu\.com", r"developer\.download\.nvidia\.com",
    r"registry\.npmjs\.org",
]
# 方法を問わず遮断する宛先（コードから LLM を呼ぶ経路。OpenAI は Codex 本体が使うので、キーを渡さないことで防ぐ）
DENY_HOSTS = [
    r"api\.anthropic\.com", r"generativelanguage\.googleapis\.com", r"(.+-)?aiplatform\.googleapis\.com",
    r"api\.mistral\.ai", r"api\.cohere\.(com|ai)", r"api\.together\.(xyz|ai)", r"openrouter\.ai", r"api\.groq\.com",
    r"api\.deepseek\.com", r"api\.perplexity\.ai", r"api\.x\.ai", r"api\.fireworks\.ai", r"api\.replicate\.com",
    r"(api-inference|router)\.huggingface\.co", r"api\.moonshot\.(ai|cn)", r"dashscope\.aliyuncs\.com",
    r"(.+\.)?openai\.azure\.com", r"bedrock(-runtime)?\..*\.amazonaws\.com",
]
# 遮断する URL（宛先と経路を合わせ、クエリを除いたものに当てる）
DENY_URL = [
    (r"generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com", "他の LLM の API"),
    (r"huggingface\.co/(api/)?datasets", "Hugging Face のデータセット"),
    (r"datasets-server\.huggingface\.co", "Hugging Face のデータセット"),
    (r"chemdner|chem[-_]?dner|bc4chemd|bc5cdr|biocreative|chemprot|nlm[-_]?chem", "CHEMDNER や関連コーパス・それで学習したモデル"),
    (r"chem[^/]*[-_]ner|ner[-_][^/]*chem|chemical[-_]?ner|chem[-_]?ner", "化学物質の NER のモデル"),
    (r"chemdataextractor|chemlistem|chemspot|oscar4|tmchem|chemtagger|en[-_]ner[-_]bc5cdr", "化学物質の NER のツール"),
    (r"lingluodlut|spyysalo/chemdner|roy29fuku|bigbio", "正解の置き場・本実験の repo"),
    (r"dmis\.korea\.ac\.kr/.*datasets|nerdata|datasets\.tar\.gz", "BioBERT などが配るデータセット（BC4CHEMD を含む）"),
]

_allow = [re.compile(rf"^{p}$", re.I) for p in ALLOW_HOSTS]
_deny_hosts = [re.compile(rf"^{p}$", re.I) for p in DENY_HOSTS]
_gcp_api = re.compile(r"^(.+\.)?googleapis\.com$", re.I)  # DENY_URL を当てない（他の LLM の API は DENY_HOSTS で止める）
_deny = [(re.compile(p, re.I), why) for p, why in DENY_URL]



def _restore_spent() -> float:
    try:
        with open(USAGE_LOG, encoding="utf-8") as f:
            lines = f.read().splitlines()
        return float(json.loads(lines[-1])["total_usd"]) if lines else 0.0
    except (FileNotFoundError, ValueError, KeyError, IndexError):
        return 0.0


_spent = _restore_spent()


def _write(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _blocked(flow: http.HTTPFlow) -> str | None:
    host = flow.request.pretty_host
    if any(p.match(host) for p in _deny_hosts):
        return "他の LLM の API"
    if flow.request.method not in READ_METHODS and not any(p.match(host) for p in _allow):
        return "書き込みを伴う通信で、許可リストに無い宛先"
    if host == "chatgpt.com" and not (flow.request.method == "GET" and flow.request.path.startswith("/backend-api/codex/models")):
        return "Codex の付随的な通信（モデル情報の取得以外）"
    if _gcp_api.match(host):
        return None
    url = f"{host}{flow.request.path.split('?')[0]}"
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
