"""OpenAI の Responses API の呼び出しと、その記録。

1 回の呼び出しごとに、要求したモデル・応答のモデル・応答 ID・状態・未完了の理由・エラー・トークン数・所要時間を
CallRecord に残す。失敗（例外・未完了・空応答）は黙って空の予測にせず、記録して呼び出し側に返す。
再試行は SDK に任せる（429・5xx・接続エラーを max_retries 回まで指数的に待って繰り返す）。
"""

import time
from dataclasses import asdict, dataclass

# 100 万トークンあたりの米ドル（入力・キャッシュされた入力・出力）。出典: https://developers.openai.com/api/docs/models/
# の各モデルのページ（2026-10-14 に確認）。推論のトークンは出力として課金される
PRICING = {
    "gpt-6-luna": (0.1, 0.01, 0.5),
    "gpt-6.1-sol": (2.0, 0.1, 10.0),
    "gpt-6-astra": (10.0, 1.0, 50.0),
    "gpt-5.2": (1.75, 0.175, 14.0),  # 2026-01 の実験で使ったモデル（比較用）
}
MODELS = tuple(PRICING)
# effort の選択肢（同じ出典）。Luna だけ none を受け付ける
EFFORTS = {
    "gpt-6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-6.1-sol": ("low", "medium", "high", "xhigh", "max"),
    "gpt-6-astra": ("low", "medium", "high", "xhigh", "max"),
    "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),  # 既定は none
}
# 各モデルの既定の effort（最もナイーブな使い方では既定を使う）
DEFAULT_EFFORT = {"gpt-6-luna": "medium", "gpt-6.1-sol": "medium", "gpt-6-astra": "medium", "gpt-5.2": "none"}


@dataclass
class CallRecord:
    key: str  # 呼び出しの識別子（文書の PMID など）
    model_requested: str
    effort: str
    model: str | None = None  # 応答に入っていたモデル名
    response_id: str | None = None
    status: str | None = None  # completed / incomplete / failed
    incomplete_reason: str | None = None
    error: str | None = None  # 例外の型とメッセージ
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    started_at: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == "completed"

    def cost_usd(self) -> float:
        price_in, price_cached, price_out = PRICING.get(self.model_requested, (0.0, 0.0, 0.0))
        uncached = self.input_tokens - self.cached_tokens
        return (uncached * price_in + self.cached_tokens * price_cached + self.output_tokens * price_out) / 1e6

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok, "cost_usd": round(self.cost_usd(), 6)}


def check_model(model: str, effort: str) -> None:
    if model not in PRICING:
        raise ValueError(f"モデルは {MODELS} のどれか: {model}")
    if effort not in EFFORTS[model]:
        raise ValueError(f"{model} の effort は {EFFORTS[model]} のどれか: {effort}")


async def respond(client, *, key: str, model: str, effort: str, instructions: str, input: list[dict],
                  json_schema: dict, max_output_tokens: int, prompt_cache_key: str | None = None) -> tuple[str | None, CallRecord]:
    """1 回呼ぶ。成功なら (出力の文字列, 記録)、失敗なら (None, 記録) を返す。"""
    from datetime import datetime, timezone

    rec = CallRecord(key=key, model_requested=model, effort=effort,
                     started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    t0 = time.monotonic()
    try:
        resp = await client.responses.create(
            model=model,
            instructions=instructions,
            input=input,
            reasoning={"effort": effort},
            text={"format": {"type": "json_schema", "name": json_schema["title"], "schema": json_schema, "strict": True}},
            max_output_tokens=max_output_tokens,
            prompt_cache_key=prompt_cache_key,
            store=False,
        )
    except Exception as e:  # noqa: BLE001（再試行の後も失敗したものは型ごと記録する）
        rec.error = f"{type(e).__name__}: {e}"[:500]
        rec.latency_s = round(time.monotonic() - t0, 3)
        return None, rec
    rec.latency_s = round(time.monotonic() - t0, 3)
    rec.model = getattr(resp, "model", None)
    rec.response_id = getattr(resp, "id", None)
    rec.status = getattr(resp, "status", None)
    details = getattr(resp, "incomplete_details", None)
    rec.incomplete_reason = getattr(details, "reason", None) if details else None
    usage = getattr(resp, "usage", None)
    if usage is not None:
        rec.input_tokens = usage.input_tokens or 0
        rec.output_tokens = usage.output_tokens or 0
        rec.cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        rec.reasoning_tokens = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    text = getattr(resp, "output_text", None)
    if not rec.ok:
        return None, rec
    if not text:
        rec.error = "EmptyOutput"
        return None, rec
    return text, rec
