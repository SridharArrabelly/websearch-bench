"""Pricing constants and cost estimator.

Verified against vendor pricing pages on 2025/2026 (see links below).
Values are still env-overridable for what-if scenarios.

- Azure AI Foundry / Azure OpenAI model pricing:
  https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
- Grounding with Bing Search (Foundry connection):
  https://www.microsoft.com/bing/apis/grounding-pricing  →  $35 / 1,000 queries
- Grounding with Bing Custom Search (legacy SKU $14/1K is being retired in 2025;
  new SKU on the same page is $35/1,000).
- OpenAI Responses API + ``web_search`` tool:
  https://openai.com/api/pricing/   →  $10 / 1,000 calls (all models)
  (``web_search_preview`` for non-reasoning models is priced separately at
  $25-30 / 1,000; use OPENAI_WEB_SEARCH_USD_PER_1K to override.)

Every tool is billed *per call*. A single user prompt can trigger 0..N
calls to the web-search tool — the model decides. We report this as
``web_search_calls`` per run and use ``cost = calls * price_per_1000 / 1000``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Model token pricing (USD per 1,000 tokens). Verify before quoting.
#
# Source: https://azure.microsoft.com/pricing/details/azure-openai/ (Global
# Standard SKU, pay-as-you-go). Stated by the vendor in USD per 1M tokens; we
# store the equivalent per-1K values here so the math lines up with
# ``model_token_cost`` below.
#
#   gpt-5.1 : input $1.25/1M, cached input $0.125/1M, output $10/1M
#   gpt-4o  : input $2.50/1M, cached input $1.25/1M,  output $10/1M
# ---------------------------------------------------------------------------

MODEL_PRICING_PER_1K: dict[str, dict[str, float]] = {
    "gpt-5.1":      {"input": 0.00125, "cached_input": 0.000125, "output": 0.010},
    "gpt-5.1-mini": {"input": 0.00025, "cached_input": 0.000025, "output": 0.002},
    "gpt-4o":       {"input": 0.00250, "cached_input": 0.00125,  "output": 0.010},
    "gpt-4o-mini":  {"input": 0.000150, "cached_input": 0.000075, "output": 0.000600},
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Tool charges. Defaults express the vendor list price PER 1,000 CALLS so the
# magnitude matches what you see on the pricing page; we divide by 1000 below.
# Env vars override either the per-1000 (preferred) or the legacy per-call name.
# ---------------------------------------------------------------------------

BING_GROUNDING_USD_PER_1K: float = _env_float(
    "BING_GROUNDING_USD_PER_1K",
    _env_float("BING_GROUNDING_USD_PER_CALL", 0.035) * 1000.0
    if os.getenv("BING_GROUNDING_USD_PER_CALL") else 35.0,
)
BING_CUSTOM_USD_PER_1K: float = _env_float(
    "BING_CUSTOM_USD_PER_1K",
    _env_float("BING_CUSTOM_USD_PER_CALL", 0.035) * 1000.0
    if os.getenv("BING_CUSTOM_USD_PER_CALL") else 35.0,
)
OPENAI_WEB_SEARCH_USD_PER_1K: float = _env_float(
    "OPENAI_WEB_SEARCH_USD_PER_1K",
    _env_float("OPENAI_WEB_SEARCH_USD_PER_CALL", 0.010) * 1000.0
    if os.getenv("OPENAI_WEB_SEARCH_USD_PER_CALL") else 10.0,
)

# Kept for backward-compatibility (HTML report still shows these).
BING_GROUNDING_USD_PER_CALL: float = BING_GROUNDING_USD_PER_1K / 1000.0
BING_CUSTOM_USD_PER_CALL: float = BING_CUSTOM_USD_PER_1K / 1000.0
OPENAI_WEB_SEARCH_USD_PER_CALL: float = OPENAI_WEB_SEARCH_USD_PER_1K / 1000.0


def model_token_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None = None,
) -> float:
    """USD cost for the model tokens of a single call. Returns 0 if unknown.

    When ``cached_input_tokens`` is provided, that portion is billed at the
    ``cached_input`` rate and only the remainder is billed at the full
    ``input`` rate (Azure OpenAI / Foundry prompt caching).
    """
    rates = MODEL_PRICING_PER_1K.get(model)
    if not rates:
        return 0.0
    cost = 0.0
    cached = cached_input_tokens or 0
    if input_tokens:
        fresh = max(0, input_tokens - cached)
        cost += (fresh / 1000.0) * rates.get("input", 0.0)
        if cached:
            cached_rate = rates.get("cached_input", rates.get("input", 0.0))
            cost += (cached / 1000.0) * cached_rate
    if output_tokens:
        cost += (output_tokens / 1000.0) * rates.get("output", 0.0)
    return cost


def tool_cost(backend: str, web_search_calls: int | None) -> float:
    """USD cost for the per-call charge of the web-search tool of this backend.

    ``web_search_calls`` here means *billable* Bing/web-search transactions —
    pass ``bing_queries`` (when known) for an accurate Foundry bill, since
    Foundry's server-side web.run can fan out one model-level call into many
    Bing transactions.
    """
    calls = web_search_calls or 0
    if backend.startswith("foundry-bing-custom"):
        return calls * BING_CUSTOM_USD_PER_1K / 1000.0
    if backend.startswith("foundry-bing") or backend.startswith("agentfx"):
        return calls * BING_GROUNDING_USD_PER_1K / 1000.0
    if backend.startswith("openai-web-search"):
        return calls * OPENAI_WEB_SEARCH_USD_PER_1K / 1000.0
    return 0.0


def estimate_cost(
    *,
    backend: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None = None,
    web_search_calls: int | None = None,
    bing_queries: int | None = None,
    # Legacy alias — older callers passed ``search_calls``. Kept so existing
    # backends keep working until they're all migrated in one PR.
    search_calls: int | None = None,
) -> float:
    """USD estimate for a single run. = model token cost + per-call tool cost.

    Tool cost is billed against ``bing_queries`` when known (the real Bing
    transaction count from Foundry's web.run fan-out), otherwise against
    ``web_search_calls`` (the model-level call count).
    """
    if bing_queries is not None:
        calls = bing_queries
    elif web_search_calls is not None:
        calls = web_search_calls
    else:
        calls = search_calls
    return (
        model_token_cost(model, input_tokens, output_tokens, cached_input_tokens)
        + tool_cost(backend, calls)
    )
