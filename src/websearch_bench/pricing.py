"""Pricing constants and cost estimator.

Verified against vendor pricing pages on 2025/2026 (see links below).
Values are still env-overridable for what-if scenarios.

- Azure AI Foundry / Azure OpenAI model pricing:
  https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
- Grounding with Bing Search (Foundry connection):
  https://www.microsoft.com/bing/apis/grounding-pricing  →  $35 / 1,000 queries
- Grounding with Bing Custom Search (legacy SKU $14/1K is being retired in 2025;
  new SKU on the same page is $35/1,000).

Every tool is billed *per call*. A single user prompt can trigger 0..N
calls to the web-search tool — the model decides. We report this as
``web_search_calls`` per run and use ``cost = calls * price_per_1000 / 1000``.

The WebSearchTool family (``foundry-ws-*``, ``agentfx-*``) is documented as
billing at the same Grounding-with-Bing rate as the legacy direct tools
(``foundry-bing-grounding*``) — see https://learn.microsoft.com/azure/foundry/openai/how-to/web-search.
The charge appears on the Foundry / Cognitive Services account bill rather
than on the user's ``Microsoft.Bing/accounts`` resource.
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
    "gpt-4.1":      {"input": 0.00200, "cached_input": 0.000500, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.000400, "cached_input": 0.000100, "output": 0.001600},
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

BING_GROUNDING_USD_PER_1K: float = _env_float("BING_GROUNDING_USD_PER_1K", 35.0)
BING_CUSTOM_USD_PER_1K: float = _env_float("BING_CUSTOM_USD_PER_1K", 35.0)


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


def tool_cost(
    backend: str,
    web_search_calls: int | None,
    bing_queries: int | None = None,
) -> float:
    """USD cost for the per-call charge of the web-search tool of this backend.

    Per Microsoft docs (https://learn.microsoft.com/azure/foundry/openai/how-to/web-search):

        Web Search uses Grounding with Bing Search and/or Grounding with
        Bing Custom Search. […] Use of Grounding with Bing Search and
        Grounding with Bing Custom Search incurs costs. Search actions
        incur tool call costs.

    Two distinct billing surfaces — both at the same Grounding-with-Bing
    rate ($35/1K), just metered against different resources:

    1. **Legacy direct tools** — ``BingGroundingTool`` /
       ``BingCustomSearchPreviewTool`` (the ``foundry-bing-grounding*``
       backends). These call the user's *own* ``Microsoft.Bing/accounts``
       resource. Each tool call increments ``TotalCalls`` on that resource
       (verifiable via ``bing_usage.py``). Pass ``bing_queries`` here.

    2. **WebSearchTool / Responses ``web_search``** — every other backend
       (``foundry-ws-*``, ``agentfx-*``). These route through
       Microsoft-managed Bing infra (not the user's Bing resource — its
       ``TotalCalls`` does *not* increment). Still Grounding with Bing
       under the hood and still billed at the Grounding-with-Bing rate,
       but the user is charged per *outer* ``web_search_call`` action,
       not per internal Bing fan-out (the fan-out happens server-side and
       isn't separately metered to the caller — see the caveat in
       README.md). The charge appears on the Foundry / Cognitive Services
       account bill as a "Grounding with Bing Search" line item. Verify
       with ``cost_lookup.py``.
    """
    # Legacy direct tools — billed against user's Bing resource per Bing call.
    if backend.startswith("foundry-bing-grounding-custom"):
        calls = bing_queries if bing_queries is not None else (web_search_calls or 0)
        return calls * BING_CUSTOM_USD_PER_1K / 1000.0
    if backend.startswith("foundry-bing-grounding") or backend.startswith("foundry-bing"):
        calls = bing_queries if bing_queries is not None else (web_search_calls or 0)
        return calls * BING_GROUNDING_USD_PER_1K / 1000.0

    # WebSearchTool family on Azure Foundry — Grounding with Bing rate,
    # priced per outer web_search_call (not per internal Bing fan-out).
    # foundry-ws-bingcustom routes to the Custom-Search SKU.
    if backend.startswith("foundry-ws-bingcustom"):
        calls = web_search_calls or 0
        return calls * BING_CUSTOM_USD_PER_1K / 1000.0
    if backend.startswith("foundry-ws-") or backend.startswith("agentfx"):
        calls = web_search_calls or 0
        return calls * BING_GROUNDING_USD_PER_1K / 1000.0

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
) -> float:
    """USD estimate for a single run. = model token cost + per-call tool cost.

    Tool cost routing (see ``tool_cost`` for details):

    * BingGroundingTool backends bill against the user's Bing resource, so
      ``bing_queries`` (the real Bing transaction count from Foundry's
      ``web.run`` span) is the right billable quantity.
    * WebSearchTool backends bill at the same Grounding-with-Bing rate,
      but per *outer* tool call — not per internal Bing fan-out.
      ``web_search_calls`` is the right quantity. ``bing_queries`` for these
      backends is observed-only (telemetry), not billable.
    """
    return (
        model_token_cost(model, input_tokens, output_tokens, cached_input_tokens)
        + tool_cost(backend, web_search_calls, bing_queries)
    )
