"""Pricing constants and cost estimator.

IMPORTANT — values below are illustrative defaults. Pricing changes; please
verify against the official pages before quoting any number to a customer:

- Azure AI Foundry / Azure OpenAI model pricing:
  https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
- Grounding with Bing Search (Foundry connection) pricing:
  https://www.microsoft.com/bing/apis/grounding-pricing
- Grounding with Bing Custom Search pricing:
  https://www.microsoft.com/bing/apis/pricing
- OpenAI Responses API + web_search tool pricing:
  https://openai.com/api/pricing/

Override any value with an env var (e.g. ``BING_GROUNDING_USD_PER_CALL=0.04``).
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Model token pricing (USD per 1,000 tokens). Placeholders — REPLACE.
# ---------------------------------------------------------------------------

MODEL_PRICING_PER_1K: dict[str, dict[str, float]] = {
    "gpt-5.1": {"input": 0.005, "output": 0.015},
    "gpt-5.5": {"input": 0.010, "output": 0.030},
    "gpt-4o":  {"input": 0.0025, "output": 0.010},
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Per-call tool charges, USD. Override via env vars.
BING_GROUNDING_USD_PER_CALL: float = _env_float("BING_GROUNDING_USD_PER_CALL", 0.035)
BING_CUSTOM_USD_PER_CALL: float = _env_float("BING_CUSTOM_USD_PER_CALL", 0.025)
OPENAI_WEB_SEARCH_USD_PER_CALL: float = _env_float("OPENAI_WEB_SEARCH_USD_PER_CALL", 0.030)


def model_token_cost(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float:
    """USD cost for the model tokens of a single call. Returns 0 if unknown."""
    rates = MODEL_PRICING_PER_1K.get(model)
    if not rates:
        return 0.0
    cost = 0.0
    if input_tokens:
        cost += (input_tokens / 1000.0) * rates.get("input", 0.0)
    if output_tokens:
        cost += (output_tokens / 1000.0) * rates.get("output", 0.0)
    return cost


def estimate_cost(
    *,
    backend: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    search_calls: int | None,
) -> float:
    """USD estimate for a single run. Tool charges depend on the backend."""
    tokens = model_token_cost(model, input_tokens, output_tokens)
    calls = search_calls or 0
    if backend.startswith("foundry-bing-custom"):
        tool = calls * BING_CUSTOM_USD_PER_CALL
    elif backend.startswith("foundry-bing") or backend.startswith("agentfx"):
        tool = calls * BING_GROUNDING_USD_PER_CALL
    elif backend.startswith("openai-web-search"):
        tool = calls * OPENAI_WEB_SEARCH_USD_PER_CALL
    else:
        tool = 0.0
    return tokens + tool
