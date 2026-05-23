"""Reconcile Foundry-hosted backend metrics against App Insights.

The Foundry ``web.run`` extension fans out into multiple Bing transactions
server-side. None of that is exposed in the Responses API or even as separate
spans in App Insights — the only place the truth lives is the chat span's
``gen_ai.input.messages`` array, where every Bing transaction shows up as a
``role="tool"`` message.

This module queries the App Insights REST API for a given response_id and
returns the real ``tool_msgs`` count (plus the chat-span token counts as a
sanity check). The bench then overwrites ``bing_queries`` with that value
and recomputes ``cost_usd``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from websearch_bench.auth import make_credential


_APPID_RE = re.compile(r"ApplicationId=([0-9a-fA-F-]+)")
_APPINSIGHTS_SCOPE = "https://api.applicationinsights.io/.default"
_APPINSIGHTS_BASE = "https://api.applicationinsights.io/v1/apps"


@dataclass
class ChatSpanFacts:
    """Ground-truth metrics for one chat span (from App Insights)."""

    chat_span_id: str
    tool_msgs: int
    total_msgs: int
    in_tokens: int | None
    out_tokens: int | None
    cached_tokens: int | None


def _application_id_from_env() -> str | None:
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        return None
    m = _APPID_RE.search(conn)
    return m.group(1) if m else None


async def fetch_chat_span(
    response_id: str,
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 5.0,
    lookback_minutes: int = 15,
) -> ChatSpanFacts | None:
    """Poll App Insights for the chat span of ``response_id``.

    Returns ``None`` if no App Insights connection string is configured, or if
    the chat span doesn't show up within ``timeout_s`` (typical ingestion lag
    is 60-120s).
    """
    app_id = _application_id_from_env()
    if not app_id:
        return None

    # Two schemas live in the same App Insights:
    #
    #   Foundry server-side (foundry-ws-bing, foundry-ws-bingcustom):
    #     - tokens in customMeasurements
    #     - fan-out = role="tool" entries in gen_ai.input.messages
    #
    #   agent_framework client-side (agentfx-bing):
    #     - tokens in customDimensions (as strings)
    #     - no role="tool" entries; tool turns live in gen_ai.output.messages
    #       as type="search_tool_call"/"search_tool_result" parts. Foundry's
    #       hosted tool hides the real fan-out from this side, so the call
    #       count here equals web_search_calls (lower bound).
    #
    # We pick the richer span (more tool_msgs / search_tool_calls) and pull
    # tokens from whichever location is populated.
    kql = f"""
let respId = '{response_id}';
dependencies
| where timestamp > ago({lookback_minutes}m)
| where tostring(customDimensions["gen_ai.response.id"]) == respId
| where tostring(customDimensions["gen_ai.operation.name"]) == "chat"
| extend raw_in  = tostring(customDimensions["gen_ai.input.messages"])
| extend raw_out = tostring(customDimensions["gen_ai.output.messages"])
| extend tool_msgs_in  = array_length(extract_all(@'("role":\\s*"tool")', raw_in))
| extend tool_calls_out = array_length(extract_all(@'("type":\\s*"search_tool_call")', raw_out))
| extend tool_calls_bing = array_length(extract_all(@'("type":\\s*"bing_grounding")', raw_out))
| extend tool_msgs = max_of(coalesce(tool_msgs_in, 0), coalesce(tool_calls_out, 0), coalesce(tool_calls_bing, 0))
| extend total_msgs = array_length(parse_json(raw_in))
| extend in_meas  = toint(customMeasurements["gen_ai.usage.input_tokens"])
| extend out_meas = toint(customMeasurements["gen_ai.usage.output_tokens"])
| extend cac_meas = toint(customMeasurements["gen_ai.usage.cached_tokens"])
| extend in_dim   = toint(customDimensions["gen_ai.usage.input_tokens"])
| extend out_dim  = toint(customDimensions["gen_ai.usage.output_tokens"])
| extend cac_dim  = toint(customDimensions["gen_ai.usage.cached_tokens"])
| extend in_tokens     = coalesce(in_meas,  in_dim)
| extend out_tokens    = coalesce(out_meas, out_dim)
| extend cached_tokens = coalesce(cac_meas, cac_dim)
| top 1 by tool_msgs desc
| project chat_span_id = id, tool_msgs, total_msgs, in_tokens, out_tokens, cached_tokens
""".strip()

    url = f"{_APPINSIGHTS_BASE}/{app_id}/query"
    deadline = time.monotonic() + timeout_s

    async with make_credential() as credential:
        token = (await credential.get_token(_APPINSIGHTS_SCOPE)).token
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            best: ChatSpanFacts | None = None
            while time.monotonic() < deadline:
                async with session.post(url, headers=headers, json={"query": kql}) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        # 401/403 = auth; bail loudly, no point polling.
                        if resp.status in (401, 403):
                            raise RuntimeError(
                                f"App Insights returned {resp.status}: {body}\n"
                                "Grant your identity the 'Log Analytics Reader' or "
                                "'Monitoring Reader' role on the App Insights resource."
                            )
                        await asyncio.sleep(poll_interval_s)
                        continue

                    data = await resp.json()
                    facts = _parse_chat_facts(data)
                    if facts is not None:
                        # Track best seen; keep polling while best.tool_msgs==0
                        # in case a richer (e.g. Foundry server-side) span
                        # ingests after an initial client-side row.
                        if best is None or facts.tool_msgs > best.tool_msgs:
                            best = facts
                        if best.tool_msgs > 0:
                            return best

                await asyncio.sleep(poll_interval_s)

    return best


def _parse_chat_facts(payload: dict[str, Any]) -> ChatSpanFacts | None:
    tables = payload.get("tables") or []
    if not tables:
        return None
    primary = tables[0]
    rows = primary.get("rows") or []
    if not rows:
        return None
    cols = [c["name"] for c in primary.get("columns", [])]
    row = dict(zip(cols, rows[0], strict=False))
    try:
        return ChatSpanFacts(
            chat_span_id=str(row.get("chat_span_id") or ""),
            tool_msgs=int(row.get("tool_msgs") or 0),
            total_msgs=int(row.get("total_msgs") or 0),
            in_tokens=_maybe_int(row.get("in_tokens")),
            out_tokens=_maybe_int(row.get("out_tokens")),
            cached_tokens=_maybe_int(row.get("cached_tokens")),
        )
    except (TypeError, ValueError):
        return None


def _maybe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_RESP_ID_RE = re.compile(r"\bresp_[A-Za-z0-9]{20,}\b")


def find_response_id(obj: Any, *, depth: int = 6) -> str | None:
    """Walk an arbitrary object/dict tree and return the first ``resp_…`` id.

    Useful for Agent-Framework results that don't expose ``.id`` directly but
    embed the Foundry response id somewhere in their inner messages.
    """
    if depth < 0:
        return None
    if isinstance(obj, str):
        m = _RESP_ID_RE.search(obj)
        return m.group(0) if m else None
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_response_id(v, depth=depth - 1)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = find_response_id(v, depth=depth - 1)
            if found:
                return found
        return None
    for attr in ("response_id", "id"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            m = _RESP_ID_RE.search(v)
            if m:
                return m.group(0)
    for attr in ("messages", "raw_response", "raw", "data", "result"):
        v = getattr(obj, attr, None)
        if v is not None:
            found = find_response_id(v, depth=depth - 1)
            if found:
                return found
    return None



async def reconcile_metrics(
    metrics: Any,
    response_id: str | None,
    *,
    console: Any = None,
    timeout_s: float = 120.0,
) -> Any:
    """Pull the real tool_msgs count from App Insights and patch ``metrics``.

    For Foundry-hosted backends the Responses API hides the per-Bing-query
    fan-out; the chat span's ``gen_ai.input.messages`` array is the only
    ground truth. This helper queries it, overwrites ``bing_queries``, and
    recomputes ``cost_usd``. Safe to call when App Insights is not configured
    (returns ``metrics`` unchanged with a note).
    """
    from websearch_bench.pricing import estimate_cost  # local import: avoid cycle

    if not response_id:
        return metrics
    if console is not None:
        console.print(
            f"[dim]Reconciling bing_queries against App Insights for {response_id} …[/dim]"
        )
    try:
        facts = await fetch_chat_span(response_id, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        if console is not None:
            console.print(f"[yellow]App Insights reconcile skipped: {exc}[/yellow]")
        return metrics

    if facts is None:
        metrics.notes = (
            (metrics.notes + " | " if metrics.notes else "")
            + "App Insights not yet available — bing_queries is lower bound"
        )
        return metrics

    if facts.tool_msgs == 0:
        # Client-side span (e.g. agent_framework OTel): captures only the
        # initial system+user input, never the post-fan-out conversation.
        # Don't overwrite the lower-bound bing_queries — that would be worse
        # than not reconciling at all.
        metrics.notes = (
            (metrics.notes + " | " if metrics.notes else "")
            + "App Insights chat span has 0 tool msgs (client-side instrumentation) — bing_queries is lower bound"
        )
        return metrics

    metrics.bing_queries = facts.tool_msgs
    is_agentfx = metrics.backend.startswith("agentfx")
    if is_agentfx:
        # agent_framework's chat span only sees the model-level
        # `search_tool_call` (= web_search_calls). Foundry hides the actual
        # per-Bing-transaction fan-out from the client. To see the real
        # fan-out we'd need Foundry's server-side App Insights.
        metrics.notes = (
            f"bing_queries from agent_framework chat span (search_tool_calls={facts.tool_msgs}); "
            "actual Bing fan-out hidden by Foundry server-side"
        )
    else:
        metrics.notes = (
            f"bing_queries from Foundry App Insights chat span (tool_msgs={facts.tool_msgs})"
        )
    metrics.cost_usd = round(
        estimate_cost(
            backend=metrics.backend,
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cached_input_tokens=metrics.cached_input_tokens,
            web_search_calls=metrics.web_search_calls,
            bing_queries=metrics.bing_queries,
        ),
        4,
    )
    return metrics
