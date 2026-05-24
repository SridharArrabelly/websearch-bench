"""Shared constants and helpers used by every backend.

The whole point of this repo is to compare token consumption and cost across
SDK surfaces. That comparison is only meaningful when the query, model, search
settings, and instructions are identical — so everything backends share lives
here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Load .env before reading any module-level env vars so MODEL / MODEL_FAST /
# OPENAI_MODEL pick up the user's overrides at import time. Backends still
# call load_dotenv(override=True) inside run() for their own per-run vars.
load_dotenv()

# ---------------------------------------------------------------------------
# Shared workload — change here, apply everywhere.
# ---------------------------------------------------------------------------

SHARED_QUERY: str = (
    "What are the current individual income tax brackets in South Africa for "
    "the 2025/2026 tax year?"
)

SHARED_INSTRUCTIONS: str = (
    "You are a research assistant for a South African audience. "
    "You MUST answer using ONLY information returned by the web_search tool. "
    "If the search tool returns no relevant results, reply: "
    "'I could not find this in the configured sources.' "
    "Every factual claim must be followed by a numbered citation [n] and a "
    "Sources list containing only URLs returned by the tool."
)

# Model. Override via MODEL env var. Keep the same model across backends.
MODEL: str = os.getenv("MODEL", "gpt-5.1")

# A non-reasoning model used by the *-fast WebSearchTool variant to test
# OpenAI's "non-reasoning web search" path (1 search, no fan-out).
# Override via MODEL_FAST env var. Must be deployed in your Foundry project.
MODEL_FAST: str = os.getenv("MODEL_FAST", "gpt-4.1-mini")

# The OpenAI direct backend can only use OpenAI-hosted models; choose a
# comparable one with OPENAI_MODEL.
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.1")

# Web-search settings. Keep identical across backends.
USER_COUNTRY: str = "ZA"
USER_CITY: str = "Johannesburg"
USER_REGION: str = "Gauteng"
ALLOWED_DOMAINS: list[str] = ["www.sars.gov.za"]
SEARCH_CONTEXT_SIZE: str = "medium"  # one of: "low" | "medium" | "high"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Normalized per-run metrics so every backend reports the same shape."""

    backend: str
    model: str
    input_tokens: int | None = None
    # Cached portion of input_tokens (billed at the cached_input rate). On
    # Azure OpenAI / Foundry this comes from
    # ``usage.input_tokens_details.cached_tokens`` and is also surfaced on the
    # App Insights span as ``gen_ai.usage.cached_tokens``.
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Number of times the model invoked the web-search tool (counted from the
    # response output items). Bills against the model-side tool charge.
    web_search_calls: int | None = None
    # Number of tool *responses* fed back into the model. For Foundry's
    # server-side ``web.run`` this is the better proxy for actual Bing queries
    # because one model-level web_search_call can fan out to many Bing
    # transactions inside the tool. Equals web_search_calls when no fan-out.
    bing_queries: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    answer_chars: int | None = None
    notes: str | None = None
    # Full answer text — included in the HTML report, excluded from CSV.
    answer: str | None = None
    # Foundry/OpenAI Responses API ``resp_…`` id (when available). Used by
    # ``compare.py`` to do a deferred App Insights reconcile after all
    # backends have run, giving telemetry time to ingest.
    response_id: str | None = None

    def as_row(self) -> list[str]:
        """Cells for the terminal rich.Table — does NOT include `answer`."""

        def fmt(v: Any, suffix: str = "") -> str:
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.4f}{suffix}" if suffix == " USD" else f"{v:.2f}{suffix}"
            return f"{v}{suffix}"

        return [
            self.backend,
            self.model,
            fmt(self.input_tokens),
            fmt(self.cached_input_tokens),
            fmt(self.output_tokens),
            fmt(self.total_tokens),
            fmt(self.web_search_calls),
            fmt(self.bing_queries),
            fmt(self.latency_s, " s"),
            fmt(self.cost_usd, " USD"),
            fmt(self.answer_chars, " chars"),
            self.notes or "",
        ]


def print_metrics(metrics: RunMetrics, console: Console | None = None) -> None:
    """Pretty-print a single run's metrics block.

    Layout:
        - Sectioned key/value table titled "Usage — <backend>"
        - notes line (if any)
        - response_id footer (if any)
        - Answer panel (if any)
    """
    console = console or Console()

    def _num(v: int | None) -> str:
        return "—" if v is None else f"{v:,}"

    def _pct(part: int | None, whole: int | None) -> str:
        if not part or not whole:
            return ""
        return f"  [dim]({part / whole:.0%} of input)[/dim]"

    def _money(v: float | None) -> str:
        return "—" if v is None else f"${v:,.4f}"

    def _secs(v: float | None) -> str:
        return "—" if v is None else f"{v:.2f} s"

    table = Table(
        title=f"[bold]Usage — {metrics.backend}[/bold]",
        title_justify="left",
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("metric", style="dim", no_wrap=True)
    table.add_column("value", justify="right", no_wrap=True)

    def section(label: str) -> None:
        table.add_row(Text(label, style="bold cyan"), "")

    def row(k: str, v: str) -> None:
        table.add_row(f"  {k}", v)

    section("Identity")
    row("backend", metrics.backend)
    row("model", metrics.model)

    section("Tokens")
    row("input", _num(metrics.input_tokens))
    row("  cached", f"{_num(metrics.cached_input_tokens)}{_pct(metrics.cached_input_tokens, metrics.input_tokens)}")
    row("output", _num(metrics.output_tokens))
    row("total", _num(metrics.total_tokens))

    section("Web search")
    row("web_search_calls", _num(metrics.web_search_calls))
    row("bing_queries", _num(metrics.bing_queries))

    section("Performance")
    row("latency", _secs(metrics.latency_s))
    row("cost", _money(metrics.cost_usd))

    section("Output")
    row("answer_chars", _num(metrics.answer_chars))

    console.print(table)

    if metrics.notes:
        console.print(Text.from_markup(f"[yellow]notes:[/yellow] {metrics.notes}"))
    if metrics.response_id:
        console.print(Text.from_markup(f"[dim]response_id: {metrics.response_id}[/dim]"))

    if metrics.answer:
        console.print(
            Panel(
                metrics.answer,
                title="Answer",
                title_align="left",
                border_style="dim",
                padding=(0, 1),
            )
        )
    console.print()


class Timer:
    """Context manager that records wall-clock seconds in ``.elapsed``."""

    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Debug dump — write a raw response payload to disk for offline inspection.
# Enable with WEBSEARCH_BENCH_DEBUG=1 (or set DEBUG_DIR explicitly).
# ---------------------------------------------------------------------------


def debug_dump(backend: str, payload: Any) -> str | None:
    """Dump a backend response to ``./debug/<backend>-<ts>.json``.

    On by default — this is a benchmark tool, the per-run debug payload is
    cheap and lets you verify counts (web_search_calls, bing_queries) against
    the App Insights span. Opt out with ``WEBSEARCH_BENCH_DEBUG=0``.

    Returns the file path it wrote, or ``None`` if dumping is disabled or the
    payload couldn't be serialized. We attempt ``.model_dump()`` first
    (pydantic Responses object), then fall back to ``vars()`` or ``str()``.
    """
    flag = os.getenv("WEBSEARCH_BENCH_DEBUG")
    if flag is not None and flag.strip().lower() in ("0", "false", "no", "off", ""):
        return None
    import json
    from pathlib import Path

    debug_dir = Path(os.getenv("DEBUG_DIR", "debug")).resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = debug_dir / f"{backend}-{ts}.json"

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return {k: _serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        try:
            return str(obj)
        except Exception:
            return None

    try:
        path.write_text(json.dumps(_serialize(payload), indent=2, default=str), encoding="utf-8")
        return str(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Usage extraction helpers
# ---------------------------------------------------------------------------


def usage_from_openai_response(response: Any) -> dict[str, int | None]:
    """Pull token counts out of an openai.responses.Response (or Foundry's).

    Also extracts cached input tokens from ``input_tokens_details.cached_tokens``
    (Responses API) or ``prompt_tokens_details.cached_tokens`` (chat
    completions style). The same number is surfaced on the Foundry App
    Insights span as ``gen_ai.usage.cached_tokens``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    try:
        d = usage.model_dump()
    except AttributeError:
        d = dict(usage)
    cached = None
    for k in ("input_tokens_details", "prompt_tokens_details"):
        details = d.get(k)
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if cached is not None:
                break
    if cached is None:
        cached = d.get("cached_tokens") or d.get("cached_input_tokens")
    return {
        "input_tokens": d.get("input_tokens") or d.get("prompt_tokens"),
        "cached_input_tokens": cached,
        "output_tokens": d.get("output_tokens") or d.get("completion_tokens"),
        "total_tokens": d.get("total_tokens"),
    }


def count_search_calls_in_openai_output(response: Any) -> int:
    """Backwards-compatible alias for ``count_web_search_calls_in_openai_output``."""
    return count_web_search_calls_in_openai_output(response)


def count_web_search_calls_in_openai_output(response: Any) -> int:
    """Count ``web_search_call`` items in an OpenAI/Foundry Responses object.

    This is the OpenAI-style ``web_search`` tool only (Foundry's
    ``WebSearchTool`` / OpenAI Responses native ``web_search``). The legacy
    Foundry ``BingGroundingTool`` is **not** a web_search call — its items
    use ``type="bing_grounding_call"`` and are counted by
    :func:`count_bing_queries_in_openai_output` instead.
    """
    output = getattr(response, "output", None) or []
    return sum(1 for item in output if getattr(item, "type", None) == "web_search_call")


def count_bing_queries_in_openai_output(response: Any) -> int | None:
    """Estimate the number of actual Bing search queries issued.

    .. warning::
       For the Foundry-hosted ``WebSearchTool`` (foundry-ws-bing / foundry-ws-
       bingcustom) this number is a **lower bound**. Foundry's grounding pipeline
       fans the tool call out into multiple Bing transactions server-side and
       only exposes a summarized ``action.queries`` list on each
       ``web_search_call`` item — the App Insights ``chat`` span on the
       Foundry account is the only ground-truth source for true billable Bing
       calls (one ``tool_call_response`` message per Bing transaction).
       Example: a response containing 2 web_search_call items with 3 entries
       in their ``action.queries`` arrays was observed driving **23**
       ``tool_call_response`` messages in App Insights. Use this column as a
       directional signal; reconcile cost against the Foundry/App Insights
       chat span for exact billing.

    Ground truth (from a real Foundry response dump): each ``web_search_call``
    item in ``response.output`` carries an ``action.queries`` list. Bing is
    billed once per query the tool actually issues; ``action.queries`` is the
    list of queries the *model* asked for, which the tool may then expand.
    The bench reports the **sum** across web_search_call items.

    Example real dump (foundry-ws-bing, single user prompt):
        output[0].type = "web_search_call"   action.queries = ["calculator: 1+1"]
        output[1].type = "web_search_call"   action.queries = ["tax tables …", "individual income tax …"]
        => 3 model-requested queries (App Insights showed 23 actual Bing hits)

    Strategy: walk every ``web_search_call`` and sum the largest list-shaped
    payload we can find (``queries`` / ``sub_queries`` / ``search_queries`` /
    ``results`` / ``sources`` / ``citations``). Falls back to 1 per call if
    none of those lists are present. As an additional signal we also count
    ``url_citation`` annotations on the assistant message; if that is larger
    than what we summed from the calls (which can happen when the SDK
    collapses the calls), we return that instead.

    Returns ``None`` if the response has no output array at all.
    """
    output = getattr(response, "output", None)
    if output is None:
        return None

    list_attrs = (
        "queries", "sub_queries", "search_queries",
        "results", "sources", "citations", "search_results",
    )

    def _list_len(holder: Any, attr: str) -> int:
        if holder is None:
            return 0
        val = holder.get(attr) if isinstance(holder, dict) else getattr(holder, attr, None)
        return len(val) if isinstance(val, list) else 0

    # A. Sum the per-call query/result counts.
    sum_queries = 0
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type not in (
            "web_search_call",
            "bing_grounding_call",
            "bing_custom_search_call",
        ):
            continue
        # BingGroundingTool / BingCustomSearchPreviewTool: each *_call item ==
        # exactly one Bing API call regardless of how many strings appear in
        # action.queries (verified against App Insights: 1 remote_functions.*
        # dependency span per item). Don't sum action.queries here or we
        # inflate the billable count.
        if item_type in ("bing_grounding_call", "bing_custom_search_call"):
            sum_queries += 1
            continue
        action = getattr(item, "action", None)
        sub = 0
        for attr in list_attrs:
            sub = max(sub, _list_len(item, attr), _list_len(action, attr))
        sum_queries += sub if sub else 1

    # B. url_citation annotations on the assistant message (fallback signal
    # for when the WebSearchTool SDK collapses fan-out). Skip this for
    # bing_grounding-only responses: citation count there reflects sources
    # cited from a single Bing call, not separate Bing transactions.
    has_web_search_call = any(
        getattr(i, "type", None) == "web_search_call" for i in output
    )
    if not has_web_search_call:
        return sum_queries

    citation_count = 0
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                atype = ann.get("type") if isinstance(ann, dict) else getattr(ann, "type", None)
                if atype in ("url_citation", "file_citation"):
                    citation_count += 1

    return max(sum_queries, citation_count)


# Item types in the OpenAI/Foundry Responses ``output`` array that represent
# a tool invocation. Any item whose ``type`` ends in ``_call`` is a tool call;
# the explicit list documents what we currently know about.
_OPENAI_TOOL_CALL_TYPES = {
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "image_generation_call",
    "computer_call",
    "function_call",
    "local_shell_call",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
}


def count_tool_calls_in_openai_output(response: Any) -> int:
    """Count *all* tool invocations in an OpenAI/Foundry Responses object.

    For the current bench this equals ``count_web_search_calls_in_openai_output``
    because web_search is the only tool we attach. The two diverge once you
    add function/MCP/code-interpreter tools.
    """
    output = getattr(response, "output", None) or []
    count = 0
    for item in output:
        t = getattr(item, "type", None) or ""
        if t in _OPENAI_TOOL_CALL_TYPES or t.endswith("_call"):
            count += 1
    return count


def count_search_calls_in_agent_response(result: Any) -> int | None:
    """Backwards-compatible alias for ``count_web_search_calls_in_agent_response``."""
    return count_web_search_calls_in_agent_response(result)


def count_bing_queries_in_agent_response(result: Any) -> int | None:
    """Count Bing query results returned to the model via the AF response.

    Probes multiple signals and returns the max:
      A.  Number of ``search_tool_result`` Content items (one per Bing
          transaction handed back to the model).
      B.  Number of ``url_citation`` annotations on any text content (same
          idea as the OpenAI Responses variant).
      C.  Falls back to ``count_web_search_calls_in_agent_response``.
    """
    messages = getattr(result, "messages", None)
    if not messages:
        return None
    results = 0
    citations = 0
    for msg in messages:
        for content in getattr(msg, "contents", None) or []:
            ctype = getattr(content, "type", None)
            if ctype == "search_tool_result":
                results += 1
            for ann in getattr(content, "annotations", None) or []:
                atype = ann.get("type") if isinstance(ann, dict) else getattr(ann, "type", None)
                if atype in ("url_citation", "file_citation"):
                    citations += 1
    fallback = count_web_search_calls_in_agent_response(result) or 0
    return max(results, citations, fallback)


def count_web_search_calls_in_agent_response(result: Any) -> int | None:
    """Count web-search tool invocations in an agent_framework ``AgentResponse``.

    Walks ``result.messages[*].contents[*]`` and counts ``Content`` items whose
    ``type`` is ``"search_tool_call"`` (Bing / web search). Generic
    ``function_call`` contents whose ``name`` contains "search" are also
    counted to handle providers that expose web search via a function tool.
    Returns ``None`` if the response has no messages at all.
    """
    messages = getattr(result, "messages", None)
    if not messages:
        return None
    count = 0
    for msg in messages:
        for content in getattr(msg, "contents", None) or []:
            ctype = getattr(content, "type", None)
            if ctype == "search_tool_call":
                count += 1
            elif ctype == "function_call":
                name = (getattr(content, "name", "") or "").lower()
                if "search" in name:
                    count += 1
    return count


# agent_framework Content types that represent a tool invocation (anything
# that triggered remote/sdk work). Used for the generic ``tool_calls`` metric.
_AF_TOOL_CALL_TYPES = {
    "function_call",
    "search_tool_call",
    "code_interpreter_tool_call",
    "image_generation_tool_call",
    "mcp_server_tool_call",
    "shell_tool_call",
}


def count_tool_calls_in_agent_response(result: Any) -> int | None:
    """Count *all* tool invocations across all messages of an AF response.

    Equals ``count_web_search_calls_in_agent_response`` when web_search is the
    only attached tool; will diverge once you add function/MCP tools.
    """
    messages = getattr(result, "messages", None)
    if not messages:
        return None
    count = 0
    for msg in messages:
        for content in getattr(msg, "contents", None) or []:
            ctype = getattr(content, "type", None) or ""
            if ctype in _AF_TOOL_CALL_TYPES or ctype.endswith("_tool_call"):
                count += 1
    return count


def usage_from_agent_framework(result: Any) -> dict[str, int | None]:
    """Token extraction from an agent_framework ``AgentResponse``.

    AF stores totals on ``response.usage_details`` (a ``UsageDetails`` TypedDict
    with ``input_token_count``/``output_token_count``/``total_token_count``).
    Per-turn usage is also attached to each ``Message`` for streaming/tool flows,
    so we sum across messages when the top-level totals aren't populated.
    """

    def _normalize(d: dict[str, Any]) -> dict[str, int | None]:
        # AF surfaces cached tokens via additional_properties in some flavors,
        # or via input_token_details.cached_tokens. Probe both.
        cached = d.get("cached_input_tokens") or d.get("cached_tokens")
        if cached is None:
            details = d.get("input_token_details") or d.get("input_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens")
        return {
            "input_tokens": d.get("input_token_count") or d.get("input_tokens") or d.get("prompt_tokens"),
            "cached_input_tokens": cached,
            "output_tokens": d.get("output_token_count") or d.get("output_tokens") or d.get("completion_tokens"),
            "total_tokens": d.get("total_token_count") or d.get("total_tokens"),
        }

    def _as_dict(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            try:
                return usage.model_dump()
            except Exception:
                return None
        try:
            return dict(usage)
        except Exception:
            return None

    for attr in ("usage_details", "usage", "token_usage"):
        d = _as_dict(getattr(result, attr, None))
        if d:
            norm = _normalize(d)
            if any(v is not None for v in norm.values()):
                return norm

    # Fallback: sum usage_details across messages.
    totals: dict[str, int] = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    saw_any = False
    for msg in getattr(result, "messages", []) or []:
        d = _as_dict(getattr(msg, "usage_details", None))
        if not d:
            continue
        n = _normalize(d)
        for k in totals:
            if n.get(k) is not None:
                totals[k] += int(n[k])
                saw_any = True
    return totals if saw_any else {}
