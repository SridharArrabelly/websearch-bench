"""Shared constants and helpers used by every backend.

The whole point of this repo is to compare token consumption and cost across
SDK surfaces. That comparison is only meaningful when the query, model, search
settings, and instructions are identical — so everything backends share lives
here.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

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
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Number of times the model invoked the web-search tool. This is what
    # Bing / OpenAI bill against ("transactions" / "calls").
    web_search_calls: int | None = None
    # Total tool invocations across ALL tools (web_search, function calls,
    # MCP, code interpreter, ...). For the current setup this equals
    # ``web_search_calls`` because we only attach web_search.
    tool_calls: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    answer_chars: int | None = None
    notes: str | None = None
    # Full answer text — included in the HTML report, excluded from CSV.
    answer: str | None = None

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
            fmt(self.output_tokens),
            fmt(self.total_tokens),
            fmt(self.web_search_calls),
            fmt(self.tool_calls),
            fmt(self.latency_s, " s"),
            fmt(self.cost_usd, " USD"),
            fmt(self.answer_chars, " chars"),
            self.notes or "",
        ]


def print_metrics(metrics: RunMetrics, console: Console | None = None) -> None:
    """Pretty-print a single run's metrics block."""
    console = console or Console()
    table = Table(title=f"Usage — {metrics.backend}", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in asdict(metrics).items():
        table.add_row(k, "—" if v is None else str(v))
    console.print(table)


class Timer:
    """Context manager that records wall-clock seconds in ``.elapsed``."""

    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Usage extraction helpers
# ---------------------------------------------------------------------------


def usage_from_openai_response(response: Any) -> dict[str, int | None]:
    """Pull token counts out of an openai.responses.Response (or Foundry's)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    try:
        d = usage.model_dump()
    except AttributeError:
        d = dict(usage)
    return {
        "input_tokens": d.get("input_tokens"),
        "output_tokens": d.get("output_tokens"),
        "total_tokens": d.get("total_tokens"),
    }


def count_search_calls_in_openai_output(response: Any) -> int:
    """Backwards-compatible alias for ``count_web_search_calls_in_openai_output``."""
    return count_web_search_calls_in_openai_output(response)


def count_web_search_calls_in_openai_output(response: Any) -> int:
    """Count ``web_search_call`` items in an OpenAI/Foundry Responses object."""
    output = getattr(response, "output", None) or []
    return sum(1 for item in output if getattr(item, "type", None) == "web_search_call")


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
        return {
            "input_tokens": d.get("input_token_count") or d.get("input_tokens") or d.get("prompt_tokens"),
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
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
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
