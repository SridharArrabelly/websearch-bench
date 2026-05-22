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
    search_calls: int | None = None
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
            fmt(self.search_calls),
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
    """Count web_search_call items in an OpenAI/Foundry Responses object."""
    output = getattr(response, "output", None) or []
    return sum(1 for item in output if getattr(item, "type", None) == "web_search_call")


def usage_from_agent_framework(result: Any) -> dict[str, int | None]:
    """Best-effort token extraction from agent_framework AgentRunResponse."""
    for attr in ("usage_details", "usage", "token_usage"):
        usage = getattr(result, attr, None)
        if usage is None:
            continue
        try:
            d = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        except Exception:
            continue
        return {
            "input_tokens": d.get("input_tokens") or d.get("prompt_tokens"),
            "output_tokens": d.get("output_tokens") or d.get("completion_tokens"),
            "total_tokens": d.get("total_tokens"),
        }
    return {}
