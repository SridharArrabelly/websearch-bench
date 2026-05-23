"""OpenAI Responses API + native ``web_search`` tool benchmark backend."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console

from websearch_bench.pricing import estimate_cost
from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    OPENAI_MODEL,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    count_bing_queries_in_openai_output,
    debug_dump,
    count_tool_calls_in_openai_output,
    count_web_search_calls_in_openai_output,
    print_metrics,
    usage_from_openai_response,
)

BACKEND_NAME = "openai-web-search"
REQUIRED_ENV: tuple[str, ...] = ("OPENAI_API_KEY",)
# This backend is opt-in because it bills against an OpenAI subscription
# (web_search is $10/1k calls + standard token rates). Most contributors only
# care about the Azure/Foundry surfaces, so we skip it unless explicitly
# enabled.
_ENABLE_VAR = "ENABLE_OPENAI_WEB_SEARCH"

console = Console()


def enabled() -> tuple[bool, str]:
    """Skip unless the user explicitly opts in.

    Returns ``(True, "")`` to run, or ``(False, reason)`` to skip. The
    benchmark runner uses this hook to decide whether to invoke ``run()``.
    """
    load_dotenv(override=True)
    val = os.getenv(_ENABLE_VAR, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True, ""
    return False, f"set {_ENABLE_VAR}=1 to enable (requires paid OpenAI subscription)"


async def run() -> RunMetrics:
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    client = AsyncOpenAI(api_key=api_key)
    console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")

    with Timer() as t:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=SHARED_INSTRUCTIONS,
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": ALLOWED_DOMAINS},
                }
            ],
            include=["web_search_call.action.sources"],
            input=SHARED_QUERY,
        )

    console.print("\n[bold green]Agent:[/bold green]")
    console.print(response.output_text)

    console.print("\n[bold]Sources[/bold]")
    for item in response.output:
        if getattr(item, "type", None) == "web_search_call":
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None) if action else None
            if sources:
                for s in sources:
                    console.print(f"- {getattr(s, 'url', s)}")

    _dump = debug_dump(BACKEND_NAME, response)
    if _dump:
        console.print(f"[dim]Debug dump: {_dump}[/dim]")
    usage = usage_from_openai_response(response)
    answer = response.output_text or ""
    metrics = RunMetrics(
        backend=BACKEND_NAME,
        model=OPENAI_MODEL,
        input_tokens=usage.get("input_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        web_search_calls=count_web_search_calls_in_openai_output(response),
        bing_queries=count_bing_queries_in_openai_output(response),
        tool_calls=count_tool_calls_in_openai_output(response),
        latency_s=round(t.elapsed, 2),
        answer_chars=len(answer),
        answer=answer,
    )
    metrics.cost_usd = round(
        estimate_cost(
            backend=metrics.backend,
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            web_search_calls=metrics.web_search_calls,
            bing_queries=metrics.bing_queries,
            cached_input_tokens=metrics.cached_input_tokens,
        ),
        4,
    )
    print_metrics(metrics, console)
    return metrics


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
