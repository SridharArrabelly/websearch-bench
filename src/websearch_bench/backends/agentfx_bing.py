"""Microsoft Agent Framework agent backed by Foundry's web-search tool."""

from __future__ import annotations

import asyncio
import logging
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.pricing import estimate_cost
from websearch_bench.appinsights import find_response_id, reconcile_metrics
from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    USER_COUNTRY,
    RunMetrics,
    Timer,
    count_bing_queries_in_agent_response,
    debug_dump,
    count_tool_calls_in_agent_response,
    count_web_search_calls_in_agent_response,
    print_metrics,
    usage_from_agent_framework,
)

BACKEND_NAME = "agentfx-bing"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT",)

console = Console()
logger = logging.getLogger(__name__)


async def run() -> RunMetrics:
    load_dotenv(override=True)

    async with DefaultAzureCredential() as cred:
        client = FoundryChatClient(
            project_endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=cred,
            model=MODEL,
        )
        web_search_tool = client.get_web_search_tool(
            user_location={"country": USER_COUNTRY},
            allowed_domains=ALLOWED_DOMAINS,
            search_context_size=SEARCH_CONTEXT_SIZE,
        )
        agent = Agent(
            client=client,
            name="WebSearchToolAgent",
            instructions=SHARED_INSTRUCTIONS,
            tools=[web_search_tool],
            description="Agent Framework + Foundry Bing benchmark backend.",
        )

        console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")
        with Timer() as t:
            result = await agent.run(SHARED_QUERY)

    console.print(f"\n[bold green]Agent:[/bold green] {result.text}")

    _dump = debug_dump(BACKEND_NAME, result)
    if _dump:
        console.print(f"[dim]Debug dump: {_dump}[/dim]")
    usage = usage_from_agent_framework(result)
    web_search_calls = count_web_search_calls_in_agent_response(result)
    bing_queries = count_bing_queries_in_agent_response(result)
    tool_calls = count_tool_calls_in_agent_response(result)
    metrics = RunMetrics(
        backend=BACKEND_NAME,
        model=MODEL,
        input_tokens=usage.get("input_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        web_search_calls=web_search_calls,
        bing_queries=bing_queries,
        tool_calls=tool_calls,
        latency_s=round(t.elapsed, 2),
        answer_chars=len(result.text or ""),
        answer=result.text or "",
        notes="bing_queries lower bound — Foundry server fan-out hidden; see App Insights" if web_search_calls else "no messages returned",
    )
    metrics.cost_usd = round(
        estimate_cost(
            backend=metrics.backend,
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cached_input_tokens=metrics.cached_input_tokens,
            web_search_calls=metrics.web_search_calls if metrics.web_search_calls else 1,
            bing_queries=metrics.bing_queries if metrics.bing_queries else None,
        ),
        4,
    )
    await reconcile_metrics(metrics, find_response_id(result), console=console)
    print_metrics(metrics, console)
    return metrics


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
