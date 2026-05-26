"""Microsoft Agent Framework agent backed by Foundry's web-search tool."""

from __future__ import annotations

import asyncio
import logging
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.auth import make_credential
from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    USER_COUNTRY,
    RunMetrics,
    Timer,
    count_web_search_calls_in_agent_response,
    metrics_from_agentfx_result,
)

BACKEND_NAME = "agentfx-bing"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT",)

console = Console()
logger = logging.getLogger(__name__)


def build_agent(client: FoundryChatClient, *, name: str, description: str) -> Agent:
    """Construct an Agent Framework agent with Foundry's hosted web-search tool.

    Centralized so the cached variant and the non-cached one stay in lockstep
    on tool config (allowed_domains, location, context size).
    """
    web_search_tool = client.get_web_search_tool(
        user_location={"country": USER_COUNTRY},
        allowed_domains=ALLOWED_DOMAINS,
        search_context_size=SEARCH_CONTEXT_SIZE,
    )
    return Agent(
        client=client,
        name=name,
        instructions=SHARED_INSTRUCTIONS,
        tools=[web_search_tool],
        description=description,
    )


async def run() -> RunMetrics:
    load_dotenv(override=True)

    async with make_credential() as cred:
        client = FoundryChatClient(
            project_endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=cred,
            model=MODEL,
        )
        agent = build_agent(
            client,
            name="agentfx-bing",
            description="Agent Framework + Foundry Bing benchmark backend.",
        )

        console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")
        with Timer() as t:
            result = await agent.run(SHARED_QUERY)

    console.print(f"\n[bold green]Agent:[/bold green] {result.text}")
    notes = (
        "bing_queries lower bound — Foundry server fan-out hidden; see App Insights"
        if count_web_search_calls_in_agent_response(result)
        else "no messages returned"
    )
    return metrics_from_agentfx_result(
        BACKEND_NAME, MODEL, result, t.elapsed, notes=notes, console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
