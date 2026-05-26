"""Foundry agent with WebSearchTool pinned to a non-reasoning model.

Same code path as :mod:`foundry_ws_bing`, but the underlying model is a
**non-reasoning** model (default ``gpt-4.1-mini``, override via
``MODEL_FAST``). The goal is to test OpenAI's "non-reasoning web search"
path through Foundry's hosted ``WebSearchTool``:

    Non-reasoning web search: The non-reasoning model sends the user's
    query to the web search tool, which returns the response based on top
    results. There's no internal planning and the model simply passes along
    the search tool's responses. (OpenAI docs)

In practice this means **1 web_search_call → 1 Bing transaction** (no
``web.run`` server-side fan-out, no agentic re-query loop), at a much
lower input-token cost than gpt-5.1's agentic search.

Reference:
https://platform.openai.com/docs/guides/tools-web-search

Required env vars:
    PROJECT_ENDPOINT   — Foundry project endpoint
    MODEL_FAST         — non-reasoning model deployment name (default
                         ``gpt-4.1-mini``); must be deployed in your project
"""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    WebSearchApproximateLocation,
    WebSearchTool,
    WebSearchToolFilters,
)
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.auth import make_credential
from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    MODEL_FAST,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    USER_CITY,
    USER_COUNTRY,
    USER_REGION,
    RunMetrics,
    Timer,
    metrics_from_openai_response,
)

BACKEND_NAME = "foundry-ws-bing-fast"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT",)

console = Console()


async def run() -> RunMetrics:
    load_dotenv(override=True)
    project_endpoint = os.environ["PROJECT_ENDPOINT"]

    async with (
        make_credential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        agent = await project.agents.create_version(
            agent_name="foundry-ws-bing-fast",
            definition=PromptAgentDefinition(
                model=MODEL_FAST,
                instructions=SHARED_INSTRUCTIONS,
                tools=[
                    WebSearchTool(
                        user_location=WebSearchApproximateLocation(
                            country=USER_COUNTRY,
                            city=USER_CITY,
                            region=USER_REGION,
                        ),
                        search_context_size=SEARCH_CONTEXT_SIZE,
                        filters=WebSearchToolFilters(allowed_domains=ALLOWED_DOMAINS),
                    )
                ],
            ),
            description="Foundry Bing Web Search benchmark backend (non-reasoning model).",
        )
        console.print(
            f"[dim]Agent created id={agent.id} name={agent.name} version={agent.version} model={MODEL_FAST}[/dim]"
        )
        console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")

        with Timer() as t:
            response = await openai.responses.create(
                input=SHARED_QUERY,
                extra_body={
                    "agent_reference": {"name": agent.name, "type": "agent_reference"}
                },
            )

    console.print(f"\n[bold green]Agent:[/bold green] {getattr(response, 'output_text', '') or ''}")
    return metrics_from_openai_response(
        BACKEND_NAME, MODEL_FAST, response, t.elapsed,
        notes=f"WebSearchTool on non-reasoning model ({MODEL_FAST}); expected: 1 web_search_call, no fan-out",
        console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
