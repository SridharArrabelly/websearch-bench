"""Azure AI Foundry agent grounded with Bing Web Search."""

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
    MODEL,
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

BACKEND_NAME = "foundry-ws-bing"
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
            agent_name="foundry-ws-bing",
            definition=PromptAgentDefinition(
                model=MODEL,
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
            description="Foundry Bing Web Search benchmark backend.",
        )
        console.print(
            f"[dim]Agent created id={agent.id} name={agent.name} version={agent.version}[/dim]"
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
    # Deferred reconciliation: compare.py batch-reconciles all backends at the
    # end so telemetry has time to ingest.
    return metrics_from_openai_response(
        BACKEND_NAME, MODEL, response, t.elapsed,
        notes="bing_queries from response is lower bound — server fan-out hidden; see App Insights",
        console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
