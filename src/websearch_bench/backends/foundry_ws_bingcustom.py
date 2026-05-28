"""Azure AI Foundry agent grounded with Bing Custom Search."""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    WebSearchConfiguration,
    WebSearchTool,
)
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.auth import make_credential
from websearch_bench.shared import (
    MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    metrics_from_openai_response,
)

BACKEND_NAME = "foundry-ws-bingcustom"
REQUIRED_ENV: tuple[str, ...] = (
    "PROJECT_ENDPOINT",
    "BING_CUSTOM_SEARCH_CONNECTION_ID",
    "BING_CUSTOM_SEARCH_INSTANCE_NAME",
)

console = Console()


async def run() -> RunMetrics:
    load_dotenv(override=True)
    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    connection_id = os.environ["BING_CUSTOM_SEARCH_CONNECTION_ID"]
    instance_name = os.environ["BING_CUSTOM_SEARCH_INSTANCE_NAME"]

    async with (
        make_credential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        agent = await project.agents.create_version(
            agent_name="foundry-ws-bingcustom",
            definition=PromptAgentDefinition(
                model=MODEL,
                instructions=SHARED_INSTRUCTIONS,
                tools=[
                    # NOTE: ALLOWED_DOMAINS (from .env / shared.py) is not
                    # applied here. When WebSearchTool is configured with a
                    # custom_search_configuration, the allowed-domain list is
                    # owned by the Bing Custom Search *instance* itself —
                    # configure it in https://www.customsearch.ai/ for the
                    # instance referenced by BING_CUSTOM_SEARCH_INSTANCE_NAME.
                    WebSearchTool(
                        custom_search_configuration=WebSearchConfiguration(
                            project_connection_id=connection_id,
                            instance_name=instance_name,
                        ),
                        search_context_size=SEARCH_CONTEXT_SIZE,
                    )
                ],
            ),
            description="Foundry Bing Custom Search benchmark backend.",
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
    return metrics_from_openai_response(
        BACKEND_NAME, MODEL, response, t.elapsed,
        notes="bing_queries from response is lower bound — server fan-out hidden; see App Insights",
        console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
