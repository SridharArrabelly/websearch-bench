"""Foundry agent grounded with the legacy ``BingGroundingTool``.

Unlike :mod:`foundry_ws_bing` which wraps OpenAI's new ``web_search`` tool
(``WebSearchTool``), this backend uses the original **Grounding with Bing
Search** tool — a single-shot search → snippet → answer pattern that does
**not** fan out into multiple Bing transactions per request.

For the same prompt, this typically uses ~5K input tokens vs ~15K for
``foundry-ws-bing``, with a single Bing API charge instead of N. The
trade-off is no per-call refinement: the model gets one snapshot of Bing
results to answer from.

Reference:
https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/bing-tools?pivots=python

Required env vars:
    PROJECT_ENDPOINT      — Foundry project endpoint
    BING_CONNECTION_NAME  — name of the Grounding-with-Bing connection on the
                            project (resolved via ``project.connections.get(NAME)``)
"""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    BingGroundingSearchConfiguration,
    BingGroundingSearchToolParameters,
    BingGroundingTool,
    PromptAgentDefinition,
)
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.auth import make_credential
from websearch_bench.shared import (
    MODEL,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    metrics_from_openai_response,
)

BACKEND_NAME = "foundry-bing-grounding"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT", "BING_CONNECTION_NAME")

console = Console()


async def run() -> RunMetrics:
    load_dotenv(override=True)
    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    bing_connection_name = os.environ["BING_CONNECTION_NAME"]

    async with (
        make_credential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        # Resolve the connection ID from its friendly name (per the docs).
        bing_connection = await project.connections.get(bing_connection_name)

        agent = await project.agents.create_version(
            agent_name="foundry-bing-grounding",
            definition=PromptAgentDefinition(
                model=MODEL,
                instructions=SHARED_INSTRUCTIONS,
                tools=[
                    BingGroundingTool(
                        bing_grounding=BingGroundingSearchToolParameters(
                            search_configurations=[
                                BingGroundingSearchConfiguration(
                                    project_connection_id=bing_connection.id,
                                )
                            ],
                        )
                    )
                ],
            ),
            description="Foundry Bing Grounding (legacy single-shot) benchmark backend.",
        )
        console.print(
            f"[dim]Agent created id={agent.id} name={agent.name} version={agent.version}[/dim]"
        )
        console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")

        with Timer() as t:
            response = await openai.responses.create(
                input=SHARED_QUERY,
                tool_choice="required",
                extra_body={
                    "agent_reference": {"name": agent.name, "type": "agent_reference"}
                },
            )

    console.print(f"\n[bold green]Agent:[/bold green] {getattr(response, 'output_text', '') or ''}")
    return metrics_from_openai_response(
        BACKEND_NAME, MODEL, response, t.elapsed,
        notes="BingGroundingTool: single-shot Bing query (no server fan-out)",
        console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
