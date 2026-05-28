"""Foundry agent grounded with the legacy ``BingCustomSearchPreviewTool``.

The Bing Custom Search variant of :mod:`foundry_bing_grounding` — same
single-shot Bing semantics (one Bing API call per ``bing_custom_search_call``
item, no server-side ``web.run`` fan-out), but restricted to the public-web
slice you configured on the Bing Custom Search instance in the Bing portal.

Use this for "ground responses on these whitelisted domains" scenarios where
you want the low-token / one-Bing-call cost profile of the legacy tool, not
the WebSearchTool's fan-out behaviour.

Reference:
https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/bing-tools?pivots=python

Required env vars:
    PROJECT_ENDPOINT                    — Foundry project endpoint
    BING_CUSTOM_SEARCH_CONNECTION_ID    — full ARM connection ID of the Bing
                                          Custom Search resource on the
                                          project (passed straight to
                                          ``project_connection_id``)
    BING_CUSTOM_SEARCH_INSTANCE_NAME    — custom-search instance name on the
                                          Bing resource (configured in the
                                          Bing portal — defines the allowed
                                          domain list)
"""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    BingCustomSearchConfiguration,
    BingCustomSearchPreviewTool,
    BingCustomSearchToolParameters,
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

BACKEND_NAME = "foundry-bing-grounding-custom"
REQUIRED_ENV: tuple[str, ...] = (
    "PROJECT_ENDPOINT",
    "BING_CUSTOM_SEARCH_CONNECTION_ID",
    "BING_CUSTOM_SEARCH_INSTANCE_NAME",
)

console = Console()


async def run() -> RunMetrics:
    load_dotenv(override=True)
    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    bing_connection_id = os.environ["BING_CUSTOM_SEARCH_CONNECTION_ID"]
    instance_name = os.environ["BING_CUSTOM_SEARCH_INSTANCE_NAME"]

    async with (
        make_credential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        agent = await project.agents.create_version(
            agent_name="foundry-bing-grounding-custom",
            definition=PromptAgentDefinition(
                model=MODEL,
                instructions=SHARED_INSTRUCTIONS,
                tools=[
                    BingCustomSearchPreviewTool(
                        bing_custom_search_preview=BingCustomSearchToolParameters(
                            search_configurations=[
                                BingCustomSearchConfiguration(
                                    project_connection_id=bing_connection_id,
                                    instance_name=instance_name,
                                )
                            ],
                        )
                    )
                ],
            ),
            description="Foundry Bing Custom Search (preview, legacy single-shot) benchmark backend.",
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
        notes="BingCustomSearchPreviewTool: single-shot Bing Custom Search query (no server fan-out)",
        console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
