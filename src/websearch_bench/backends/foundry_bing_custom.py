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
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from rich.console import Console

from websearch_bench.pricing import estimate_cost
from websearch_bench.shared import (
    MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    count_tool_calls_in_openai_output,
    count_web_search_calls_in_openai_output,
    print_metrics,
    usage_from_openai_response,
)

BACKEND_NAME = "foundry-bing-custom"
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
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        agent = await project.agents.create_version(
            agent_name="WebSearchToolAgent-BingCustom",
            definition=PromptAgentDefinition(
                model=MODEL,
                instructions=SHARED_INSTRUCTIONS,
                tools=[
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

    answer = getattr(response, "output_text", "") or ""
    console.print(f"\n[bold green]Agent:[/bold green] {answer}")

    usage = usage_from_openai_response(response)
    metrics = RunMetrics(
        backend=BACKEND_NAME,
        model=MODEL,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        web_search_calls=count_web_search_calls_in_openai_output(response),
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
        ),
        4,
    )
    print_metrics(metrics, console)
    return metrics


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
