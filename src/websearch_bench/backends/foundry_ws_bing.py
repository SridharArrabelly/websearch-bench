"""Azure AI Foundry agent grounded with Bing Web Search."""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    WebSearchApproximateLocation,
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
    USER_CITY,
    USER_COUNTRY,
    USER_REGION,
    RunMetrics,
    Timer,
    count_bing_queries_in_openai_output,
    debug_dump,
    count_web_search_calls_in_openai_output,
    print_metrics,
    usage_from_openai_response,
)

BACKEND_NAME = "foundry-ws-bing"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT",)

console = Console()


async def run() -> RunMetrics:
    load_dotenv(override=True)
    project_endpoint = os.environ["PROJECT_ENDPOINT"]

    async with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project,
    ):
        openai = project.get_openai_client()

        agent = await project.agents.create_version(
            agent_name="WebSearchToolAgent-Bing",
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

    answer = getattr(response, "output_text", "") or ""
    console.print(f"\n[bold green]Agent:[/bold green] {answer}")

    _dump = debug_dump(BACKEND_NAME, response)
    if _dump:
        console.print(f"[dim]Debug dump: {_dump}[/dim]")
    usage = usage_from_openai_response(response)
    metrics = RunMetrics(
        backend=BACKEND_NAME,
        model=MODEL,
        input_tokens=usage.get("input_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        web_search_calls=count_web_search_calls_in_openai_output(response),
        bing_queries=count_bing_queries_in_openai_output(response),
        latency_s=round(t.elapsed, 2),
        answer_chars=len(answer),
        answer=answer,
        notes="bing_queries from response is lower bound — server fan-out hidden; see App Insights",
    )
    metrics.cost_usd = round(
        estimate_cost(
            backend=metrics.backend,
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cached_input_tokens=metrics.cached_input_tokens,
            web_search_calls=metrics.web_search_calls,
            bing_queries=metrics.bing_queries,
        ),
        4,
    )

    # Defer reconciliation: compare.py will batch-reconcile all backends at
    # the end (so telemetry has time to ingest).
    metrics.response_id = getattr(response, "id", None)

    print_metrics(metrics, console)
    return metrics


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
