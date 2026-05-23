"""Cost-optimised variant of foundry-ws-bing.

Same plumbing as :mod:`foundry_ws_bing` but with three changes designed to
minimise Bing fan-out and input-token bloat:

1. ``max_tool_calls=1`` on the Responses request — hard cap on how many
   ``web_search_call`` rounds the model can emit. Eliminates "refine the
   query and search again" loops, which are the dominant source of input
   token growth (each round re-feeds the conversation including all prior
   Bing results).
2. ``parallel_tool_calls=False`` — additionally prevents concurrent
   tool execution; defence in depth alongside ``max_tool_calls``.
3. ``tool_choice={"type": "web_search"}`` — forces exactly one
   ``web_search`` invocation instead of letting the model decide whether
   to search at all.
4. Stricter system prompt (``SHARED_INSTRUCTIONS_SINGLE_QUERY``) telling
   the model to craft a single optimised search phrase.

Caveats:

- Foundry's server-side ``web.run`` extension still fans one
  ``web_search_call`` into N Bing transactions internally; the API has no
  knob to control that. So this backend reduces fan-out at the *model*
  layer (rounds), not at the *Bing* layer (transactions per round).
- The single-query constraint can hurt answer quality on multi-faceted
  questions. The whole point of this backend is to put the trade-off in
  hard numbers next to the unconstrained ``foundry-ws-bing``.
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
from websearch_bench.pricing import estimate_cost
from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS_SINGLE_QUERY,
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

BACKEND_NAME = "foundry-ws-bing-cheap"
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
            agent_name="foundry-ws-bing-cheap",
            definition=PromptAgentDefinition(
                model=MODEL,
                instructions=SHARED_INSTRUCTIONS_SINGLE_QUERY,
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
            description="Cost-optimised Foundry Bing Web Search backend (max_tool_calls=1).",
        )
        console.print(
            f"[dim]Agent created id={agent.id} name={agent.name} version={agent.version}[/dim]"
        )
        console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")

        with Timer() as t:
            response = await openai.responses.create(
                input=SHARED_QUERY,
                max_tool_calls=1,
                parallel_tool_calls=False,
                tool_choice={"type": "web_search"},
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
        notes="max_tool_calls=1 + tool_choice=web_search; bing_queries lower bound — server fan-out hidden",
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

    metrics.response_id = getattr(response, "id", None)

    print_metrics(metrics, console)
    return metrics


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
