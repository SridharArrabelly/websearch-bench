"""OpenAI Responses API + native ``web_search`` tool benchmark backend."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console

from websearch_bench.shared import (
    ALLOWED_DOMAINS,
    OPENAI_MODEL,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    metrics_from_openai_response,
)

BACKEND_NAME = "openai-ws"
REQUIRED_ENV: tuple[str, ...] = ("OPENAI_API_KEY",)
# This backend is opt-in because it bills against an OpenAI subscription
# (web_search is $10/1k calls + standard token rates). Most contributors only
# care about the Azure/Foundry surfaces, so we skip it unless explicitly
# enabled.
_ENABLE_VAR = "ENABLE_OPENAI_WS"

console = Console()


def enabled() -> tuple[bool, str]:
    """Skip unless the user explicitly opts in.

    Returns ``(True, "")`` to run, or ``(False, reason)`` to skip. The
    benchmark runner uses this hook to decide whether to invoke ``run()``.
    """
    load_dotenv(override=True)
    val = os.getenv(_ENABLE_VAR, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True, ""
    return False, f"set {_ENABLE_VAR}=1 to enable (requires paid OpenAI subscription)"


async def run() -> RunMetrics:
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    client = AsyncOpenAI(api_key=api_key)
    console.print(f"[bold cyan]User:[/bold cyan] {SHARED_QUERY}")

    with Timer() as t:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=SHARED_INSTRUCTIONS,
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": ALLOWED_DOMAINS},
                }
            ],
            include=["web_search_call.action.sources"],
            input=SHARED_QUERY,
        )

    console.print("\n[bold green]Agent:[/bold green]")
    console.print(response.output_text)

    console.print("\n[bold]Sources[/bold]")
    for item in response.output:
        if getattr(item, "type", None) == "web_search_call":
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None) if action else None
            if sources:
                for s in sources:
                    console.print(f"- {getattr(s, 'url', s)}")

    return metrics_from_openai_response(
        BACKEND_NAME, OPENAI_MODEL, response, t.elapsed, console=console,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
