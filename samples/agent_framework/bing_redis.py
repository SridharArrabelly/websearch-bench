"""Run a Foundry agent that answers SARS tax questions using Bing web search.

The agent is constrained to the SARS website and must cite every claim with
URLs returned by the web search tool. Answers are cached in Redis to avoid
repeated (and expensive) Bing-grounded calls for the same question.

Usage:
    python bing_redis.py
    python bing_redis.py "What is the medical tax credit for 2025?"
    python bing_redis.py --no-cache "..."   # bypass + refresh cache
    python bing_redis.py --clear-cache      # wipe app keys and exit

Required environment variables (loaded from .env):
    PROJECT_ENDPOINT  Azure AI Foundry project endpoint.
    MODEL             (optional) Model deployment name. Defaults to "gpt-4o".
    REDIS_URL         (optional) Redis connection URL. Defaults to redis://localhost:6379.
    CACHE_TTL_HOURS   (optional) Cache lifetime in hours. Defaults to 24.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import redis.asyncio as redis
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.observability import enable_instrumentation
from azure.identity.aio import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logger = logging.getLogger("websearch_agent")

DEFAULT_MODEL = os.getenv("MODEL", "gpt-5.1")  # Default to gpt-4o if MODEL is not set in .env
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_QUERY = "What are the individual income tax brackets for the 2027 tax year (1 March 2026 to 28 February 2027)?"

AGENT_NAME = "WebSearchToolAgent"
AGENT_DESCRIPTION = "Agent for SARS-grounded web search."
AGENT_INSTRUCTIONS = (
    "You are a research assistant for a South African audience. "
    "You MUST answer using ONLY information returned by the web_search tool. "
    "If the search tool returns no relevant results, reply: "
    "'I could not find this in the configured sources.' "
    "Every factual claim must be followed by a numbered citation [n] and a Sources list "
    "containing only URLs returned by the tool."
)

# Web search tool configuration.
USER_COUNTRY = "ZA"
ALLOWED_DOMAINS = ["www.sars.gov.za"]
SEARCH_CONTEXT_SIZE = "low"

# Cache configuration.
CACHE_KEY_PREFIX = "websearch:answer:"
DEFAULT_CACHE_TTL_HOURS = 24


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from the environment."""

    project_endpoint: str
    model: str
    redis_url: str
    cache_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "PROJECT_ENDPOINT is not set. Add it to your .env file or environment."
            )
        ttl_hours = float(os.getenv("CACHE_TTL_HOURS", DEFAULT_CACHE_TTL_HOURS))
        return cls(
            project_endpoint=endpoint,
            model=os.getenv("MODEL", DEFAULT_MODEL),
            redis_url=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
            cache_ttl_seconds=int(ttl_hours * 3600),
        )


# ---------------------------------------------------------------------------
# Cache (Redis-backed; one key per question, TTL via Redis EXPIRE).
# ---------------------------------------------------------------------------

class RedisAnswerCache:
    """Async Redis cache so repeated questions don't re-call Bing.

    Keys:   websearch:answer:<sha256(model|normalized_query)>
    Value:  JSON {model, query, answer, timestamp}
    TTL:    set on write via Redis EXPIRE (seconds)
    """

    def __init__(self, client: redis.Redis, ttl_seconds: int) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(model: str, query: str) -> str:
        normalized = query.strip().lower()
        digest = hashlib.sha256(f"{model}|{normalized}".encode("utf-8")).hexdigest()
        return f"{CACHE_KEY_PREFIX}{digest}"

    async def get(self, model: str, query: str) -> str | None:
        try:
            raw = await self.client.get(self._key(model, query))
        except redis.RedisError:
            logger.exception("Redis GET failed; treating as cache miss")
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Cache entry malformed; ignoring")
            return None

    async def put(self, model: str, query: str, answer: str) -> None:
        key = self._key(model, query)
        payload = json.dumps(
            {
                "model": model,
                "query": query,
                "answer": answer,
                "timestamp": time.time(),
            },
            ensure_ascii=False,
        )
        logger.info("Cache put: key=%s ttl=%ds payload_bytes=%d", key, self.ttl_seconds, len(payload))
        try:
            ok = await self.client.set(key, payload, ex=self.ttl_seconds)
            logger.info("Cache put result: %r", ok)
        except redis.RedisError:
            logger.exception("Redis SET failed; answer not cached")

    async def clear(self) -> int:
        """Delete all keys under our prefix. Returns the number deleted."""
        deleted = 0
        async for key in self.client.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
            deleted += await self.client.delete(key)
        logger.info("Cleared %d cache entries", deleted)
        return deleted


def make_redis_client(url: str) -> redis.Redis:
    """Build a Redis client with text decoding so .get() returns str, not bytes."""
    return redis.from_url(url, decode_responses=True)


# ---------------------------------------------------------------------------
# Logging / agent wiring
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Configure rich-formatted logging once for the whole process."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)


def build_agent(client: FoundryChatClient) -> Agent:
    """Create the Foundry agent wired up with the constrained web search tool."""
    logger.info("Creating web search tool (country=%s, domains=%s)", USER_COUNTRY, ALLOWED_DOMAINS)
    web_search_tool = client.get_web_search_tool(
        user_location={"country": USER_COUNTRY},
        allowed_domains=ALLOWED_DOMAINS,
        search_context_size=SEARCH_CONTEXT_SIZE,
    )

    logger.info("Creating agent: %s", AGENT_NAME)
    return Agent(
        client=client,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=[web_search_tool],
        description=AGENT_DESCRIPTION,
    )


async def ask(query: str, *, use_cache: bool = True) -> str:
    """Run the agent for a single query, using the Redis cache when possible."""
    settings = Settings.from_env()
    redis_client = make_redis_client(settings.redis_url)
    cache = RedisAnswerCache(redis_client, settings.cache_ttl_seconds)

    try:
        if use_cache:
            cached = await cache.get(settings.model, query)
            if cached is not None:
                logger.info("Cache hit — skipping agent call")
                console.print(f"[bold cyan]User:[/bold cyan] {query}")
                console.print(f"[bold green]Agent (cached):[/bold green] {cached}")
                return cached
            logger.info("Cache miss — calling agent")

        async with DefaultAzureCredential() as credential:
            client = FoundryChatClient(
                project_endpoint=settings.project_endpoint,
                credential=credential,
                model=settings.model,
            )
            agent = build_agent(client)

            console.print(f"[bold cyan]User:[/bold cyan] {query}")
            result = await agent.run(query)

        await cache.put(settings.model, query, result.text)
        console.print(f"[bold green]Agent:[/bold green] {result.text}")
        return result.text
    finally:
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CliArgs:
    query: str
    use_cache: bool
    clear_cache: bool


def parse_args(argv: list[str]) -> CliArgs:
    """Tiny hand-rolled arg parser — enough for two flags + a free-form query."""
    use_cache = True
    clear_cache = False
    positional: list[str] = []

    for arg in argv:
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--clear-cache":
            clear_cache = True
        else:
            positional.append(arg)

    query = " ".join(positional).strip() or DEFAULT_QUERY
    return CliArgs(query=query, use_cache=use_cache, clear_cache=clear_cache)


async def clear_cache_command() -> None:
    settings = Settings.from_env()
    redis_client = make_redis_client(settings.redis_url)
    try:
        await RedisAnswerCache(redis_client, settings.cache_ttl_seconds).clear()
    finally:
        await redis_client.aclose()


async def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.clear_cache:
        await clear_cache_command()
        return

    try:
        await ask(args.query, use_cache=args.use_cache)
    except Exception:
        logger.exception("Agent run failed")
        raise


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    load_dotenv(override=True)
    configure_logging()
    # Send traces to the Foundry project's Application Insights resource.
    # Requires APPLICATIONINSIGHTS_CONNECTION_STRING in the environment.
    configure_azure_monitor()
    enable_instrumentation(enable_sensitive_data=True)
    asyncio.run(main())

