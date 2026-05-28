"""Agent Framework + Foundry web search, with a Redis answer cache.

Cache hits return the previous answer without calling Bing, which is the
main lever for cutting cost on repeat queries.

CLI:
    python -m websearch_bench.backends.agentfx_ws_cached
    python -m websearch_bench.backends.agentfx_ws_cached "What is the medical tax credit for 2025?"
    python -m websearch_bench.backends.agentfx_ws_cached --no-cache "..."
    python -m websearch_bench.backends.agentfx_ws_cached --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import redis.asyncio as redis
from agent_framework.foundry import FoundryChatClient
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from websearch_bench.auth import make_credential
from websearch_bench.backends.agentfx_ws import build_agent
from websearch_bench.shared import (
    MODEL,
    SHARED_QUERY,
    RunMetrics,
    Timer,
    metrics_from_agentfx_result,
    print_metrics,
    setup_tracing,
)

BACKEND_NAME = "agentfx-bing-cached"
REQUIRED_ENV: tuple[str, ...] = ("PROJECT_ENDPOINT",)

DEFAULT_REDIS_URL = "redis://localhost:6379"
CACHE_KEY_PREFIX = "websearch:answer:"
DEFAULT_CACHE_TTL_HOURS = 24

console = Console()
logger = logging.getLogger("websearch_bench.agentfx_bing_cached")


@dataclass(frozen=True)
class Settings:
    project_endpoint: str
    model: str
    redis_url: str
    cache_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        endpoint = os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            raise RuntimeError("PROJECT_ENDPOINT is not set. Add it to your .env file.")
        ttl_hours = float(os.getenv("CACHE_TTL_HOURS", DEFAULT_CACHE_TTL_HOURS))
        return cls(
            project_endpoint=endpoint,
            model=os.getenv("MODEL", MODEL),
            redis_url=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
            cache_ttl_seconds=int(ttl_hours * 3600),
        )


class RedisAnswerCache:
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
            {"model": model, "query": query, "answer": answer, "timestamp": time.time()},
            ensure_ascii=False,
        )
        try:
            await self.client.set(key, payload, ex=self.ttl_seconds)
        except redis.RedisError:
            logger.exception("Redis SET failed; answer not cached")

    async def clear(self) -> int:
        deleted = 0
        async for key in self.client.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
            deleted += await self.client.delete(key)
        return deleted


def _make_redis_client(url: str) -> redis.Redis:
    return redis.from_url(url, decode_responses=True)


async def ask(query: str = SHARED_QUERY, *, use_cache: bool = True) -> RunMetrics:
    settings = Settings.from_env()
    redis_client = _make_redis_client(settings.redis_url)
    cache = RedisAnswerCache(redis_client, settings.cache_ttl_seconds)

    try:
        if use_cache:
            cached = await cache.get(settings.model, query)
            if cached is not None:
                console.print(f"[bold cyan]User:[/bold cyan] {query}")
                console.print(f"[bold green]Agent (cached):[/bold green] {cached}")
                metrics = RunMetrics(
                    backend=f"{BACKEND_NAME} (hit)",
                    model=settings.model,
                    answer_chars=len(cached),
                    answer=cached,
                    latency_s=0.0,
                    cost_usd=0.0,
                    notes="served from Redis cache — no Bing call",
                )
                print_metrics(metrics, console)
                return metrics

        async with make_credential() as credential:
            client = FoundryChatClient(
                project_endpoint=settings.project_endpoint,
                credential=credential,
                model=settings.model,
            )
            agent = build_agent(
                client,
                name="agentfx-bing-cached",
                description="Agent Framework + Foundry Bing benchmark backend (cached).",
            )
            console.print(f"[bold cyan]User:[/bold cyan] {query}")
            with Timer() as t:
                result = await agent.run(query)

        await cache.put(settings.model, query, result.text)
        console.print(f"[bold green]Agent:[/bold green] {result.text}")
        # Pricing routes on the prefix "agentfx" — pass that as cost_backend
        # so the "(miss)" suffix in the displayed backend label doesn't break
        # the lookup.
        return metrics_from_agentfx_result(
            f"{BACKEND_NAME} (miss)", settings.model, result, t.elapsed,
            notes="cache miss — Bing called, result cached (bing_queries lower bound; see App Insights)",
            cost_backend="agentfx-bing",
            console=console,
        )
    finally:
        await redis_client.aclose()


async def run() -> RunMetrics:
    return await ask(SHARED_QUERY, use_cache=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached Bing-grounded agent benchmark.")
    parser.add_argument("query", nargs="*", help="Question to ask. Defaults to SHARED_QUERY.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass and refresh cache.")
    parser.add_argument("--clear-cache", action="store_true", help="Wipe cache and exit.")
    return parser.parse_args(argv)


async def _clear_cache_command() -> None:
    settings = Settings.from_env()
    redis_client = _make_redis_client(settings.redis_url)
    try:
        deleted = await RedisAnswerCache(redis_client, settings.cache_ttl_seconds).clear()
        console.print(f"Cleared {deleted} cache entries")
    finally:
        await redis_client.aclose()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)


async def _amain() -> None:
    args = _parse_args(sys.argv[1:])
    if args.clear_cache:
        await _clear_cache_command()
        return
    query = " ".join(args.query).strip() or SHARED_QUERY
    await ask(query, use_cache=not args.no_cache)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    load_dotenv(override=True)
    _configure_logging()
    setup_tracing(console)
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
