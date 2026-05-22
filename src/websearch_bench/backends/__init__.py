"""Registry of available benchmark backends.

Each backend module exposes the same contract:

    BACKEND_NAME: str
    REQUIRED_ENV: tuple[str, ...]
    async def run() -> RunMetrics

The list below is the single source of truth — ``compare.py`` iterates it.
Add a new backend by writing the module and appending to ``BACKENDS``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_BACKEND_MODULES: list[str] = [
    "websearch_bench.backends.foundry_bing",
    "websearch_bench.backends.foundry_bing_custom",
    "websearch_bench.backends.agentfx_bing",
    "websearch_bench.backends.agentfx_bing_cached",
    "websearch_bench.backends.openai_web_search",
]


def discover() -> list[ModuleType]:
    """Import every registered backend module and return them in order."""
    return [import_module(name) for name in _BACKEND_MODULES]


__all__ = ["discover"]
