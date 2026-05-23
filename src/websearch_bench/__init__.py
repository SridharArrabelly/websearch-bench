"""websearch_bench — compare web-search grounding across SDK surfaces.

Stable public API:
    from websearch_bench import RunMetrics, SHARED_QUERY, estimate_cost
    from websearch_bench.backends import BACKENDS
"""

from .pricing import estimate_cost
from .shared import (
    ALLOWED_DOMAINS,
    MODEL,
    OPENAI_MODEL,
    SEARCH_CONTEXT_SIZE,
    SHARED_INSTRUCTIONS,
    SHARED_QUERY,
    USER_CITY,
    USER_COUNTRY,
    USER_REGION,
    RunMetrics,
    Timer,
)

__all__ = [
    "ALLOWED_DOMAINS",
    "MODEL",
    "OPENAI_MODEL",
    "SEARCH_CONTEXT_SIZE",
    "SHARED_INSTRUCTIONS",
    "SHARED_QUERY",
    "USER_CITY",
    "USER_COUNTRY",
    "USER_REGION",
    "RunMetrics",
    "Timer",
    "estimate_cost",
]
