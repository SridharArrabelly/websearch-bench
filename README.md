# foundry-websearch-tool

A small Python package — **`websearch_bench`** — that runs the *same* grounded
web-search question through every popular SDK surface and prints a side-by-side
table of **tokens, latency, and estimated cost** so you can pick the cheapest
backend for your workload with eyes open.

## Why this exists

Web-search grounding is available three different ways for the same Azure / OpenAI
ecosystem, and each one bills differently:

| Backend | Library | Tool plumbing | Per-call charge (besides model tokens) |
| --- | --- | --- | --- |
| `foundry-bing` | `azure-ai-projects` | `WebSearchTool` on a Foundry `PromptAgentDefinition` (Bing Web Search) | Grounding with Bing Search |
| `foundry-bing-custom` | `azure-ai-projects` | `WebSearchTool` + `WebSearchConfiguration` (Bing Custom Search) | Grounding with Bing Custom Search |
| `agentfx-bing` | `agent-framework-foundry` | `FoundryChatClient.get_web_search_tool(...)` wired into an `Agent` | Grounding with Bing Search |
| `agentfx-bing-cached` | `agent-framework-foundry` + Redis | Same as above, with Redis answer cache | 0 on cache hit; otherwise as above |
| `openai-web-search` | `openai` | Responses API native `web_search` tool (`allowed_domains`) | OpenAI `web_search` per call |

The whole point is that **every backend hits the same model with the same query,
same `search_context_size`, same domain filter, and same instructions** — so the
numbers are comparable. All shared workload lives in
[`src/websearch_bench/shared.py`](src/websearch_bench/shared.py).

## Repository layout

```
foundry-websearch-tool/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── src/
    └── websearch_bench/
        ├── __init__.py
        ├── __main__.py                  # python -m websearch_bench  → compare
        ├── shared.py                    # query, model, instructions, RunMetrics
        ├── pricing.py                   # USD constants + estimate_cost()
        ├── compare.py                   # harness — runs all backends, writes results.csv
        └── backends/
            ├── __init__.py              # registry of backends
            ├── foundry_bing.py
            ├── foundry_bing_custom.py
            ├── agentfx_bing.py
            ├── agentfx_bing_cached.py
            └── openai_web_search.py
```

Each backend module exposes the same contract:

```python
BACKEND_NAME: str
REQUIRED_ENV: tuple[str, ...]
async def run() -> RunMetrics
def main() -> None      # standalone entry point
```

Adding a new backend = one file in `backends/` + one line in
`backends/__init__.py`.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager
- **Azure CLI** (`az`) for local credential sign-in
- An **Azure AI Foundry** project with:
  - A deployed chat model (e.g. `gpt-5.1`) — used as `MODEL`
  - A **Grounding with Bing Search** connection (for `foundry-bing`, `agentfx-bing`, `agentfx-bing-cached`)
  - A **Grounding with Bing Custom Search** connection + instance (for `foundry-bing-custom`)
- An **OpenAI API key** (only for `openai-web-search`)
- (Optional) A reachable **Redis** instance for `agentfx-bing-cached`
- (Optional) An **Application Insights** connection string for tracing in `agentfx-bing-cached`

## Setup

```powershell
cd C:\path\to\foundry-websearch-tool
uv sync                                  # creates .venv, installs the package + deps
Copy-Item .env.example .env              # then edit .env (see "Environment" below)
az login
az account set --subscription <your-subscription-id>
```

Your signed-in identity needs the **Azure AI User** role (or equivalent) on the
Foundry project.

## Environment

Minimum env vars per backend (set in `.env`):

| Backend | Required env vars |
| --- | --- |
| `foundry-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `foundry-bing-custom` | `PROJECT_ENDPOINT`, `MODEL`, `BING_CUSTOM_SEARCH_CONNECTION_ID`, `BING_CUSTOM_SEARCH_INSTANCE_NAME` |
| `agentfx-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `agentfx-bing-cached` | above + `REDIS_URL` (and a running Redis) |
| `openai-web-search` | `OPENAI_API_KEY`, optional `OPENAI_MODEL` |

Optional everywhere:

- `APPLICATIONINSIGHTS_CONNECTION_STRING` — tracing for the cached backend.
- `BING_GROUNDING_USD_PER_1K`, `BING_CUSTOM_USD_PER_1K`,
  `OPENAI_WEB_SEARCH_USD_PER_1K` — override the per-1,000-call pricing
  (verified defaults: $35 / $35 / $10 — see [`pricing.py`](src/websearch_bench/pricing.py)
  for source links).

## Run

### Side-by-side comparison (the main artifact)

```powershell
uv run websearch-bench
# or
uv run python -m websearch_bench
```

This runs every backend whose env vars are present, prints a `rich` table in
the terminal, and writes two artifacts to the current working directory:

- **`results.html`** — self-contained report. Sortable summary table, bar
  charts (cost / total tokens / latency), and each backend's full answer
  collapsed in a `<details>` block. Open it in any browser. Chart.js is
  loaded from a CDN, so the page needs internet to render the charts (the
  table and answers still work offline).
- **`results.csv`** — same metrics minus the answer text, for spreadsheets
  and downstream scripting.

Backends with missing env vars are **skipped with a warning** — they don't
fail the run, and they still appear (greyed out) in both outputs.

### A single backend in isolation

```powershell
uv run python -m websearch_bench.backends.foundry_bing
uv run python -m websearch_bench.backends.foundry_bing_custom
uv run python -m websearch_bench.backends.agentfx_bing
uv run python -m websearch_bench.backends.agentfx_bing_cached
uv run python -m websearch_bench.backends.openai_web_search
```

Each prints the agent's answer plus a normalized usage block (tokens, web-search
calls, total tool calls, latency, estimated USD cost).

## Metrics & cost model

Each run reports a normalized `RunMetrics` row:

| Column | Meaning |
| --- | --- |
| `input_tokens` / `output_tokens` / `total_tokens` | Model usage as reported by the SDK (OpenAI `response.usage`, Foundry's mirror of it, or `agent_framework`'s `usage_details`). |
| `web_search_calls` | Number of times **the model** invoked the web-search tool to answer your one question. This is what Bing / OpenAI bill against — the user only sends 1 query but the model may issue 0, 1, 2, … web searches per response. |
| `tool_calls` | Number of **all** tool invocations (web_search, function tools, MCP tools, …). Equals `web_search_calls` in this bench because web_search is the only tool attached. |
| `latency_s` | Wall-clock seconds from sending the request to receiving the final response. |
| `cost_usd` | `model_token_cost + (web_search_calls / 1000) * vendor_price_per_1k` — see below. |
| `answer_chars` | Length of the agent's final answer text. |

Cost formula:

```
cost = (input_tokens  / 1000) * model_input_rate
     + (output_tokens / 1000) * model_output_rate
     + (web_search_calls / 1000) * tool_rate_per_1k
```

Default `tool_rate_per_1k` (verified mid-2025, override via env):

| Backend | Tool rate | Source |
| --- | --- | --- |
| `foundry-bing`, `agentfx-bing`, `agentfx-bing-cached (miss)` | **$35 / 1,000 calls** | [Grounding with Bing](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `foundry-bing-custom` | **$35 / 1,000 calls** (new SKU; legacy $14 retired Aug 2025) | [Grounding with Bing Custom](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `openai-web-search` | **$10 / 1,000 calls** (all models) | [OpenAI pricing](https://openai.com/api/pricing/) |
| `agentfx-bing-cached (hit)` | $0 — answer served from Redis, no Bing call | n/a |

Default model token rates (Azure OpenAI Global Standard, USD per **1M** tokens —
source: <https://azure.microsoft.com/pricing/details/azure-openai/>):

| Model | Input | Cached input | Output |
| --- | ---:| ---:| ---:|
| `gpt-5.1` | $1.25 | $0.125 | $10.00 |
| `gpt-5.1-mini` | $0.25 | $0.025 | $2.00 |
| `gpt-4o` | $2.50 | $1.25 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.075 | $0.60 |

Set `MODEL` (Foundry / Agent Framework runs) and `OPENAI_MODEL` (OpenAI run)
in `.env` — the harness picks the matching row automatically. Unknown models
fall through with a model-token cost of $0 (only the per-call tool charge is
billed), so add new models to `MODEL_PRICING_PER_1K` in `pricing.py` before
quoting.

## Change the workload

Every backend reads its workload from `src/websearch_bench/shared.py`. To
benchmark a different query / model / domain / `search_context_size`, edit
those module-level constants once and rerun `websearch-bench`.

## Pricing

`src/websearch_bench/pricing.py` ships with **verified defaults** for mid-2025
(see the table in [Metrics & cost model](#metrics--cost-model) above for sources).
Verify against the official pages before quoting a customer:

- Azure OpenAI / Foundry model pricing: <https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/>
- Grounding with Bing Search / Custom Search: <https://www.microsoft.com/bing/apis/grounding-pricing>
- OpenAI Responses API + `web_search`: <https://openai.com/api/pricing/>

Override per-1,000-call rates via env vars (see `.env.example`).

### Cache-only backend: extras

```powershell
# Ask a different question (cache key is per-query)
uv run python -m websearch_bench.backends.agentfx_bing_cached "What is the medical tax credit for 2025?"

# Bypass cache and refresh
uv run python -m websearch_bench.backends.agentfx_bing_cached --no-cache "..."

# Wipe all cache keys
uv run python -m websearch_bench.backends.agentfx_bing_cached --clear-cache
```

Start a local Redis if you don't already have one:

```powershell
docker run --rm -p 6379:6379 redis:7
```

## Troubleshooting

- **`DefaultAzureCredential` failures** — `az login` again; confirm
  `az account show` returns the expected subscription.
- **`PermissionDenied` on the Foundry project** — your identity needs the
  *Azure AI User* role on the project resource.
- **Bing Custom Search returns empty results** — verify
  `BING_CUSTOM_SEARCH_INSTANCE_NAME` matches an instance in the Bing Custom
  Search portal and that the instance includes the expected domains.
- **`web_search` errors from OpenAI** — your account must have access to the
  Responses API web-search tool and a model that supports it.
- **`agent_framework` import errors after `uv sync`** — this repo opts in to
  pre-release packages via `[tool.uv] prerelease = "allow"`; re-run `uv sync`.

## Security

Never commit `.env`, API keys, or connection strings. Rotate any key that has
been pasted into a chat or shared screen.
