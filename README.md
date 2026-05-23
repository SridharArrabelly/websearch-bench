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
| `foundry-ws-bing` | `azure-ai-projects` | `WebSearchTool` on a Foundry `PromptAgentDefinition` (Bing Web Search) | Grounding with Bing Search |
| `foundry-ws-bingcustom` | `azure-ai-projects` | `WebSearchTool` + `WebSearchConfiguration` (Bing Custom Search) | Grounding with Bing Custom Search |
| `agentfx-bing` | `agent-framework-foundry` | `FoundryChatClient.get_web_search_tool(...)` wired into an `Agent` | Grounding with Bing Search |
| `agentfx-bing-cached` | `agent-framework-foundry` + Redis | Same as above, with Redis answer cache | 0 on cache hit; otherwise as above |
| `openai-ws` | `openai` | Responses API native `web_search` tool (`allowed_domains`) | OpenAI `web_search` per call |

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
        ├── auth.py                      # make_credential() — narrowed DefaultAzureCredential
        ├── appinsights.py               # fetch_chat_span() + reconcile_metrics()
        ├── compare.py                   # harness — runs all backends, writes results.csv/html
        └── backends/
            ├── __init__.py              # registry of backends
            ├── foundry_ws_bing.py
            ├── foundry_ws_bingcustom.py
            ├── agentfx_ws.py
            ├── agentfx_ws_cached.py
            └── openai_ws.py
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
  - A **Grounding with Bing Search** connection (for `foundry-ws-bing`, `agentfx-bing`, `agentfx-bing-cached`)
  - A **Grounding with Bing Custom Search** connection + instance (for `foundry-ws-bingcustom`)
- An **OpenAI API key** (only for `openai-ws`)
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
| `foundry-ws-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `foundry-ws-bingcustom` | `PROJECT_ENDPOINT`, `MODEL`, `BING_CUSTOM_SEARCH_CONNECTION_ID`, `BING_CUSTOM_SEARCH_INSTANCE_NAME` |
| `agentfx-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `agentfx-bing-cached` | above + `REDIS_URL` (and a running Redis) |
| `openai-ws` | `OPENAI_API_KEY`, `ENABLE_OPENAI_WS=1`, optional `OPENAI_MODEL` |

`openai-ws` is **opt-in** — it bills against your OpenAI subscription
(token rates + $10/1k for the `web_search` tool). Set
`ENABLE_OPENAI_WS=1` to include it in the comparison; it's skipped by
default so you can run the Azure/Foundry surfaces without an OpenAI key.

### Toggling backends

Every backend can be skipped without removing its env vars by setting its
`ENABLE_<NAME>` flag to `0` (also accepts `false` / `no` / `off`). The name is
the upper-case backend label with dashes turned into underscores:

| Flag                          | Default | Effect when `=0`              |
| ----------------------------- | ------- | ----------------------------- |
| `ENABLE_FOUNDRY_WS_BING`      | on      | skip `foundry-ws-bing`        |
| `ENABLE_FOUNDRY_WS_BINGCUSTOM`| on      | skip `foundry-ws-bingcustom`  |
| `ENABLE_AGENTFX_BING`         | on      | skip `agentfx-bing`           |
| `ENABLE_AGENTFX_BING_CACHED`  | on      | skip `agentfx-bing-cached`    |
| `ENABLE_OPENAI_WS`            | **off** | set to `1` to enable          |

Optional everywhere:

- `APPLICATIONINSIGHTS_CONNECTION_STRING` — used for **two** things: distributed
  tracing for **every agent_framework backend** (`agentfx-bing`,
  `agentfx-bing-cached`) via `azure.monitor.opentelemetry.configure_azure_monitor`
  + `agent_framework.observability.enable_instrumentation(enable_sensitive_data=True)`,
  **and** post-run reconciliation of `bing_queries` against the chat span (the
  only place Foundry's true server-side fan-out is visible — see
  [Metrics & cost model](#metrics--cost-model)). The harness also auto-sets
  `AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING=true` +
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` so chat spans carry
  the full `gen_ai.*` attributes. Your identity needs the **Monitoring Reader**
  or **Log Analytics Reader** role on the App Insights resource for the
  reconciler to work.
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
uv run python -m websearch_bench.backends.foundry_ws_bing
uv run python -m websearch_bench.backends.foundry_ws_bingcustom
uv run python -m websearch_bench.backends.agentfx_ws
uv run python -m websearch_bench.backends.agentfx_ws_cached
uv run python -m websearch_bench.backends.openai_ws
```

Each prints the agent's answer plus a normalized usage block (tokens, web-search
calls, total tool calls, latency, estimated USD cost).

## Metrics & cost model

Each run reports a normalized `RunMetrics` row:

| Column | Meaning |
| --- | --- |
| `input_tokens` / `output_tokens` / `total_tokens` | Model usage as reported by the SDK (OpenAI `response.usage`, Foundry's mirror of it, or `agent_framework`'s `usage_details`). |
| `cached_input_tokens` | Portion of `input_tokens` served from Azure OpenAI prompt caching. Billed at the **cached_input** rate (≈10× cheaper). Same number surfaced on the Foundry App Insights span as `gen_ai.usage.cached_tokens`. |
| `web_search_calls` | Number of `web_search_call` items in the response — i.e. distinct tool invocations the model emitted. |
| `bing_queries` | True Bing transaction count, **reconciled from App Insights** when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set. <br>• For **Foundry-hosted** backends (`foundry-ws-bing`, `foundry-ws-bingcustom`) this is `count(role=="tool")` from the server-side `chat` span's `gen_ai.input.messages` array — the only place the real `web.run` fan-out is visible. The Responses API only exposes a summarized `action.queries`. <br>• For **`agentfx-bing*`** the agent_framework client-side span exposes `search_tool_call` parts in `gen_ai.output.messages` — one per *model-level* call. The actual per-Bing-transaction fan-out is performed by Foundry server-side and is **not visible** to client-side instrumentation, so this value equals `web_search_calls` and is a **lower bound** (notes column says so). To see the true fan-out you need Foundry's own App Insights instance. <br>• For **`openai-ws`** there is no server fan-out — `action.queries` length is exact. |
| `latency_s` | Wall-clock seconds from sending the request to receiving the final response. |
| `cost_usd` | See cost formula below. |
| `answer_chars` | Length of the agent's final answer text. |

Cost formula:

```
fresh_in  = max(0, input_tokens - cached_input_tokens)
tokens_$  = (fresh_in           / 1000) * model_input_rate
          + (cached_input_tokens/ 1000) * model_cached_input_rate
          + (output_tokens      / 1000) * model_output_rate

tool_$    = ((bing_queries OR web_search_calls) / 1000) * tool_rate_per_1k

cost      = tokens_$ + tool_$
```

> **Foundry server-side fan-out is reconciled from App Insights.** Foundry's
> `web.run` extension is one billable "tool execution" from the model's POV but
> dispatches multiple Bing transactions internally. Each Bing hit appears as a
> separate `role="tool"` message in the **next** `chat` span's
> `gen_ai.input.messages` array, but the Responses API only exposes a
> summarized `action.queries`. So the bench auto-queries App Insights after
> every Foundry-backed run (using `gen_ai.response.id` as the join key) and
> overwrites `bing_queries` + `cost_usd` with the truth. Set
> `APPLICATIONINSIGHTS_CONNECTION_STRING` and grant your identity Monitoring
> Reader on the App Insights resource. Without it, the column falls back to
> the `action.queries` lower bound. Typical ingestion lag is 30-90s; the
> reconciler polls for up to 2 minutes.
>
> The fan-out is **variable per run** — the same question may produce
> 2, 14, 17, 23 … Bing hits depending on what the tool decides.

### Inspecting the raw response

To dump the raw SDK response (or `AgentResponse`) to disk for any run, set:

```powershell
$env:WEBSEARCH_BENCH_DEBUG="1"     # writes ./debug/<backend>-<timestamp>.json
uv run websearch-bench
```

Open the JSON to verify how the SDK reports `web_search_call` items and their
fan-out. Useful when the `bing_queries` column doesn't match the count you see
on the Foundry App Insights `execute_tool web.run` span — open a real dump and
extend `count_bing_queries_in_openai_output` in `shared.py` with the field
names you find.

Default `tool_rate_per_1k` (verified mid-2025, override via env):

| Backend | Tool rate | Source |
| --- | --- | --- |
| `foundry-ws-bing`, `agentfx-bing`, `agentfx-bing-cached (miss)` | **$35 / 1,000 calls** | [Grounding with Bing](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `foundry-ws-bingcustom` | **$35 / 1,000 calls** (new SKU; legacy $14 retired Aug 2025) | [Grounding with Bing Custom](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `openai-ws` | **$10 / 1,000 calls** (all models) | [OpenAI pricing](https://openai.com/api/pricing/) |
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
uv run python -m websearch_bench.backends.agentfx_ws_cached "What is the medical tax credit for 2025?"

# Bypass cache and refresh
uv run python -m websearch_bench.backends.agentfx_ws_cached --no-cache "..."

# Wipe all cache keys
uv run python -m websearch_bench.backends.agentfx_ws_cached --clear-cache
```

Start a local Redis if you don't already have one:

```powershell
docker run --rm -p 6379:6379 redis:7
```

## Troubleshooting

- **`DefaultAzureCredential` failures** — `az login` again; confirm
  `az account show` returns the expected subscription. The harness uses a
  narrowed credential chain (`websearch_bench.auth.make_credential`) that
  excludes Managed Identity / VS Code / shared-token-cache / Workload Identity
  probes — this avoids the spurious 504 `GET 169.254.169.254/metadata/...`
  dependency that shows up red in App Insights when running off-Azure. Auth
  chain becomes: environment vars → Azure CLI → Azure Developer CLI → Azure
  PowerShell.
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
