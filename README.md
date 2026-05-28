# websearch-bench

A small Python package — **`websearch_bench`** — that runs the *same* grounded
web-search question through every Azure AI Foundry grounding surface and prints
a side-by-side table of **tokens, latency, and estimated cost** so you can pick
the cheapest backend for your workload with eyes open.

Every backend bills tool calls at the same **Grounding with Bing** rate
($35 / 1,000 calls). The differences that matter are **what each backend
exposes** (1 outer call vs N inner Bing transactions), **how token-heavy each
SDK surface is**, and **where the charge lands on your invoice** (your Bing
resource vs your Foundry account).

## Backends

Two tool families × two SDK surfaces, on a Foundry project:

| Backend | Tool family | SDK | Notes |
| --- | --- | --- | --- |
| `foundry-bing-grounding` | Legacy `BingGroundingTool` | `azure-ai-projects` (Foundry agent + `agent_reference`) | Single-shot. No server-side `web.run` fan-out. Bills the user's `Microsoft.Bing/accounts` resource. |
| `foundry-bing-grounding-custom` | Legacy `BingCustomSearchPreviewTool` | `azure-ai-projects` | Single-shot. Domain restriction lives on the Bing Custom Search instance. |
| `foundry-ws-bing` | `WebSearchTool` (Bing Web Search) | `azure-ai-projects` | Hosted tool. Server-side `web.run` fan-out — one outer `web_search_call` can dispatch many Bing transactions. |
| `foundry-ws-bing-fast` | `WebSearchTool` (Bing Web Search) | `azure-ai-projects` | Same as above but pinned to `MODEL_FAST` (default `gpt-4o`) — tests the non-reasoning path (typically 1 search, no fan-out). |
| `foundry-ws-bingcustom` | `WebSearchTool` + `WebSearchConfiguration` (Bing Custom Search) | `azure-ai-projects` | Hosted Custom-Search variant. Allowed-domain list lives on the instance. |
| `agentfx-bing` | `WebSearchTool` (Bing Web Search) | `agent-framework-foundry` (`FoundryChatClient.get_web_search_tool`) | Agent loop runs **client-side** in the SDK; each tool call is a separate `responses.create()` to Foundry. |
| `agentfx-bing-cached` | Same as `agentfx-bing` + Redis answer cache | `agent-framework-foundry` + Redis | Reports `(hit)` (cost = 0) or `(miss)` (cost = same as `agentfx-bing`). |

The whole point: **every backend hits the same model with the same query and
the same instructions** — so the numbers are comparable. All shared workload
lives in [`src/websearch_bench/shared.py`](src/websearch_bench/shared.py).

### What the SDKs do and don't expose

Two per-tool knobs can't be wired uniformly because the SDKs don't all expose
them:

| Setting | `foundry-bing-grounding` | `foundry-bing-grounding-custom` | `foundry-ws-bing` | `foundry-ws-bingcustom` | `agentfx-bing*` |
| --- | --- | --- | --- | --- | --- |
| `SEARCH_CONTEXT_SIZE` | n/a — `BingGroundingTool` has no context-size knob | n/a — same | ✅ passed to `WebSearchTool` | ✅ passed to `WebSearchTool` | ✅ passed via `get_web_search_tool` |
| `ALLOWED_DOMAINS` (`.env`) | n/a — use a Bing Custom Search connection if you need domain restriction | ❌ configure on the Bing Custom Search instance in the [Bing portal](https://www.customsearch.ai/) | ✅ passed as `WebSearchToolFilters(allowed_domains=…)` | ❌ configure on the Bing Custom Search instance (same as above) | ✅ passed via `get_web_search_tool(allowed_domains=…)` |

For a strict apples-to-apples comparison, pin the Bing Custom Search instance
to the same domain list you've set in `ALLOWED_DOMAINS`.

## Repository layout

```
websearch-bench/
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
        ├── bing_usage.py                # Azure Monitor TotalCalls — validates direct Bing tool billing
        ├── cost_lookup.py               # Azure Cost Management — validates WebSearchTool billing
        ├── auth.py                      # make_credential() — narrowed DefaultAzureCredential
        ├── appinsights.py               # fetch_chat_span() + reconcile_metrics()
        ├── compare.py                   # harness — runs all backends, writes results.csv/html
        ├── report.py                    # HTML report renderer
        └── backends/
            ├── __init__.py              # registry of backends
            ├── foundry_bing_grounding.py
            ├── foundry_bing_grounding_custom.py
            ├── foundry_ws_bing.py
            ├── foundry_ws_bing_fast.py
            ├── foundry_ws_bingcustom.py
            ├── agentfx_ws.py            # label: agentfx-bing
            └── agentfx_ws_cached.py     # label: agentfx-bing-cached
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
  - A deployed reasoning chat model (e.g. `gpt-5.1`) — used as `MODEL`
  - A deployed non-reasoning chat model (e.g. `gpt-4o`) — used as `MODEL_FAST`
  - A **Grounding with Bing Search** connection (for `foundry-bing-grounding`, `foundry-ws-bing*`, `agentfx-bing*`)
  - A **Grounding with Bing Custom Search** connection + instance (for `foundry-bing-grounding-custom`, `foundry-ws-bingcustom`)
- (Optional) An **Application Insights** connection string for tracing + true Bing fan-out reconciliation
- (Optional) A reachable **Redis** instance for `agentfx-bing-cached`

## Setup

```powershell
cd C:\path\to\websearch-bench
uv sync                                  # creates .venv, installs the package + deps
Copy-Item .env.example .env              # then edit .env (see "Environment" below)
az login
az account set --subscription <your-subscription-id>
```

Your signed-in identity needs the **Azure AI User** role (or equivalent) on
the Foundry project.

## Environment

Minimum env vars per backend (set in `.env`):

| Backend | Required env vars |
| --- | --- |
| `foundry-bing-grounding` | `PROJECT_ENDPOINT`, `MODEL`, `BING_CONNECTION_NAME` |
| `foundry-bing-grounding-custom` | `PROJECT_ENDPOINT`, `MODEL`, `BING_CUSTOM_SEARCH_CONNECTION_ID`, `BING_CUSTOM_SEARCH_INSTANCE_NAME` |
| `foundry-ws-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `foundry-ws-bing-fast` | `PROJECT_ENDPOINT`, `MODEL_FAST` (default `gpt-4o`) |
| `foundry-ws-bingcustom` | `PROJECT_ENDPOINT`, `MODEL`, `BING_CUSTOM_SEARCH_CONNECTION_ID`, `BING_CUSTOM_SEARCH_INSTANCE_NAME` |
| `agentfx-bing` | `PROJECT_ENDPOINT`, `MODEL` |
| `agentfx-bing-cached` | above + `REDIS_URL` (and a running Redis) |

### Toggling backends

Every backend has an `ENABLE_<NAME>` flag derived from its label (upper-case,
dashes → underscores). The flag accepts `1` / `true` / `yes` / `on` to enable
and `0` / `false` / `no` / `off` to disable. **All backends are on by default**
— set the flag to `0` only when you want to skip a row.

| Flag | Default |
| --- | --- |
| `ENABLE_FOUNDRY_BING_GROUNDING` | enabled |
| `ENABLE_FOUNDRY_BING_GROUNDING_CUSTOM` | enabled |
| `ENABLE_FOUNDRY_WS_BING` | enabled |
| `ENABLE_FOUNDRY_WS_BING_FAST` | enabled |
| `ENABLE_FOUNDRY_WS_BINGCUSTOM` | enabled |
| `ENABLE_AGENTFX_BING` | enabled |
| `ENABLE_AGENTFX_BING_CACHED` | enabled |

### Optional everywhere

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
- `BING_GROUNDING_USD_PER_1K`, `BING_CUSTOM_USD_PER_1K` — override the
  per-1,000-call pricing (verified default: $35 for both — see
  [`pricing.py`](src/websearch_bench/pricing.py)).

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
  collapsed in a `<details>` block. Open in any browser. Chart.js is
  loaded from a CDN, so the page needs internet to render the charts (the
  table and answers still work offline).
- **`results.csv`** — same metrics minus the answer text, for spreadsheets
  and downstream scripting.

Backends with missing env vars are **skipped with a warning** — they don't
fail the run, and they still appear (greyed out) in both outputs.

### A single backend in isolation

```powershell
uv run python -m websearch_bench.backends.foundry_bing_grounding
uv run python -m websearch_bench.backends.foundry_bing_grounding_custom
uv run python -m websearch_bench.backends.foundry_ws_bing
uv run python -m websearch_bench.backends.foundry_ws_bing_fast
uv run python -m websearch_bench.backends.foundry_ws_bingcustom
uv run python -m websearch_bench.backends.agentfx_ws
uv run python -m websearch_bench.backends.agentfx_ws_cached
```

Each prints the agent's answer plus a normalized usage block (tokens,
web-search calls, total tool calls, latency, estimated USD cost).

## Metrics & cost model

Each run reports a normalized `RunMetrics` row:

| Column | Meaning |
| --- | --- |
| `input_tokens` / `output_tokens` / `total_tokens` | Model usage as reported by the SDK (OpenAI `response.usage`, Foundry's mirror of it, or `agent_framework`'s `usage_details`). |
| `cached_input_tokens` | Portion of `input_tokens` served from Azure OpenAI prompt caching. Billed at the **cached_input** rate (≈10× cheaper). Same number surfaced on the Foundry App Insights span as `gen_ai.usage.cached_tokens`. |
| `web_search_calls` | Number of `web_search_call` items in the response — i.e. distinct outer tool invocations the model emitted. **This is the billable quantity for WebSearchTool / agent_framework backends.** |
| `bing_queries` | True Bing transaction count, **reconciled from App Insights** when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set. <br>• For **Foundry-hosted** backends (`foundry-ws-bing*`, `foundry-ws-bingcustom`) this is `count(role=="tool")` from the server-side `chat` span's `gen_ai.input.messages` array — the only place the real `web.run` fan-out is visible. The Responses API only exposes a summarized `action.queries`. <br>• For **`agentfx-bing*`** the agent_framework client-side span exposes `search_tool_call` parts in `gen_ai.output.messages` — one per *model-level* call. The actual per-Bing-transaction fan-out is performed by Foundry server-side and is **not visible** to client-side instrumentation, so this value equals `web_search_calls` and is a **lower bound** (notes column says so). To see the true fan-out you need Foundry's own App Insights instance. <br>• For the **legacy `foundry-bing-grounding*`** backends there's no fan-out — 1 tool call = 1 Bing transaction. |
| `latency_s` | Wall-clock seconds from sending the request to receiving the final response. |
| `cost_usd` | See cost formula below. |
| `answer_chars` | Length of the agent's final answer text. |

Cost formula:

```
fresh_in  = max(0, input_tokens - cached_input_tokens)
tokens_$  = (fresh_in           / 1000) * model_input_rate
          + (cached_input_tokens/ 1000) * model_cached_input_rate
          + (output_tokens      / 1000) * model_output_rate

tool_$    = (billable_calls / 1000) * tool_rate_per_1k
            # billable_calls = bing_queries for foundry-bing-grounding*
            #                  web_search_calls for everything else

cost      = tokens_$ + tool_$
```

> **Foundry server-side fan-out is reconciled from App Insights.** Foundry's
> `web.run` extension is one billable "tool execution" from the model's POV
> but dispatches multiple Bing transactions internally. Each Bing hit appears
> as a separate `role="tool"` message in the **next** `chat` span's
> `gen_ai.input.messages` array, but the Responses API only exposes a
> summarized `action.queries`. So the bench auto-queries App Insights after
> every Foundry-backed run (using `gen_ai.response.id` as the join key) and
> overwrites `bing_queries` with the truth. Set
> `APPLICATIONINSIGHTS_CONNECTION_STRING` and grant your identity Monitoring
> Reader on the App Insights resource. Without it, the column falls back to
> the `action.queries` lower bound. Typical ingestion lag is 30-90s; the
> reconciler polls for up to 2 minutes.
>
> The fan-out is **variable per run** — the same question may produce
> 2, 14, 17, 23 … Bing hits depending on what the tool decides. The bench
> still bills the WebSearchTool family per *outer* `web_search_call`, since
> that's the quantity Microsoft charges the caller for (the inner fan-out
> isn't separately metered to you).

### Inspecting the raw response

To dump the raw SDK response (or `AgentResponse`) to disk for any run:

```powershell
$env:WEBSEARCH_BENCH_DEBUG="1"     # writes ./debug/<backend>-<timestamp>.json
uv run websearch-bench
```

Open the JSON to verify how the SDK reports `web_search_call` items and their
fan-out. Useful when the `bing_queries` column doesn't match the count you see
on the Foundry App Insights `execute_tool web.run` span — open a real dump and
extend `count_bing_queries_in_openai_output` in `shared.py` with the field
names you find.

### Default tool rate

| Tool family | Backends | Quantity | Rate | Source |
| --- | --- | --- | --- | --- |
| `BingGroundingTool` | `foundry-bing-grounding` | `bing_queries` | **$35 / 1,000** | [Grounding with Bing](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `BingCustomSearchPreviewTool` | `foundry-bing-grounding-custom` | `bing_queries` | **$35 / 1,000** (new SKU; legacy $14 retired Aug 2025) | [Grounding with Bing Custom](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `WebSearchTool` (Bing Web Search) | `foundry-ws-bing`, `foundry-ws-bing-fast`, `agentfx-bing`, `agentfx-bing-cached (miss)` | `web_search_calls` | **$35 / 1,000** | [Grounding with Bing](https://www.microsoft.com/bing/apis/grounding-pricing) |
| `WebSearchTool` (Bing Custom Search) | `foundry-ws-bingcustom` | `web_search_calls` | **$35 / 1,000** | [Grounding with Bing Custom](https://www.microsoft.com/bing/apis/grounding-pricing) |
| Cache hit | `agentfx-bing-cached (hit)` | — | $0 — answer served from Redis | n/a |

Default model token rates (Azure OpenAI Global Standard, USD per **1M** tokens —
source: <https://azure.microsoft.com/pricing/details/azure-openai/>):

| Model | Input | Cached input | Output |
| --- | ---:| ---:| ---:|
| `gpt-5.1` | $1.25 | $0.125 | $10.00 |
| `gpt-5.1-mini` | $0.25 | $0.025 | $2.00 |
| `gpt-4.1` | $2.00 | $0.50 | $8.00 |
| `gpt-4.1-mini` | $0.40 | $0.10 | $1.60 |
| `gpt-4o` | $2.50 | $1.25 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.075 | $0.60 |

Set `MODEL` (every backend) and `MODEL_FAST` (`foundry-ws-bing-fast`) in
`.env` — the harness picks the matching row automatically. Unknown models
fall through with a model-token cost of $0 (only the per-call tool charge is
billed), so add new models to `MODEL_PRICING_PER_1K` in `pricing.py` before
quoting.

### How tool cost is routed (and why it matters)

Per Microsoft's
[Web search with the Responses API docs](https://learn.microsoft.com/azure/foundry/openai/how-to/web-search):

> "Web Search uses Grounding with Bing Search and/or Grounding with Bing
> Custom Search […] Search actions incur tool call costs."

So **all** Foundry grounding tools bill at the same Grounding-with-Bing rate
— but they bill from *different places* and against *different quantities*:

| Tool surface | Backends | Billed where | Quantity used | Rate |
| --- | --- | --- | --- | --- |
| `BingGroundingTool` / `BingCustomSearchPreviewTool` | `foundry-bing-grounding`, `foundry-bing-grounding-custom` | Your **own** `Microsoft.Bing/accounts` resource (its `TotalCalls` metric increments — verifiable via `bing_usage.py`) | `bing_queries` (1 tool call = 1 Bing call, no fan-out) | $35/1K |
| `WebSearchTool` (Foundry-hosted) | `foundry-ws-bing*`, `foundry-ws-bingcustom`, `agentfx-bing*` | Microsoft-managed Bing infra (charge appears on your **Foundry / Cognitive Services** account bill as a "Grounding with Bing Search" line item — `bing_usage.py` does **not** see it; use `cost_lookup.py`) | `web_search_calls` (the *outer* tool call; server-side fan-out isn't separately metered to the caller) | $35/1K |

The crucial nuance: Foundry's server-side `web.run` fan-out (which can turn
one `web_search_call` into many Bing transactions in the App Insights span)
is *not* separately billed to the caller — Microsoft charges per outer
`web_search_call` action. That's why `web_search_calls` is the right quantity
for the WebSearchTool family, not `bing_queries`.

> ⚠️ **Caveat — awaiting vendor confirmation.** Microsoft docs confirm
> WebSearchTool incurs Grounding-with-Bing tool-call costs but don't
> explicitly state the per-call quantity (outer call vs internal fan-out).
> The current `tool_cost` implementation assumes per *outer* call, which
> aligns with how the OpenAI Responses `web_search_call` action is modelled
> and is the only quantity exposed to the caller. Use `cost_lookup.py` (see
> below) to verify against an actual Cost Management line item, and override
> `BING_GROUNDING_USD_PER_1K` if your invoice tells a different story.

## Validating billing

The in-process telemetry can't tell the whole story:

- Legacy direct tools (`foundry-bing-grounding*`) bill against the user's
  Bing resource → use **`bing_usage.py`**.
- WebSearchTool (`foundry-ws-*`, `agentfx-*`) bills against the Foundry
  account → use **`cost_lookup.py`**.

### `bing_usage.py` — for the legacy direct tools

For the legacy direct tools, the billing truth lives on the
`Microsoft.Bing/accounts` resource as the `TotalCalls` platform metric. One
Bing API call = one increment. That's what `bing_usage.py` queries:

```powershell
# Last 30 minutes (default)
uv run python -m websearch_bench.bing_usage

# Last N minutes / hours / days
uv run python -m websearch_bench.bing_usage --since 10m
uv run python -m websearch_bench.bing_usage --since 2h

# Explicit ISO-8601 window
uv run python -m websearch_bench.bing_usage `
  --start 2026-05-24T09:00:00Z --end 2026-05-24T09:15:00Z
```

> ⚠️ **`bing_usage.py` does NOT capture WebSearchTool charges.**
> WebSearchTool / Responses `web_search` routes through Microsoft-managed
> Bing infrastructure, not the user's `Microsoft.Bing/accounts` resource,
> so its `TotalCalls` metric won't move. Use `cost_lookup.py` for that.

### `cost_lookup.py` — for WebSearchTool

`cost_lookup.py` queries Azure Cost Management
(`Microsoft.Consumption/usageDetails`) for the Foundry account over a chosen
time window, so you can answer the key billing question for the
WebSearchTool family:

> For N runs of `foundry-ws-bing`, does Cost Management record N units
> (per outer `web_search_call`) or N × fan-out units (per inner Bing
> transaction)?

```powershell
# Default: last 24h, filter to Bing/Search meters
uv run python -m websearch_bench.cost_lookup

# Controlled experiment: ran the bench 10 times in this window
uv run python -m websearch_bench.cost_lookup `
  --start 2026-05-26T07:00:00Z --end 2026-05-26T07:05:00Z `
  --runs 10

# Restrict to your Foundry RG
uv run python -m websearch_bench.cost_lookup --since 48h `
  --resource-group rg-foundry-prod
```

Required: `AZURE_SUBSCRIPTION_ID` env var (or `--subscription`), `az login`,
and **Cost Management Reader** on the subscription / RG.

> ⏳ **24–48h ingestion lag.** Same-day queries typically return empty —
> that's expected. Run the bench today, query tomorrow.

When `--runs N` is set, the script prints the **qty/run ratio**:

- ≈ **1.0** → billing is per outer `web_search_call` (current bench assumption).
- ≈ your harness's `bing_q` → billing is per inner Bing transaction.

### End-to-end workflow when comparing backends

1. Note the current time.
2. Run `uv run websearch-bench`.
3. Wait ~3 minutes (Azure Monitor metric ingestion lag).
4. Run `uv run python -m websearch_bench.bing_usage --since 5m`.
5. `TotalCalls` should equal the sum of `bing_queries` from your
   `foundry-bing-grounding*` rows **only**. WebSearchTool rows contribute
   zero to this metric — that's expected, not a bug.
6. Wait 24-48h, then run `uv run python -m websearch_bench.cost_lookup
   --start <step-1-time> --end <now> --runs <N>` to verify WebSearchTool
   billing against the Foundry account.

## Change the workload

Every backend reads its workload from `src/websearch_bench/shared.py`. Edit
those module-level constants once and rerun `websearch-bench`:

- `SHARED_QUERY` — the prompt every backend gets.
- `SHARED_INSTRUCTIONS` — system instructions.
- `MODEL` — model used by all backends (override via env var).
- `MODEL_FAST` — non-reasoning model used by `foundry-ws-bing-fast`.
- `USER_COUNTRY` / `USER_REGION` / `USER_CITY` — `user_location` hint.
- `SEARCH_CONTEXT_SIZE` — `"low" | "medium" | "high"`.
- `ALLOWED_DOMAINS` — set in `.env` (comma-separated; full URLs like
  `https://www.sars.gov.za/` are accepted and normalized to hostnames).
  Honored by `foundry-ws-bing`, `foundry-ws-bing-fast`, and `agentfx-bing*`.
  For `foundry-ws-bingcustom` and `foundry-bing-grounding-custom` you must
  set the allowed-domain list on the Bing Custom Search instance itself in
  the [Bing portal](https://www.customsearch.ai/) — the API does not accept
  a `filters` block when a `custom_search_configuration` is set.

## Cache-only backend: extras

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
- **`agent_framework` import errors after `uv sync`** — this repo opts in to
  pre-release packages via `[tool.uv] prerelease = "allow"`; re-run `uv sync`.

## Security

Never commit `.env`, API keys, or connection strings. Rotate any key that has
been pasted into a chat or shared screen.
