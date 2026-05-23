"""Run the shared query through every backend and print a side-by-side table."""

from __future__ import annotations

import asyncio
import csv
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from websearch_bench.backends import discover
from websearch_bench.appinsights import reconcile_metrics
from websearch_bench.report import write_html
from websearch_bench.shared import MODEL, SHARED_QUERY, RunMetrics

console = Console()
RESULTS_CSV = Path.cwd() / "results.csv"
RESULTS_HTML = Path.cwd() / "results.html"

# Columns written to CSV. `answer` is intentionally excluded — it is multi-KB
# free text and belongs in the HTML report, not a spreadsheet cell.
_CSV_COLUMNS = [
    "backend", "model", "input_tokens", "cached_input_tokens", "output_tokens",
    "total_tokens", "web_search_calls", "bing_queries",
    "latency_s", "cost_usd", "answer_chars", "notes",
]


def _missing(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not os.getenv(name)]


def _enable_var_for(label: str) -> str:
    """ENABLE_FOUNDRY_WS_BINGCUSTOM for label 'foundry-ws-bingcustom', etc."""
    return "ENABLE_" + label.upper().replace("-", "_")


def _is_disabled(label: str) -> bool:
    """Honor central opt-out: ENABLE_<NAME>=0/false/no/off skips the backend.

    Unset or any other value means 'not explicitly disabled' — the backend's
    own ``enabled()`` hook (if any) still gets the final say.
    """
    val = os.getenv(_enable_var_for(label), "").strip().lower()
    return val in ("0", "false", "no", "off")


async def run_all() -> list[RunMetrics]:
    load_dotenv(override=True)

    # Detect stale/typo'd ENABLE_* vars so a rename doesn't silently run
    # everything because the user's .env still uses the old names.
    valid_enable_vars = {_enable_var_for(getattr(m, "BACKEND_NAME", m.__name__)) for m in discover()}
    stray = sorted(
        k for k in os.environ
        if k.startswith("ENABLE_") and k not in valid_enable_vars
    )
    if stray:
        console.print(
            "[bold yellow]Warning:[/bold yellow] unknown ENABLE_* env vars "
            "(ignored; will NOT toggle anything):"
        )
        for k in stray:
            console.print(f"  [yellow]{k}[/yellow]={os.environ[k]!r}")
        console.print(
            f"  [dim]Valid names: {', '.join(sorted(valid_enable_vars))}[/dim]\n"
        )

    results: list[RunMetrics] = []
    for module in discover():
        label: str = getattr(module, "BACKEND_NAME", module.__name__)
        required: tuple[str, ...] = getattr(module, "REQUIRED_ENV", ())
        if _is_disabled(label):
            reason = f"set {_enable_var_for(label)}=1 to enable"
            console.print(f"[yellow]Skipping {label}: disabled via {_enable_var_for(label)}[/yellow]")
            results.append(RunMetrics(backend=label, model="—", notes=f"skipped ({reason})"))
            continue
        enabled_fn = getattr(module, "enabled", None)
        if callable(enabled_fn):
            enabled, reason = enabled_fn()
            if not enabled:
                console.print(f"[yellow]Skipping {label}: {reason}[/yellow]")
                results.append(RunMetrics(backend=label, model="—", notes=f"skipped ({reason})"))
                continue
        missing = _missing(required)
        if missing:
            console.print(f"[yellow]Skipping {label}: missing env {', '.join(missing)}[/yellow]")
            results.append(
                RunMetrics(backend=label, model="—", notes=f"skipped (missing {', '.join(missing)})")
            )
            continue

        console.rule(f"[bold]{label}")
        try:
            metrics = await module.run()
            results.append(metrics)
        except Exception as exc:
            console.print(f"[red]{label} failed: {exc}[/red]")
            traceback.print_exc()
            results.append(RunMetrics(backend=label, model="—", notes=f"error: {exc}"))
    return results


def render(results: list[RunMetrics]) -> None:
    table = Table(title=f"Web-search comparison — query: {SHARED_QUERY!r}")
    for col in [
        "backend", "model", "in_tok", "cached", "out_tok", "total_tok",
        "ws_calls", "bing_q", "latency", "cost", "answer", "notes",
    ]:
        table.add_column(col)
    for r in results:
        table.add_row(*r.as_row())
    console.print(table)


def write_csv(results: list[RunMetrics], path: Path) -> Path:
    if not results:
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k in _CSV_COLUMNS}
            writer.writerow(row)
    return path.resolve()


async def reconcile_all(results: list[RunMetrics]) -> None:
    """After all backends have run, batch-reconcile against App Insights.

    Telemetry has a 1-5 min ingestion lag, so reconciling inline (in each
    backend's run()) usually times out for everything except the first
    backend. Running a single pass at the end gives the earlier backends
    plenty of time to ingest while the later ones execute. We still poll
    each id for up to 3 min in case App Insights is slow.
    """
    targets = [r for r in results if getattr(r, "response_id", None)]
    if not targets:
        return
    console.rule("[bold]Reconciling against App Insights")
    for r in targets:
        console.print(f"[dim]-> {r.backend}: {r.response_id}[/dim]")

    async def _one(r: RunMetrics) -> None:
        await reconcile_metrics(r, r.response_id, console=None, timeout_s=180)
        notes = r.notes or ""
        if "App Insights chat span" in notes and "0 tool msgs" not in notes:
            status = f"bing_queries={r.bing_queries}  cost=${r.cost_usd}"
        elif "0 tool msgs" in notes:
            status = "client-side span (no tool msgs visible) — lower bound retained"
        else:
            status = "not reconciled (lower bound retained)"
        console.print(f"[dim]   {r.backend}: {status}[/dim]")

    await asyncio.gather(*(_one(r) for r in targets))


def _setup_tracing() -> None:
    """Enable Azure Monitor + agent_framework OTel instrumentation, if available.

    Without this the agent_framework backends emit no spans and reconcile
    can't find their chat span in App Insights. The Foundry-hosted backends
    don't need this — their server-side instrumentation emits spans regardless.
    Safe to call when APPLICATIONINSIGHTS_CONNECTION_STRING is unset (no-op).
    """
    if not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        return
    # Enable the Azure SDK's experimental GenAI tracing so chat/agent
    # spans carry the full gen_ai.* attributes (input.messages, etc.).
    # Must be set BEFORE the SDK clients are instantiated.
    os.environ.setdefault("AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING", "true")
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from agent_framework.observability import create_resource, enable_instrumentation
    except ImportError:
        return
    try:
        configure_azure_monitor(
            connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
            resource=create_resource(),
            enable_live_metrics=True,
        )
        enable_instrumentation(enable_sensitive_data=True)
    except Exception as exc:
        console.print(f"[yellow]Tracing setup skipped: {exc}[/yellow]")


async def amain() -> None:
    load_dotenv(override=True)
    _setup_tracing()
    results = await run_all()
    await reconcile_all(results)
    console.rule("[bold]Summary")
    render(results)

    csv_path = write_csv(results, RESULTS_CSV)
    html_path = write_html(
        results,
        RESULTS_HTML,
        query=SHARED_QUERY,
        model=MODEL,
        generated_at=datetime.now(),
    )
    console.print(f"[dim]Wrote CSV : [link=file:///{csv_path.as_posix()}]{csv_path}[/link][/dim]")
    console.print(f"[dim]Wrote HTML: [link=file:///{html_path.as_posix()}]{html_path}[/link][/dim]")


def cli() -> None:
    """Entry point exposed as the ``websearch-bench`` console script."""
    # Force UTF-8 on stdout/stderr so agent answers containing characters
    # outside cp1252 (arrows, em-dashes, currency symbols) don't crash rich
    # on Windows legacy consoles.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    asyncio.run(amain())


if __name__ == "__main__":
    cli()
