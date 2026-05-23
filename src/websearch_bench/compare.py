"""Run the shared query through every backend and print a side-by-side table."""

from __future__ import annotations

import asyncio
import csv
import os
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from websearch_bench.backends import discover
from websearch_bench.report import write_html
from websearch_bench.shared import MODEL, SHARED_QUERY, RunMetrics

console = Console()
RESULTS_CSV = Path.cwd() / "results.csv"
RESULTS_HTML = Path.cwd() / "results.html"

# Columns written to CSV. `answer` is intentionally excluded — it is multi-KB
# free text and belongs in the HTML report, not a spreadsheet cell.
_CSV_COLUMNS = [
    "backend", "model", "input_tokens", "output_tokens", "total_tokens",
    "web_search_calls", "tool_calls", "latency_s", "cost_usd",
    "answer_chars", "notes",
]


def _missing(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not os.getenv(name)]


async def run_all() -> list[RunMetrics]:
    results: list[RunMetrics] = []
    for module in discover():
        label: str = getattr(module, "BACKEND_NAME", module.__name__)
        required: tuple[str, ...] = getattr(module, "REQUIRED_ENV", ())
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
        "backend", "model", "in_tok", "out_tok", "total_tok",
        "web_search", "tool_calls", "latency", "cost", "answer", "notes",
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


async def amain() -> None:
    load_dotenv(override=True)
    results = await run_all()
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
    asyncio.run(amain())


if __name__ == "__main__":
    cli()
