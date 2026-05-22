"""Run the shared query through every backend and print a side-by-side table."""

from __future__ import annotations

import asyncio
import csv
import os
import traceback
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from websearch_bench.backends import discover
from websearch_bench.shared import SHARED_QUERY, RunMetrics

console = Console()
RESULTS_CSV = Path.cwd() / "results.csv"


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
        "search_calls", "latency", "cost", "answer", "notes",
    ]:
        table.add_column(col)
    for r in results:
        table.add_row(*r.as_row())
    console.print(table)


def write_csv(results: list[RunMetrics], path: Path) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    console.print(f"[dim]Wrote {path}[/dim]")


async def amain() -> None:
    load_dotenv(override=True)
    results = await run_all()
    console.rule("[bold]Summary")
    render(results)
    write_csv(results, RESULTS_CSV)


def cli() -> None:
    """Entry point exposed as the ``websearch-bench`` console script."""
    asyncio.run(amain())


if __name__ == "__main__":
    cli()
