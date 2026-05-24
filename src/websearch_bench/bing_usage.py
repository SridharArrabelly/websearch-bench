"""Query Azure Monitor for *true* Bing API call counts.

In-process telemetry can't tell the whole story:

* The Foundry SDK backends (``foundry-ws-bing*``) emit a separate App Insights
  span for every Bing tool invocation, so we see fan-out in ``bing_q``.
* The agent_framework backends (``agentfx-bing*``) only emit a single outer
  ``search_tool_calls`` count — Foundry's server-side fan-out is invisible to
  the tracer.

Whatever the SDK reports, the *billing-side* truth lives on the
``Microsoft.Bing/accounts`` resource as the ``TotalCalls`` platform metric.
One Bing API call = one increment, regardless of which SDK surfaced it.

This module queries Azure Monitor for ``TotalCalls`` on your configured Bing
resources over a chosen window. Use it to validate (or refute) the
``bing_q`` numbers in the benchmark table.

Usage::

    # Last 30 minutes (default window is 30m)
    uv run python -m websearch_bench.bing_usage

    # Last N minutes
    uv run python -m websearch_bench.bing_usage --since 10m

    # Explicit ISO-8601 window
    uv run python -m websearch_bench.bing_usage \\
        --start 2026-05-24T09:00:00Z --end 2026-05-24T09:15:00Z

Required env vars (add to ``.env``)::

    AZURE_SUBSCRIPTION_ID=<subscription id>
    BING_GROUNDING_RESOURCE_ID=/subscriptions/.../Microsoft.Bing/accounts/<name>
    BING_CUSTOM_RESOURCE_ID=/subscriptions/.../Microsoft.Bing/accounts/<name>

If the resource-ID env vars are unset, the script auto-discovers every
``Microsoft.Bing/accounts`` resource in the subscription using the resource
graph (requires Reader on the subscription).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from azure.monitor.query.aio import MetricsQueryClient
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from websearch_bench.auth import make_credential

console = Console()


_SINCE_RE = re.compile(r"^(\d+)\s*([smhd])?$", re.IGNORECASE)


def _parse_since(value: str) -> timedelta:
    """Parse '30m', '2h', '45', '1d' → timedelta. Plain int = minutes."""
    m = _SINCE_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid --since value: {value!r} (try '30m', '2h', '1d').")
    n = int(m.group(1))
    unit = (m.group(2) or "m").lower()
    if unit == "s":
        return timedelta(seconds=n)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(days=n)


def _parse_iso(value: str) -> datetime:
    """Parse ISO-8601, force UTC."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resource_kind(resource_id: str) -> str:
    """Best-effort label for the resource (last segment of the id)."""
    return resource_id.rstrip("/").rsplit("/", 1)[-1]


def _discover_resources_via_cli(subscription_id: str) -> list[str]:
    """Fallback resource discovery via the Azure CLI (avoids adding mgmt SDK)."""
    import json
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "az",
                "resource",
                "list",
                "--subscription",
                subscription_id,
                "--resource-type",
                "Microsoft.Bing/accounts",
                "--query",
                "[].id",
                "-o",
                "json",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return list(json.loads(out))
    except FileNotFoundError:
        console.print(
            "[red]Azure CLI ('az') not found and no BING_*_RESOURCE_ID env vars are "
            "set. Either install az or set the env vars explicitly.[/red]"
        )
        return []
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]az resource list failed: {exc.output.strip()}[/red]")
        return []


def _collect_resource_ids() -> list[tuple[str, str]]:
    """Return [(label, resource_id), ...] in the order they should be displayed."""
    ids: list[tuple[str, str]] = []
    grounding = os.getenv("BING_GROUNDING_RESOURCE_ID")
    custom = os.getenv("BING_CUSTOM_RESOURCE_ID")
    if grounding:
        ids.append(("Bing.Grounding", grounding))
    if custom:
        ids.append(("Bing.CustomSearch", custom))
    if ids:
        return ids

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        console.print(
            "[red]AZURE_SUBSCRIPTION_ID not set and no BING_*_RESOURCE_ID env vars "
            "configured. Cannot discover Bing resources.[/red]"
        )
        return []

    console.print(
        "[dim]No BING_*_RESOURCE_ID env vars set; auto-discovering via Azure CLI...[/dim]"
    )
    discovered = _discover_resources_via_cli(subscription_id)
    return [(_resource_kind(rid), rid) for rid in discovered]


async def _total_calls(
    client: MetricsQueryClient,
    resource_id: str,
    start: datetime,
    end: datetime,
) -> int | None:
    """Sum TotalCalls metric over the window. Returns None on error."""
    try:
        result = await client.query_resource(
            resource_uri=resource_id,
            metric_names=["TotalCalls"],
            timespan=(start, end),
            granularity=timedelta(minutes=1),
            aggregations=["Total"],
        )
    except Exception as exc:  # noqa: BLE001 — we want any failure surfaced, not raised
        console.print(f"[red]Metric query failed for {resource_id}: {exc}[/red]")
        return None

    if not result.metrics:
        return 0

    total = 0
    for metric in result.metrics:
        for ts in metric.timeseries:
            for point in ts.data:
                if point.total is not None:
                    total += int(point.total)
    return total


async def query(start: datetime, end: datetime) -> int:
    """Print TotalCalls per Bing resource over [start, end]. Returns exit code."""
    resources = _collect_resource_ids()
    if not resources:
        return 2

    console.print(
        f"[bold]Bing usage window:[/bold] {start.isoformat()} → {end.isoformat()} "
        f"([dim]{(end - start).total_seconds() / 60:.1f} min[/dim])"
    )

    async with make_credential() as credential:
        async with MetricsQueryClient(credential) as client:
            rows = []
            for label, resource_id in resources:
                count = await _total_calls(client, resource_id, start, end)
                rows.append((label, resource_id, count))

    table = Table(show_lines=False, header_style="bold")
    table.add_column("kind")
    table.add_column("resource", overflow="fold")
    table.add_column("TotalCalls", justify="right")

    grand_total = 0
    for label, resource_id, count in rows:
        shown = "n/a" if count is None else str(count)
        if count is not None:
            grand_total += count
        table.add_row(label, _resource_kind(resource_id), shown)

    table.add_row("[bold]sum[/bold]", "", f"[bold]{grand_total}[/bold]")
    console.print(table)
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Query Microsoft.Bing/accounts TotalCalls over a time window."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--since",
        default="30m",
        help="Lookback window relative to now (e.g. '30m', '2h', '1d'). Default 30m.",
    )
    group.add_argument(
        "--start",
        help="ISO-8601 start (e.g. 2026-05-24T09:00:00Z). Requires --end.",
    )
    parser.add_argument(
        "--end",
        help="ISO-8601 end. Defaults to now.",
    )
    args = parser.parse_args()

    if args.start:
        start = _parse_iso(args.start)
        end = _parse_iso(args.end) if args.end else datetime.now(timezone.utc)
    else:
        end = datetime.now(timezone.utc)
        start = end - _parse_since(args.since)

    return asyncio.run(query(start, end))


if __name__ == "__main__":
    sys.exit(main())
