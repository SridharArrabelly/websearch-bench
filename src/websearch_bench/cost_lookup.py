"""Query Azure Cost Management for *true* Web Search tool billing.

Scope: this script audits billing for the ``WebSearchTool`` family
(``foundry-ws-*``, ``agentfx-*`` backends), which route through
Microsoft-managed Bing infrastructure and bill against the customer's
**Foundry / Cognitive Services account** — not their own
``Microsoft.Bing/accounts`` resource. (For the latter, use
``bing_usage.py`` instead.)

The goal is to answer the question:

    For a known number N of ``foundry-ws-bing`` invocations, does the
    Foundry account get charged **N units** (one per outer
    ``web_search_call``) or **N × fan-out** units (one per inner Bing
    transaction)?

How it works
------------
Azure Cost Management is the authoritative source. We query
``Microsoft.Consumption/usageDetails`` via the Azure CLI for a given
time window, filter to Bing/Search-related meters on the Foundry
account, sum ``usageQuantity`` per meter, and print the per-call ratio
if ``--runs N`` is provided.

Important: usage data has **24-48 hour ingestion lag**. Same-day
queries will return empty results — that's expected, not a bug.

Usage::

    # Last 24h, all Bing/Search line items in the subscription
    uv run python -m websearch_bench.cost_lookup --since 24h

    # Specific window, divide by N=10 runs to get per-call ratio
    uv run python -m websearch_bench.cost_lookup \\
        --start 2026-05-26T07:00:00Z --end 2026-05-26T07:05:00Z \\
        --runs 10

    # Restrict to a specific Foundry resource group
    uv run python -m websearch_bench.cost_lookup --since 48h \\
        --resource-group rg-foundry-prod

Required setup
--------------
* ``AZURE_SUBSCRIPTION_ID`` set in env (or passed via ``--subscription``).
* ``az login`` completed with **Cost Management Reader** (or higher) on
  the subscription / resource group.
* Azure CLI available on PATH (``az`` command).

Interpreting the output
-----------------------
* If ``qty/run`` ≈ **1** → billing is per outer ``web_search_call``.
* If ``qty/run`` ≈ your harness's reported ``bing_q`` → billing is per
  underlying Bing transaction (server-side fan-out).
* If ``qty/run`` is anything else → ask the PG to explain.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()


_SINCE_RE = re.compile(r"^(\d+)\s*([smhd])?$", re.IGNORECASE)

# Substrings (case-insensitive) we treat as "this is a Bing / Web Search
# meter worth surfacing". Defensive: Microsoft renames meters periodically,
# so we cast a wide net and let the user eyeball the result.
_BING_HINTS = (
    "bing",
    "grounding",
    "web search",
    "websearch",
    "search action",
)


def _parse_since(value: str) -> timedelta:
    """Parse '30m', '2h', '45', '1d' → timedelta. Plain int = minutes."""
    m = _SINCE_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid --since value: {value!r} (try '30m', '24h', '7d').")
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


@dataclass
class UsageRow:
    meter_category: str
    meter_subcategory: str
    meter_name: str
    consumed_service: str
    instance_name: str
    quantity: float
    cost: float
    currency: str


def _resolve_az() -> str | None:
    """Locate the Azure CLI entry-point.

    ``subprocess`` on Windows does *not* honour ``PATHEXT``, so a bare
    ``["az", ...]`` fails with ``FileNotFoundError`` even when ``az.cmd``
    is on PATH and resolvable from PowerShell. We try ``shutil.which``
    (which *does* honour ``PATHEXT``) for each known launcher name, then
    fall back to common Windows install locations. Returns ``None`` if
    nothing was found.
    """
    for name in ("az", "az.cmd", "az.exe"):
        path = shutil.which(name)
        if path:
            return path
    if os.name == "nt":
        candidates = (
            r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
            r"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        )
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
    return None


def _run_az_consumption(
    *,
    subscription_id: str,
    start: datetime,
    end: datetime,
    resource_group: str | None,
) -> list[UsageRow]:
    """Shell out to ``az consumption usage list`` and parse the result."""
    az = _resolve_az()
    if az is None:
        console.print(
            "[red]Azure CLI ('az') not found.[/red] Tried PATH (via shutil.which) "
            "and common Windows install locations under "
            "'C:\\Program Files\\Microsoft SDKs\\Azure\\CLI2\\wbin'. "
            "Install from https://aka.ms/installazurecliwindows or activate "
            "a shell where 'az' is on PATH."
        )
        return []

    cmd = [
        az,
        "consumption",
        "usage",
        "list",
        "--subscription",
        subscription_id,
        "--start-date",
        start.strftime("%Y-%m-%d"),
        "--end-date",
        end.strftime("%Y-%m-%d"),
        "-o",
        "json",
    ]
    if resource_group:
        cmd.extend(["--resource-group", resource_group])

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        console.print(f"[red]Azure CLI launcher not executable: {az}[/red]")
        return []
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]az consumption usage list failed:\n{exc.output.strip()}[/red]")
        return []

    try:
        raw = json.loads(out)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse az output as JSON: {exc}[/red]")
        return []

    rows: list[UsageRow] = []
    for item in raw:
        rows.append(
            UsageRow(
                meter_category=str(item.get("meterDetails", {}).get("meterCategory") or ""),
                meter_subcategory=str(item.get("meterDetails", {}).get("meterSubCategory") or ""),
                meter_name=str(item.get("meterDetails", {}).get("meterName") or ""),
                consumed_service=str(item.get("consumedService") or ""),
                instance_name=str(item.get("instanceName") or item.get("instanceId") or ""),
                quantity=float(item.get("usageQuantity") or 0.0),
                cost=float(item.get("pretaxCost") or 0.0),
                currency=str(item.get("currency") or "USD"),
            )
        )
    return rows


def _matches_bing(row: UsageRow) -> bool:
    """True if this row plausibly relates to Bing/Web Search billing."""
    haystack = " ".join(
        (row.meter_category, row.meter_subcategory, row.meter_name, row.consumed_service)
    ).lower()
    return any(hint in haystack for hint in _BING_HINTS)


def _aggregate(rows: list[UsageRow]) -> dict[tuple[str, str, str], tuple[float, float, str, set[str]]]:
    """Group by (category, subcategory, meterName); sum qty + cost; collect instances."""
    grouped: dict[tuple[str, str, str], tuple[float, float, str, set[str]]] = {}
    for row in rows:
        key = (row.meter_category, row.meter_subcategory, row.meter_name)
        qty, cost, currency, instances = grouped.get(key, (0.0, 0.0, row.currency, set()))
        instances.add(row.instance_name.rsplit("/", 1)[-1])
        grouped[key] = (qty + row.quantity, cost + row.cost, currency, instances)
    return grouped


def query(
    *,
    subscription_id: str,
    start: datetime,
    end: datetime,
    resource_group: str | None,
    runs: int | None,
    include_all: bool,
) -> int:
    console.print(
        f"[bold]Cost Management window:[/bold] {start.isoformat()} → {end.isoformat()} "
        f"([dim]{(end - start).total_seconds() / 3600:.1f} h[/dim])"
    )
    if resource_group:
        console.print(f"[dim]Resource group filter: {resource_group}[/dim]")
    console.print(
        "[dim]Note: usage records have a 24–48h ingestion lag — same-day queries "
        "may return empty results.[/dim]\n"
    )

    rows = _run_az_consumption(
        subscription_id=subscription_id, start=start, end=end, resource_group=resource_group
    )
    if not rows:
        console.print("[yellow]No usage records returned for the window.[/yellow]")
        return 0

    filtered = rows if include_all else [r for r in rows if _matches_bing(r)]
    if not filtered:
        console.print(
            "[yellow]No Bing / Web Search meter line items found in the window. "
            "Try --include-all to inspect everything, or widen the window if the "
            "data has not ingested yet.[/yellow]"
        )
        return 0

    grouped = _aggregate(filtered)

    table = Table(show_lines=False, header_style="bold")
    table.add_column("category")
    table.add_column("subcategory")
    table.add_column("meterName", overflow="fold")
    table.add_column("qty", justify="right")
    table.add_column("cost", justify="right")
    if runs:
        table.add_column(f"qty/run (N={runs})", justify="right")
    table.add_column("instances", overflow="fold")

    total_qty = 0.0
    total_cost = 0.0
    currency = "USD"
    for (cat, sub, meter), (qty, cost, cur, instances) in sorted(grouped.items()):
        total_qty += qty
        total_cost += cost
        currency = cur or currency
        row_values = [
            cat or "—",
            sub or "—",
            meter or "—",
            f"{qty:,.2f}",
            f"{cost:,.4f}",
        ]
        if runs:
            row_values.append(f"{qty / runs:,.2f}")
        row_values.append(", ".join(sorted(i for i in instances if i)) or "—")
        table.add_row(*row_values)

    sum_row = ["[bold]sum[/bold]", "", "", f"[bold]{total_qty:,.2f}[/bold]", f"[bold]{total_cost:,.4f}[/bold]"]
    if runs:
        sum_row.append(f"[bold]{total_qty / runs:,.2f}[/bold]")
    sum_row.append("")
    table.add_row(*sum_row)

    console.print(table)
    console.print(f"[dim]Currency: {currency}[/dim]")

    if runs:
        ratio = total_qty / runs
        console.print()
        if 0.5 <= ratio <= 1.5:
            console.print(
                f"[bold green]≈ 1 unit per run[/bold green] — billing appears to be "
                f"**per outer `web_search_call`** (ratio={ratio:.2f})."
            )
        elif ratio > 1.5:
            console.print(
                f"[bold yellow]≈ {ratio:.1f} units per run[/bold yellow] — billing appears to "
                f"include **server-side Bing fan-out** (per-inner-transaction)."
            )
        else:
            console.print(
                f"[bold red]≈ {ratio:.2f} units per run[/bold red] — unexpected ratio. "
                f"Check your run count, the window, and the meter filter."
            )
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Query Cost Management for Web Search tool billing on a Foundry account."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--since",
        default="24h",
        help="Lookback window relative to now (e.g. '24h', '48h', '7d'). Default 24h.",
    )
    group.add_argument(
        "--start",
        help="ISO-8601 start (e.g. 2026-05-26T07:00:00Z). Requires --end.",
    )
    parser.add_argument("--end", help="ISO-8601 end. Defaults to now.")
    parser.add_argument(
        "--subscription",
        help="Azure subscription ID. Defaults to AZURE_SUBSCRIPTION_ID env var.",
    )
    parser.add_argument(
        "--resource-group",
        help="Restrict query to a single resource group (e.g. the Foundry RG).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="If set, divide total usageQuantity by this number to show the per-call ratio.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Don't filter on Bing/Search keywords — show every meter in the window.",
    )
    args = parser.parse_args()

    import os

    subscription_id = args.subscription or os.getenv("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        console.print(
            "[red]AZURE_SUBSCRIPTION_ID not set and --subscription not provided.[/red]"
        )
        return 2

    if args.start:
        start = _parse_iso(args.start)
        end = _parse_iso(args.end) if args.end else datetime.now(timezone.utc)
    else:
        end = datetime.now(timezone.utc)
        start = end - _parse_since(args.since)

    if end <= start:
        console.print("[red]--end must be after --start.[/red]")
        return 2

    return query(
        subscription_id=subscription_id,
        start=start,
        end=end,
        resource_group=args.resource_group,
        runs=args.runs,
        include_all=args.include_all,
    )


if __name__ == "__main__":
    sys.exit(main())
