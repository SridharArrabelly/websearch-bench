"""Render a self-contained HTML report for a benchmark run.

The HTML is a single file: inline CSS, all data baked in, one external script
(Chart.js from a CDN) for the bar charts. Everything dynamic is HTML-escaped.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .pricing import (
    BING_CUSTOM_USD_PER_1K,
    BING_GROUNDING_USD_PER_1K,
    MODEL_PRICING_PER_1K,
    OPENAI_WEB_SEARCH_USD_PER_1K,
)
from .shared import RunMetrics

CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


def _fmt(value: object, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if suffix == " USD":
            return f"{value:.4f}{suffix}"
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _row(m: RunMetrics) -> tuple[bool, list[str]]:
    skipped = m.model == "—" or (m.notes or "").startswith(("skipped", "error"))
    cells = [
        m.backend,
        m.model,
        _fmt(m.input_tokens),
        _fmt(m.cached_input_tokens),
        _fmt(m.output_tokens),
        _fmt(m.total_tokens),
        _fmt(m.web_search_calls),
        _fmt(m.bing_queries),
        _fmt(m.latency_s, " s"),
        _fmt(m.cost_usd, " USD"),
        _fmt(m.answer_chars),
        m.notes or "",
    ]
    return skipped, cells


def _chart_payload(results: list[RunMetrics]) -> dict[str, list]:
    """Build the per-backend arrays Chart.js consumes. Skipped runs become 0."""
    labels: list[str] = []
    cost: list[float] = []
    tokens: list[int] = []
    latency: list[float] = []
    for m in results:
        labels.append(m.backend)
        cost.append(float(m.cost_usd) if m.cost_usd is not None else 0.0)
        tokens.append(int(m.total_tokens) if m.total_tokens is not None else 0)
        latency.append(float(m.latency_s) if m.latency_s is not None else 0.0)
    return {"labels": labels, "cost": cost, "tokens": tokens, "latency": latency}


def render_html(
    *,
    results: list[RunMetrics],
    query: str,
    model: str,
    generated_at: datetime,
) -> str:
    rows_html_parts: list[str] = []
    for m in results:
        skipped, cells = _row(m)
        cls = ' class="skipped"' if skipped else ""
        cells_html = "".join(f"<td>{html.escape(c)}</td>" for c in cells)
        rows_html_parts.append(f"<tr{cls}>{cells_html}</tr>")
    rows_html = "\n".join(rows_html_parts)

    answers_html_parts: list[str] = []
    for m in results:
        if not m.answer:
            continue
        backend_label = html.escape(m.backend)
        summary_preview = html.escape((m.answer.splitlines() or [""])[0][:120])
        body = html.escape(m.answer)
        answers_html_parts.append(
            f"<details><summary><strong>{backend_label}</strong>"
            f" — {summary_preview}</summary><pre>{body}</pre></details>"
        )
    answers_html = "\n".join(answers_html_parts) or "<p><em>No answers to display.</em></p>"

    chart_data = _chart_payload(results)
    chart_data_json = json.dumps(chart_data)

    pricing_json = json.dumps(
        {
            "model_per_1k_tokens": MODEL_PRICING_PER_1K,
            "bing_grounding_per_1k_calls": BING_GROUNDING_USD_PER_1K,
            "bing_custom_per_1k_calls": BING_CUSTOM_USD_PER_1K,
            "openai_web_search_per_1k_calls": OPENAI_WEB_SEARCH_USD_PER_1K,
        },
        indent=2,
    )

    generated_iso = generated_at.isoformat(timespec="seconds")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>websearch-bench results</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    color-scheme: light dark;
    --fg: #1a1a1a; --bg: #ffffff; --muted: #666; --accent: #2563eb;
    --border: #e5e7eb; --skipped: #9ca3af; --code-bg: #f6f8fa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e5e7eb; --bg: #0b0d10; --muted: #9ca3af;
             --border: #1f2937; --code-bg: #111827; }}
  }}
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          color: var(--fg); background: var(--bg); margin: 0; padding: 24px; max-width: 1100px; margin-inline: auto; }}
  h1 {{ margin: 0 0 4px; font-size: 22px; }}
  h2 {{ margin-top: 32px; font-size: 16px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
  .meta {{ color: var(--muted); margin-bottom: 16px; font-size: 13px; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--code-bg); }}
  code {{ padding: 2px 6px; border-radius: 4px; }}
  pre {{ padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  th {{ cursor: pointer; user-select: none; }}
  th::after {{ content: " ↕"; color: var(--muted); font-size: 10px; }}
  tr.skipped td {{ color: var(--skipped); font-style: italic; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
  .chart-card {{ border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
  details {{ margin: 8px 0; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; }}
  details summary {{ cursor: pointer; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border);
            color: var(--muted); font-size: 12px; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>websearch-bench results</h1>
<div class="meta">
  Generated <time datetime="{html.escape(generated_iso)}">{html.escape(generated_iso)}</time>
  · Model: <code>{html.escape(model)}</code>
</div>
<p>Query: <code>{html.escape(query)}</code></p>

<h2>Summary</h2>
<table id="summary">
  <thead><tr>
    <th>backend</th><th>model</th><th>in tok</th><th>cached in</th><th>out tok</th><th>total tok</th>
    <th>web search calls</th><th>bing queries</th><th>latency</th><th>cost</th>
    <th>answer chars</th><th>notes</th>
  </tr></thead>
  <tbody>
{rows_html}
  </tbody>
</table>

<h2>Charts</h2>
<div class="charts">
  <div class="chart-card"><canvas id="costChart"></canvas></div>
  <div class="chart-card"><canvas id="tokensChart"></canvas></div>
  <div class="chart-card"><canvas id="latencyChart"></canvas></div>
</div>

<h2>Answers</h2>
{answers_html}

<h2>Pricing inputs (illustrative)</h2>
<details><summary>Show pricing constants used for cost estimation</summary>
<pre>{html.escape(pricing_json)}</pre>
<p>Override via env vars (<code>BING_GROUNDING_USD_PER_1K</code>,
<code>BING_CUSTOM_USD_PER_1K</code>, <code>OPENAI_WEB_SEARCH_USD_PER_1K</code>)
or edit <code>src/websearch_bench/pricing.py</code>. Verify against the official
pages before quoting:
<a href="https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/">Azure OpenAI</a>,
<a href="https://www.microsoft.com/bing/apis/grounding-pricing">Bing Grounding</a>,
<a href="https://www.microsoft.com/bing/apis/pricing">Bing Custom Search</a>,
<a href="https://openai.com/api/pricing/">OpenAI Responses + web_search</a>.</p>
</details>

<footer>
  Values are estimates. websearch-bench is a benchmark, not a quoting tool.
</footer>

<script src="{CHART_JS_CDN}"></script>
<script>
  const data = {chart_data_json};
  const mkChart = (id, label, values) => new Chart(document.getElementById(id), {{
    type: "bar",
    data: {{ labels: data.labels, datasets: [{{ label, data: values }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: label }} }} }},
  }});
  mkChart("costChart", "Cost (USD)", data.cost);
  mkChart("tokensChart", "Total tokens", data.tokens);
  mkChart("latencyChart", "Latency (s)", data.latency);

  // Tiny sortable table — click a header to sort by that column.
  document.querySelectorAll("#summary th").forEach((th, idx) => {{
    let asc = true;
    th.addEventListener("click", () => {{
      const tbody = th.closest("table").querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const num = (s) => {{ const m = String(s).match(/-?\\d+(\\.\\d+)?/); return m ? parseFloat(m[0]) : NaN; }};
      rows.sort((a, b) => {{
        const x = a.cells[idx].textContent.trim();
        const y = b.cells[idx].textContent.trim();
        const nx = num(x), ny = num(y);
        const cmp = !isNaN(nx) && !isNaN(ny) ? nx - ny : x.localeCompare(y);
        return asc ? cmp : -cmp;
      }});
      asc = !asc;
      rows.forEach((r) => tbody.appendChild(r));
    }});
  }});
</script>
</body>
</html>
"""


def write_html(
    results: list[RunMetrics],
    path: Path,
    *,
    query: str,
    model: str,
    generated_at: datetime | None = None,
) -> Path:
    """Write the HTML report to ``path``. Returns the resolved path."""
    rendered = render_html(
        results=results,
        query=query,
        model=model,
        generated_at=generated_at or datetime.now(),
    )
    path.write_text(rendered, encoding="utf-8")
    return path.resolve()
