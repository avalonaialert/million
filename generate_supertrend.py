#!/usr/bin/env python3
"""Build index.html listing S&P 500 stocks in a bullish Supertrend on Heikin Ashi candles.

For each CSV in the data directory (as written by download_sp500.py) this:
  1. converts the daily OHLC bars to Heikin Ashi candles,
  2. computes Supertrend on those candles (ATR uses Wilder smoothing, like TradingView),
  3. keeps symbols whose latest Supertrend direction is up.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Only the most recent bars are needed. Heikin Ashi open and ATR are recursive,
# but their dependence on old bars fades quickly, so 400 bars is plenty.
LOOKBACK_BARS = 400
CHART_BARS = 40
# Skip symbols whose data stops this many days before the newest file
# (e.g. companies that have left the index).
STALE_DAYS = 7


@dataclass
class Signal:
    symbol: str
    date: str
    close: float
    ha_close: float
    supertrend: float
    distance_pct: float
    bull_since: str
    bars_in_trend: int
    change_pct: float
    chart: str


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[col].to_numpy(dtype=float) for col in ("Open", "High", "Low", "Close"))
    ha_close = (o + h + l + c) / 4
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    return pd.DataFrame(
        {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close},
        index=df.index,
    )


def supertrend(ha: pd.DataFrame, period: int, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (supertrend line, direction) where direction is 1 for bull, -1 for bear."""
    high, low, close = ha["high"].to_numpy(), ha["low"].to_numpy(), ha["close"].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    # Wilder's moving average (RMA), seeded with a simple average.
    atr = np.full_like(close, np.nan)
    atr[period - 1] = true_range[:period].mean()
    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period

    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    upper = basic_upper.copy()
    lower = basic_lower.copy()
    direction = np.zeros(len(close), dtype=int)
    line = np.full_like(close, np.nan)

    start = period - 1
    direction[start] = 1 if close[start] > hl2[start] else -1
    line[start] = lower[start] if direction[start] == 1 else upper[start]
    for i in range(start + 1, len(close)):
        if not (basic_upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1]):
            upper[i] = upper[i - 1]
        if not (basic_lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1]):
            lower[i] = lower[i - 1]
        if direction[i - 1] == -1:
            direction[i] = 1 if close[i] > upper[i] else -1
        else:
            direction[i] = -1 if close[i] < lower[i] else 1
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return line, direction


def mini_chart(ha: pd.DataFrame, line: np.ndarray, direction: np.ndarray) -> str:
    """Inline SVG of the last CHART_BARS Heikin Ashi candles plus the Supertrend line."""
    ha = ha.iloc[-CHART_BARS:]
    line = line[-CHART_BARS:]
    direction = direction[-CHART_BARS:]
    width, height, pad = 160, 40, 2
    lo = min(ha["low"].min(), np.nanmin(line))
    hi = max(ha["high"].max(), np.nanmax(line))
    span = (hi - lo) or 1.0

    def y(value: float) -> float:
        return round(pad + (hi - value) / span * (height - 2 * pad), 1)

    step = width / len(ha)
    body = max(step * 0.6, 1)
    up_path, down_path = [], []
    for i, (o, h, l, c) in enumerate(ha[["open", "high", "low", "close"]].itertuples(index=False)):
        x = round(i * step + step / 2, 1)
        top, bottom = y(max(o, c)), y(min(o, c))
        segment = (
            f"M{x} {y(h)}V{y(l)}"
            f"M{round(x - body / 2, 1)} {top}h{round(body, 1)}V{max(bottom, top + 0.5)}"
            f"h{-round(body, 1)}Z"
        )
        (up_path if c >= o else down_path).append(segment)

    # Draw the Supertrend line only through the current (bullish) run, so it
    # doesn't jump between the upper and lower bands.
    run_start = len(direction) - 1
    while run_start > 0 and direction[run_start - 1] == 1:
        run_start -= 1
    points = " ".join(
        f"{round(i * step + step / 2, 1)},{y(line[i])}" for i in range(run_start, len(line))
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Last {len(ha)} Heikin Ashi candles">'
        f'<polyline class="st" points="{points}"/>'
        f'<path class="up" d="{"".join(up_path)}"/>'
        f'<path class="down" d="{"".join(down_path)}"/>'
        "</svg>"
    )


def analyze(path: Path, period: int, multiplier: float) -> tuple[Signal | None, str | None]:
    """Return (signal if bullish, last date) for one CSV."""
    df = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close"])
    df = df.dropna().tail(LOOKBACK_BARS)
    if len(df) < period + 2:
        return None, None
    dates = df["Date"].str[:10].to_numpy()
    ha = heikin_ashi(df)
    line, direction = supertrend(ha, period, multiplier)
    last_date = dates[-1]
    if direction[-1] != 1:
        return None, last_date

    flips = np.flatnonzero(direction[:-1] != 1)
    flip_index = flips[-1] + 1 if len(flips) else period - 1
    close = df["Close"].to_numpy()
    signal = Signal(
        symbol=path.stem,
        date=last_date,
        close=float(close[-1]),
        ha_close=float(ha["close"].iloc[-1]),
        supertrend=float(line[-1]),
        distance_pct=float((ha["close"].iloc[-1] / line[-1] - 1) * 100),
        bull_since=dates[flip_index],
        bars_in_trend=int(len(direction) - flip_index),
        change_pct=float((close[-1] / close[-2] - 1) * 100),
        chart=mini_chart(ha, line, direction),
    )
    return signal, last_date


def render(signals: list[Signal], total: int, as_of: str, period: int, multiplier: float) -> str:
    rows = []
    for s in signals:
        fresh = ' class="fresh"' if s.bars_in_trend <= 3 else ""
        change_class = "pos" if s.change_pct >= 0 else "neg"
        rows.append(
            f"<tr{fresh}>"
            f'<td data-v="{html.escape(s.symbol)}"><a href="https://finance.yahoo.com/quote/'
            f'{html.escape(s.symbol)}" target="_blank" rel="noopener">{html.escape(s.symbol)}</a></td>'
            f'<td class="num" data-v="{s.close}">{s.close:,.2f}</td>'
            f'<td class="num {change_class}" data-v="{s.change_pct}">{s.change_pct:+.2f}%</td>'
            f'<td class="num" data-v="{s.ha_close}">{s.ha_close:,.2f}</td>'
            f'<td class="num" data-v="{s.supertrend}">{s.supertrend:,.2f}</td>'
            f'<td class="num" data-v="{s.distance_pct}">{s.distance_pct:.2f}%</td>'
            f'<td data-v="{s.bull_since}">{s.bull_since}</td>'
            f'<td class="num" data-v="{s.bars_in_trend}">{s.bars_in_trend}</td>'
            f"<td>{s.chart}</td>"
            "</tr>"
        )
    fresh_count = sum(s.bars_in_trend <= 3 for s in signals)
    return TEMPLATE.format(
        count=len(signals),
        total=total,
        pct=round(len(signals) / total * 100) if total else 0,
        fresh=fresh_count,
        as_of=html.escape(as_of),
        period=period,
        multiplier=json.dumps(multiplier),
        rows="\n".join(rows),
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heikin Ashi Supertrend Bulls</title>
<style>
:root {{
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66;
  --line: #e4e4df; --accent: #1f7a4d; --up: #1f9d5c; --down: #d0473d;
  --fresh: #eaf6ef; --st: #2b6cb0;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #121412; --panel: #1a1d1a; --text: #e8e8e3; --muted: #9a9a92;
    --line: #2c302c; --accent: #4cc38a; --up: #3fbf7a; --down: #ef6a5e;
    --fresh: #18301f; --st: #6ea8f0;
    color-scheme: dark;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 48px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); margin: 0 0 24px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
.stat {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; min-width: 140px; flex: 1 1 140px;
}}
.stat b {{ display: block; font-size: 22px; font-variant-numeric: tabular-nums; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.tools {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }}
input[type=search] {{
  flex: 1 1 220px; max-width: 320px; padding: 8px 12px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text); font: inherit;
}}
label {{ color: var(--muted); display: flex; gap: 6px; align-items: center; }}
.table-wrap {{
  overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
}}
table {{ border-collapse: collapse; width: 100%; min-width: 820px; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
tbody tr:last-child td {{ border-bottom: 0; }}
th {{
  position: sticky; top: 0; background: var(--panel); font-size: 12px; font-weight: 600;
  color: var(--muted); cursor: pointer; user-select: none;
}}
th[aria-sort=ascending]::after {{ content: " \\25B2"; }}
th[aria-sort=descending]::after {{ content: " \\25BC"; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td a {{ color: var(--text); font-weight: 600; text-decoration: none; }}
td a:hover {{ color: var(--accent); text-decoration: underline; }}
tr.fresh td {{ background: var(--fresh); }}
tr.fresh td:first-child a::after {{
  content: "new"; margin-left: 6px; font-size: 10px; font-weight: 600; color: var(--accent);
  border: 1px solid var(--accent); border-radius: 4px; padding: 0 4px; vertical-align: 1px;
}}
.pos {{ color: var(--up); }} .neg {{ color: var(--down); }}
.spark {{ width: 160px; height: 40px; display: block; }}
.spark .up {{ fill: var(--up); stroke: var(--up); stroke-width: 1; }}
.spark .down {{ fill: var(--down); stroke: var(--down); stroke-width: 1; }}
.spark .st {{ fill: none; stroke: var(--st); stroke-width: 1.5; stroke-dasharray: 3 2; }}
.empty {{ padding: 32px; text-align: center; color: var(--muted); }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 16px; }}
</style>
</head>
<body>
<main>
  <h1>Heikin Ashi Supertrend Bulls</h1>
  <p class="sub">S&amp;P 500 stocks whose Supertrend ({period}, {multiplier}) on daily Heikin Ashi candles is bullish &middot; data as of {as_of}</p>

  <div class="stats">
    <div class="stat"><b>{count}</b><span>bullish stocks</span></div>
    <div class="stat"><b>{pct}%</b><span>of {total} tracked</span></div>
    <div class="stat"><b>{fresh}</b><span>turned bullish in last 3 bars</span></div>
  </div>

  <div class="tools">
    <input type="search" id="q" placeholder="Filter symbols" aria-label="Filter symbols">
    <label><input type="checkbox" id="freshOnly"> New signals only</label>
  </div>

  <div class="table-wrap">
    <table id="t">
      <thead><tr>
        <th data-type="text">Symbol</th>
        <th class="num" data-type="num">Close</th>
        <th class="num" data-type="num">Day</th>
        <th class="num" data-type="num">HA close</th>
        <th class="num" data-type="num">Supertrend</th>
        <th class="num" data-type="num">Above ST</th>
        <th data-type="text">Bull since</th>
        <th class="num" data-type="num" aria-sort="ascending">Bars</th>
        <th>Last 40 HA candles</th>
      </tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
    <div class="empty" id="empty" hidden>No matching stocks.</div>
  </div>
  <footer>"Above ST" is how far the Heikin Ashi close sits above the Supertrend line. Not investment advice.</footer>
</main>
<script>
(() => {{
  const table = document.getElementById("t");
  const body = table.tBodies[0];
  const headers = [...table.tHead.rows[0].cells];
  const q = document.getElementById("q");
  const freshOnly = document.getElementById("freshOnly");
  const empty = document.getElementById("empty");

  function sortBy(index, dir) {{
    const type = headers[index].dataset.type;
    const rows = [...body.rows];
    rows.sort((a, b) => {{
      const x = a.cells[index].dataset.v, y = b.cells[index].dataset.v;
      const cmp = type === "num" ? x - y : x.localeCompare(y);
      return dir === "ascending" ? cmp : -cmp;
    }});
    headers.forEach(h => h.removeAttribute("aria-sort"));
    headers[index].setAttribute("aria-sort", dir);
    body.append(...rows);
  }}

  headers.forEach((h, i) => {{
    if (!h.dataset.type) return;
    h.addEventListener("click", () => {{
      sortBy(i, h.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending");
    }});
  }});

  function filter() {{
    const term = q.value.trim().toUpperCase();
    let shown = 0;
    for (const row of body.rows) {{
      const match = row.cells[0].dataset.v.includes(term) &&
        (!freshOnly.checked || row.classList.contains("fresh"));
      row.hidden = !match;
      shown += match;
    }}
    empty.hidden = shown > 0;
  }}
  q.addEventListener("input", filter);
  freshOnly.addEventListener("change", filter);

  sortBy(7, "ascending");
  filter();
}})();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV directory (default: data/)")
    parser.add_argument("--output", type=Path, default=Path("index.html"), help="output file (default: index.html)")
    parser.add_argument("--period", type=int, default=10, help="Supertrend ATR period (default: 10)")
    parser.add_argument("--multiplier", type=float, default=3.0, help="Supertrend ATR multiplier (default: 3)")
    args = parser.parse_args()
    if args.period < 1 or args.multiplier <= 0:
        parser.error("--period must be >= 1 and --multiplier must be > 0")

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        parser.error(f"no CSV files found in {args.data_dir}")

    results: list[tuple[Signal | None, str]] = []
    for path in files:
        try:
            signal, last_date = analyze(path, args.period, args.multiplier)
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if last_date:
            results.append((signal, last_date))

    as_of = max(date for _, date in results)
    cutoff = (pd.Timestamp(as_of) - pd.Timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")
    current = [(s, d) for s, d in results if d >= cutoff]
    signals = sorted((s for s, _ in current if s), key=lambda s: (s.bars_in_trend, s.symbol))

    args.output.write_text(render(signals, len(current), as_of, args.period, args.multiplier), encoding="utf-8")
    print(f"{len(signals)} of {len(current)} stocks bullish as of {as_of} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
