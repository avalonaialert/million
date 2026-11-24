#!/usr/bin/env python3
"""Build index.html listing S&P 500 stocks in a bullish Supertrend above the yearly VWAP band.

For each CSV in the data directory (as written by download_sp500.py) this:
  1. converts the daily OHLC bars to Heikin Ashi candles,
  2. computes Supertrend on those candles (ATR uses Wilder smoothing, like TradingView),
  3. computes a VWAP anchored to each calendar year on those candles, using
     (high + low + close) / 3 as the source, with an upper band of
     VWAP + N standard deviations (TradingView's VWAP bands),
  4. keeps symbols whose Supertrend is bullish and above the VWAP upper band.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
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
    st_change_pct: float  # vs previous bar; NaN on the bar the trend flipped
    sma3_change_pct: float  # SMA(3) of Heikin Ashi close, vs previous bar
    sma9_change_pct: float  # SMA(9) of Heikin Ashi close, vs previous bar
    vwap: float
    vwap_upper: float
    above_band_pct: float
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


def yearly_vwap(
    ha: pd.DataFrame, volume: np.ndarray, years: np.ndarray, band: float
) -> tuple[np.ndarray, np.ndarray]:
    """VWAP of Heikin Ashi (H+L+C)/3, reset each calendar year, and its upper band.

    The band matches TradingView: VWAP + band * sqrt(sum(v*src^2)/sum(v) - VWAP^2),
    with all sums restarting at the first bar of each year.
    """
    source = ((ha["high"] + ha["low"] + ha["close"]) / 3).to_numpy()
    frame = pd.DataFrame({"year": years, "v": volume, "pv": source * volume, "p2v": source**2 * volume})
    sums = frame.groupby("year", sort=False)[["v", "pv", "p2v"]].cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = (sums["pv"] / sums["v"]).to_numpy()
        variance = (sums["p2v"] / sums["v"]).to_numpy() - vwap**2
    stdev = np.sqrt(np.clip(variance, 0, None))
    return vwap, vwap + band * stdev


def mini_chart(
    ha: pd.DataFrame, line: np.ndarray, direction: np.ndarray, upper: np.ndarray, years: np.ndarray
) -> str:
    """Inline SVG of the last CHART_BARS Heikin Ashi candles, Supertrend and VWAP upper band."""
    ha = ha.iloc[-CHART_BARS:]
    line = line[-CHART_BARS:]
    direction = direction[-CHART_BARS:]
    upper = upper[-CHART_BARS:]
    years = years[-CHART_BARS:]
    width, height, pad = 160, 40, 2
    # The band restarts each January, so only draw it for the current year.
    band_start = int(np.argmax(years == years[-1]))
    lo = min(ha["low"].min(), np.nanmin(line), np.nanmin(upper[band_start:]))
    hi = max(ha["high"].max(), np.nanmax(line), np.nanmax(upper[band_start:]))
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
    def polyline(values: np.ndarray, start: int) -> str:
        return " ".join(
            f"{round(i * step + step / 2, 1)},{y(values[i])}"
            for i in range(start, len(values))
            if not np.isnan(values[i])
        )

    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Last {len(ha)} Heikin Ashi candles with Supertrend and VWAP upper band">'
        f'<polyline class="band" points="{polyline(upper, band_start)}"/>'
        f'<polyline class="st" points="{polyline(line, run_start)}"/>'
        f'<path class="up" d="{"".join(up_path)}"/>'
        f'<path class="down" d="{"".join(down_path)}"/>'
        "</svg>"
    )


@dataclass
class Result:
    last_date: str
    bullish: bool
    above_band: bool  # bullish and Supertrend > VWAP upper band
    signal: Signal | None  # set for every bullish stock


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"]).dropna()


def analyze(path: Path, period: int, multiplier: float, band: float) -> Result | None:
    return analyze_frame(load(path), path.stem, period, multiplier, band)


def analyze_frame(
    df: pd.DataFrame, symbol: str, period: int, multiplier: float, band: float
) -> Result | None:
    df = df.tail(LOOKBACK_BARS)
    if len(df) < period + 2:
        return None
    dates = df["Date"].str[:10].to_numpy()
    years = df["Date"].str[:4].to_numpy()
    ha = heikin_ashi(df)
    line, direction = supertrend(ha, period, multiplier)
    vwap, upper = yearly_vwap(ha, df["Volume"].to_numpy(dtype=float), years, band)
    last_date = dates[-1]
    if direction[-1] != 1:
        return Result(last_date, False, False, None)
    # NaN (e.g. no volume yet this year) compares False, so it never passes.
    above_band = bool(line[-1] > upper[-1])

    flips = np.flatnonzero(direction[:-1] != 1)
    flip_index = flips[-1] + 1 if len(flips) else period - 1
    close = df["Close"].to_numpy()
    sma3 = ha["close"].rolling(3).mean().to_numpy()
    sma9 = ha["close"].rolling(9).mean().to_numpy()
    signal = Signal(
        symbol=symbol,
        date=last_date,
        close=float(close[-1]),
        ha_close=float(ha["close"].iloc[-1]),
        supertrend=float(line[-1]),
        st_change_pct=float((line[-1] / line[-2] - 1) * 100) if direction[-2] == 1 else float("nan"),
        sma3_change_pct=float((sma3[-1] / sma3[-2] - 1) * 100),
        sma9_change_pct=float((sma9[-1] / sma9[-2] - 1) * 100),
        vwap=float(vwap[-1]),
        vwap_upper=float(upper[-1]),
        above_band_pct=float((line[-1] / upper[-1] - 1) * 100),
        bull_since=dates[flip_index],
        bars_in_trend=int(len(direction) - flip_index),
        change_pct=float((close[-1] / close[-2] - 1) * 100),
        chart=mini_chart(ha, line, direction, upper, years),
    )
    return Result(last_date, True, above_band, signal)


@dataclass
class Flipped:
    signal: Signal
    in_today: bool
    days_listed: int
    first_listed: str
    last_listed: str


def find_flipped(results: list[Result], history: dict[str, list[dict]], as_of: str) -> list[Flipped]:
    """Symbols ever saved in signals.json that turned bullish again after first being
    listed (the current bull run began after the first listed date) and whose
    Supertrend line rose on today's bar."""
    listed: dict[str, list[str]] = {}
    for day, stocks in sorted(history.items()):
        for stock in stocks:
            listed.setdefault(stock["symbol"], []).append(day)
    today = {stock["symbol"] for stock in history.get(as_of, [])}
    flipped = []
    for r in results:
        s = r.signal
        if not s or s.date != as_of or s.symbol not in listed:
            continue
        days = listed[s.symbol]
        # st_change_pct is NaN on the flip bar itself, so that never counts as rising.
        if s.bull_since > days[0] and s.st_change_pct > 0:
            flipped.append(Flipped(s, s.symbol in today, len(days), days[0], days[-1]))
    return sorted(flipped, key=lambda f: (f.signal.bars_in_trend, f.signal.symbol))


def find_rising(signals: list[Signal]) -> list[Signal]:
    """Over-band stocks whose Supertrend, SMA(3) and SMA(9) all rose on today's bar.

    NaN (e.g. the Supertrend's flip bar) compares False, so it never counts as rising.
    """
    return [
        s for s in signals
        if s.st_change_pct > 0 and s.sma3_change_pct > 0 and s.sma9_change_pct > 0
    ]


def symbol_cell(symbol: str) -> str:
    sym = html.escape(symbol)
    return (
        f'<td data-v="{sym}"><a href="https://finance.yahoo.com/quote/{sym}" '
        f'target="_blank" rel="noopener">{sym}</a></td>'
    )


def num_cell(value: float, text: str, extra: str = "") -> str:
    return f'<td class="num{extra}" data-v="{value}">{text}</td>'


def company_cell(symbol: str, companies: dict[str, str]) -> str:
    name = html.escape(companies.get(symbol, ""))
    return f'<td class="company" data-v="{name}" title="{name}">{name}</td>'


def price_cells(s: Signal) -> list[str]:
    change_class = " pos" if s.change_pct >= 0 else " neg"
    return [
        num_cell(s.close, f"{s.close:,.2f}"),
        num_cell(s.change_pct, f"{s.change_pct:+.2f}%", change_class),
    ]


def render(
    signals: list[Signal], rising: list[Signal], flipped: list[Flipped], companies: dict[str, str],
    watched: int, total: int, bullish: int,
    as_of: str, period: int, multiplier: float, band: float,
) -> str:
    signal_rows = []
    for s in signals:
        fresh = ' class="fresh"' if s.bars_in_trend <= 3 else ""
        cells = [
            symbol_cell(s.symbol), company_cell(s.symbol, companies), *price_cells(s),
            num_cell(s.bars_in_trend, str(s.bars_in_trend)),
            f"<td>{s.chart}</td>",
        ]
        signal_rows.append(f"<tr{fresh}>{''.join(cells)}</tr>")

    rising_rows = []
    for s in rising:
        cells = [
            symbol_cell(s.symbol), company_cell(s.symbol, companies), *price_cells(s),
            num_cell(s.st_change_pct, f"{s.st_change_pct:+.2f}%", " pos"),
            num_cell(s.sma3_change_pct, f"{s.sma3_change_pct:+.2f}%", " pos"),
            num_cell(s.sma9_change_pct, f"{s.sma9_change_pct:+.2f}%", " pos"),
            num_cell(s.bars_in_trend, str(s.bars_in_trend)),
            f"<td>{s.chart}</td>",
        ]
        rising_rows.append(f"<tr>{''.join(cells)}</tr>")

    flipped_rows = []
    for f in flipped:
        s = f.signal
        cells = [
            symbol_cell(s.symbol), company_cell(s.symbol, companies), *price_cells(s),
            num_cell(s.st_change_pct, f"{s.st_change_pct:+.2f}%", " pos"),
            num_cell(s.bars_in_trend, str(s.bars_in_trend)),
            f'<td data-v="{f.first_listed}">{f.first_listed}</td>',
            num_cell(f.days_listed, str(f.days_listed)),
            f'<td data-v="{int(f.in_today)}">{"Yes" if f.in_today else "No"}</td>',
            f"<td>{s.chart}</td>",
        ]
        flipped_rows.append(f"<tr>{''.join(cells)}</tr>")

    return TEMPLATE.format(
        count=len(signals),
        flipped_count=len(flipped),
        rising_count=len(rising),
        watched=watched,
        total=total,
        bullish=bullish,
        fresh=sum(s.bars_in_trend <= 3 for s in signals),
        as_of=html.escape(as_of),
        period=period,
        multiplier=json.dumps(multiplier),
        band=json.dumps(band),
        signal_rows="\n".join(signal_rows),
        flipped_rows="\n".join(flipped_rows),
        rising_rows="\n".join(rising_rows),
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supertrend over VWAP</title>
<style>
:root {{
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66;
  --line: #e4e4df; --accent: #1f7a4d; --up: #1f9d5c; --down: #d0473d;
  --fresh: #eaf6ef; --st: #2b6cb0; --band: #b7791f; --tab-active: #e9f3ed;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #121412; --panel: #1a1d1a; --text: #e8e8e3; --muted: #9a9a92;
    --line: #2c302c; --accent: #4cc38a; --up: #3fbf7a; --down: #ef6a5e;
    --fresh: #18301f; --st: #6ea8f0; --band: #e0a84a; --tab-active: #1d2e23;
    color-scheme: dark;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.page {{ max-width: 1320px; margin: 0 auto; padding: 32px 16px 48px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); margin: 0 0 24px; }}
.layout {{ display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 24px; align-items: start; }}
.tabs {{ display: flex; flex-direction: column; gap: 4px; position: sticky; top: 16px; }}
.tab {{
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
  padding: 10px 12px; border: 1px solid transparent; border-radius: 8px; background: none;
  color: var(--muted); font: inherit; font-weight: 600; text-align: left; cursor: pointer;
}}
.tab:hover {{ color: var(--text); background: var(--panel); }}
.tab[aria-selected=true] {{ color: var(--text); background: var(--tab-active); border-color: var(--line); }}
.tab .badge {{
  font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted);
  background: var(--bg); border: 1px solid var(--line); border-radius: 999px; padding: 0 8px;
}}
@media (max-width: 760px) {{
  .layout {{ grid-template-columns: minmax(0, 1fr); gap: 16px; }}
  .tabs {{ flex-direction: row; flex-wrap: wrap; position: static; }}
  .tab {{ flex: 1 1 160px; }}
}}
.desc {{ color: var(--muted); margin: 0 0 16px; }}
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
  color: var(--muted); user-select: none;
}}
th[data-type] {{ cursor: pointer; }}
th[aria-sort=ascending]::after {{ content: " \\25B2"; }}
th[aria-sort=descending]::after {{ content: " \\25BC"; }}
td.company {{ max-width: 240px; overflow: hidden; text-overflow: ellipsis; color: var(--muted); }}
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
.spark .band {{ fill: none; stroke: var(--band); stroke-width: 1.5; }}
.legend {{ display: inline-flex; gap: 14px; color: var(--muted); font-size: 12px; margin-left: auto; }}
.legend i {{ display: inline-block; width: 14px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 4px; }}
.empty {{ padding: 32px; text-align: center; color: var(--muted); }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 16px; }}
</style>
</head>
<body>
<div class="page">
  <h1>Supertrend over VWAP</h1>
  <p class="sub">S&amp;P 500 &middot; daily Heikin Ashi candles &middot; Supertrend ({period}, {multiplier}) &middot; yearly VWAP of (H+L+C)/3 with &plusmn;{band}&sigma; band &middot; data as of {as_of}</p>

  <div class="layout">
    <nav class="tabs" role="tablist" aria-orientation="vertical" aria-label="Views">
      <button class="tab" role="tab" id="tab-flipped" aria-controls="flipped" aria-selected="true">Flipped since listed <span class="badge">{flipped_count}</span></button>
      <button class="tab" role="tab" id="tab-rising" aria-controls="rising" aria-selected="false" tabindex="-1">Over VWAP band &ndash; Increasing <span class="badge">{rising_count}</span></button>
      <button class="tab" role="tab" id="tab-signals" aria-controls="signals" aria-selected="false" tabindex="-1">Over VWAP band <span class="badge">{count}</span></button>
    </nav>

    <div>
      <section class="panel" role="tabpanel" id="flipped" aria-labelledby="tab-flipped">
        <p class="desc">Symbols from the saved daily lists (signals.json) whose Supertrend turned bearish after they were first listed, has since flipped back to bullish, and rose on today's candle. They don't need to be above the VWAP band today.</p>
        <div class="stats">
          <div class="stat"><b>{flipped_count}</b><span>flipped since listed, Supertrend rising</span></div>
          <div class="stat"><b>{watched}</b><span>symbols ever listed in signals.json</span></div>
        </div>
        <div class="tools">
          <input type="search" class="q" placeholder="Filter symbol or company" aria-label="Filter symbol or company">
          <span class="legend"><span><i style="color:var(--st);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP upper band</span></span>
        </div>
        <div class="table-wrap">
          <table data-sort="5">
            <thead><tr>
              <th data-type="text">Symbol</th>
              <th data-type="text">Company</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num">ST change</th>
              <th class="num" data-type="num">Bars</th>
              <th data-type="text">First listed</th>
              <th class="num" data-type="num">Days listed</th>
              <th data-type="num">In today's list</th>
              <th>Last 40 HA candles</th>
            </tr></thead>
            <tbody>
{flipped_rows}
            </tbody>
          </table>
          <div class="empty" hidden>No listed symbol has flipped bullish again with a rising Supertrend.</div>
        </div>
        <footer>"ST change" is the rise in the Supertrend line since the previous candle. "Bars" counts candles since the current bull run began, after the symbol was first listed. "Days listed" counts the days the symbol appeared in signals.json.</footer>
      </section>

      <section class="panel" role="tabpanel" id="rising" aria-labelledby="tab-rising" hidden>
        <p class="desc">Stocks from the Over VWAP band list whose Supertrend, SMA 3 and SMA 9 all rose on today's candle. The SMAs use Heikin Ashi closes.</p>
        <div class="stats">
          <div class="stat"><b>{rising_count}</b><span>over VWAP band and increasing</span></div>
          <div class="stat"><b>{count}</b><span>over VWAP band</span></div>
        </div>
        <div class="tools">
          <input type="search" class="q" placeholder="Filter symbol or company" aria-label="Filter symbol or company">
          <span class="legend"><span><i style="color:var(--st);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP upper band</span></span>
        </div>
        <div class="table-wrap">
          <table data-sort="7">
            <thead><tr>
              <th data-type="text">Symbol</th>
              <th data-type="text">Company</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num">ST change</th>
              <th class="num" data-type="num">SMA 3 change</th>
              <th class="num" data-type="num">SMA 9 change</th>
              <th class="num" data-type="num">Bars</th>
              <th>Last 40 HA candles</th>
            </tr></thead>
            <tbody>
{rising_rows}
            </tbody>
          </table>
          <div class="empty" hidden>No over-band stock has a rising Supertrend, SMA 3 and SMA 9 today.</div>
        </div>
        <footer>Changes are today's candle versus the previous one. "Bars" is the number of candles since the Supertrend turned bullish.</footer>
      </section>

      <section class="panel" role="tabpanel" id="signals" aria-labelledby="tab-signals" hidden>
        <p class="desc">Stocks whose Supertrend is bullish and above the VWAP upper band today.</p>
        <div class="stats">
          <div class="stat"><b>{count}</b><span>Supertrend above VWAP upper band</span></div>
          <div class="stat"><b>{bullish}</b><span>bullish Supertrend</span></div>
          <div class="stat"><b>{total}</b><span>stocks tracked</span></div>
          <div class="stat"><b>{fresh}</b><span>listed stocks that turned bullish in last 3 bars</span></div>
        </div>
        <div class="tools">
          <input type="search" class="q" placeholder="Filter symbol or company" aria-label="Filter symbol or company">
          <label><input type="checkbox" class="fresh-only"> New signals only</label>
          <span class="legend"><span><i style="color:var(--st);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP upper band</span></span>
        </div>
        <div class="table-wrap">
          <table data-sort="4">
            <thead><tr>
              <th data-type="text">Symbol</th>
              <th data-type="text">Company</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num">Bars</th>
              <th>Last 40 HA candles</th>
            </tr></thead>
            <tbody>
{signal_rows}
            </tbody>
          </table>
          <div class="empty" hidden>No matching stocks.</div>
        </div>
        <footer>"Bars" is the number of candles since the Supertrend turned bullish. The VWAP restarts on the first trading day of each year. Not investment advice.</footer>
      </section>
    </div>
  </div>
</div>
<script>
(() => {{
  // Tabs (vertical on wide screens). The selected tab is kept in the URL hash.
  const tabs = [...document.querySelectorAll(".tab")];
  function select(tab, focus) {{
    for (const t of tabs) {{
      const on = t === tab;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    }}
    if (focus) tab.focus();
    history.replaceState(null, "", "#" + tab.getAttribute("aria-controls"));
  }}
  tabs.forEach((tab, i) => {{
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", e => {{
      const step = {{ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1}}[e.key];
      if (step) {{ e.preventDefault(); select(tabs[(i + step + tabs.length) % tabs.length], true); }}
    }});
  }});
  const initial = tabs.find(t => "#" + t.getAttribute("aria-controls") === location.hash);
  if (initial) select(initial);

  // Sorting and filtering, per panel.
  for (const panel of document.querySelectorAll(".panel")) {{
    const table = panel.querySelector("table");
    const body = table.tBodies[0];
    const headers = [...table.tHead.rows[0].cells];
    const q = panel.querySelector(".q");
    const freshOnly = panel.querySelector(".fresh-only");
    const empty = panel.querySelector(".empty");

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
        const text = (row.cells[0].dataset.v + " " + row.cells[1].dataset.v).toUpperCase();
        const match = text.includes(term) &&
          (!freshOnly || !freshOnly.checked || row.classList.contains("fresh"));
        row.hidden = !match;
        shown += match;
      }}
      empty.hidden = shown > 0;
    }}
    q.addEventListener("input", filter);
    if (freshOnly) freshOnly.addEventListener("change", filter);

    sortBy(Number(table.dataset.sort), "ascending");
    filter();
  }}
}})();
</script>
</body>
</html>
"""


def current_results(results: list[Result]) -> tuple[list[Result], str]:
    """Drop stale symbols; return (current results, as-of date)."""
    as_of = max(r.last_date for r in results)
    cutoff = (pd.Timestamp(as_of) - pd.Timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")
    return [r for r in results if r.last_date >= cutoff], as_of


def filtered(results: list[Result]) -> list[Signal]:
    return sorted(
        (r.signal for r in results if r.above_band), key=lambda s: (s.bars_in_trend, s.symbol)
    )


def signal_records(signals: list[Signal]) -> list[dict]:
    return [
        {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in asdict(s).items()
            if k not in ("date", "chart", "st_change_pct", "sma3_change_pct", "sma9_change_pct")
        }
        for s in sorted(signals, key=lambda s: s.symbol)
    ]


def save_signals(signals: list[Signal], path: Path, as_of: str) -> dict[str, list[dict]]:
    """Add the day's filtered list to a JSON file of {date: [stocks...]}.

    Earlier days are kept; running again on the same day replaces that day's
    entry (e.g. the 5 PM run replaces the intraday 12 PM list).
    """
    return write_history(path, {as_of: signal_records(signals)})


def write_history(path: Path, days: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Merge {date: records} into the JSON file, replacing those dates."""
    history: dict[str, list[dict]] = {}
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
    history.update(days)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(history.items()))
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return ordered


def backfill(files: list[Path], days: int, args: argparse.Namespace) -> None:
    """Recompute the filtered list for each trading day in the last `days` calendar days.

    Each day uses only data up to and including that day, exactly as if the
    script had run then (except that the data reflects today's Yahoo values).
    """
    frames = {}
    for path in files:
        try:
            df = load(path)
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        frames[path.stem] = (df, df["Date"].str[:10].to_numpy())

    latest = max(dates[-1] for _, dates in frames.values() if len(dates))
    start = (pd.Timestamp(latest) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    trading_days = sorted({d for _, dates in frames.values() for d in dates if d > start})

    filled: dict[str, list[dict]] = {}
    for day in trading_days:
        results = []
        for symbol, (df, dates) in frames.items():
            end = int(np.searchsorted(dates, day, side="right"))
            if end == 0:
                continue
            result = analyze_frame(df.iloc[:end], symbol, args.period, args.multiplier, args.vwap_band)
            if result:
                results.append(result)
        current, as_of = current_results(results)
        if as_of != day:
            continue
        filled[day] = signal_records(filtered(current))
        print(f"{day}: {len(filled[day])} stocks")
    write_history(args.signals_file, filled)
    print(f"Backfilled {len(filled)} trading days ({trading_days[0]} to {trading_days[-1]}) -> {args.signals_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV directory (default: data/)")
    parser.add_argument("--output", type=Path, default=Path("index.html"), help="output file (default: index.html)")
    parser.add_argument("--period", type=int, default=10, help="Supertrend ATR period (default: 10)")
    parser.add_argument("--multiplier", type=float, default=3.0, help="Supertrend ATR multiplier (default: 3)")
    parser.add_argument(
        "--signals-file",
        type=Path,
        default=Path("data/signals.json"),
        help="JSON file that accumulates each day's filtered list (default: data/signals.json)",
    )
    parser.add_argument("--vwap-band", type=float, default=1.0, help="VWAP upper band in standard deviations (default: 1)")
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help="one-off: recompute signals.json for each trading day in the last DAYS calendar days",
    )
    args = parser.parse_args()
    if args.period < 1 or args.multiplier <= 0 or args.vwap_band < 0:
        parser.error("--period must be >= 1, --multiplier > 0 and --vwap-band >= 0")

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        parser.error(f"no CSV files found in {args.data_dir}")
    if args.backfill:
        backfill(files, args.backfill, args)

    results: list[Result] = []
    for path in files:
        try:
            result = analyze(path, args.period, args.multiplier, args.vwap_band)
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if result:
            results.append(result)

    current, as_of = current_results(results)
    bullish = sum(r.bullish for r in current)
    signals = filtered(current)

    # Save today's list first so the "flipped since listed" tab also covers today's symbols.
    history = save_signals(signals, args.signals_file, as_of)
    flipped = find_flipped(current, history, as_of)
    watched = len({stock["symbol"] for stocks in history.values() for stock in stocks})

    companies_path = args.data_dir / "companies.json"
    companies = json.loads(companies_path.read_text(encoding="utf-8")) if companies_path.exists() else {}
    rising = find_rising(signals)
    page = render(
        signals, rising, flipped, companies, watched, len(current), bullish,
        as_of, args.period, args.multiplier, args.vwap_band,
    )
    args.output.write_text(page, encoding="utf-8")
    print(
        f"{len(signals)} of {len(current)} stocks have a bullish Supertrend above the "
        f"VWAP upper band ({bullish} bullish), {len(rising)} of them increasing; {len(flipped)} of {watched} listed symbols "
        f"flipped bullish since listed with a rising Supertrend; as of {as_of} -> {args.output}, {args.signals_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
