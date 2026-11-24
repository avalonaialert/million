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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Only the most recent bars are needed. Heikin Ashi open and ATR are recursive,
# but their dependence on old bars fades quickly, so 400 bars is plenty.
LOOKBACK_BARS = 400
CHART_BARS = 40
# Bars embedded for the pop-up chart: it opens on the last CHART_BARS and zooms out to this.
DETAIL_BARS = 180
# Skip symbols whose data stops this many days before the newest file
# (e.g. companies that have left the index).
STALE_DAYS = 7
# Saved TradingView chart layout that stock links open (with the stock's symbol filled in).
TRADINGVIEW_LAYOUT = "R6fMef6E"


def tradingview_url(symbol: str) -> str:
    """TradingView chart link; TradingView writes share classes with a dot (BRK.B), Yahoo with a dash."""
    return f"https://www.tradingview.com/chart/{TRADINGVIEW_LAYOUT}/?symbol={symbol.replace('-', '.')}"
# SMA periods (on Heikin Ashi closes) checked by the "SMA Gaining" tab.
SMA_PERIODS = (3, 9, 20, 50, 100, 200)
# Daily bars used to build weekly and monthly candles (~15 years).
MTF_DAILY_BARS = 3800


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

    # Draw the Supertrend line only through the current run, so it doesn't
    # jump between the upper and lower bands.
    run_start = len(direction) - 1
    while run_start > 0 and direction[run_start - 1] == direction[-1]:
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
class TrendStep:
    """Supertrend on one timeframe, latest bar versus the previous bar."""
    state: str  # "up" rising, "flat" bullish but unchanged, "new" just flipped bullish, "bear", "na"
    change_pct: float = float("nan")

    @property
    def rising(self) -> bool:
        return self.state == "up"


def trend_step(line: np.ndarray, direction: np.ndarray) -> TrendStep:
    if len(line) < 2 or direction[-1] != 1:
        return TrendStep("bear")
    if direction[-2] != 1:
        return TrendStep("new")
    change = float((line[-1] / line[-2] - 1) * 100)
    return TrendStep("up" if change > 0 else "flat", change)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Daily OHLC rows -> weekly ("W-FRI") or monthly ("ME") bars; the last bar may be in progress."""
    frame = df.set_index(pd.to_datetime(df["Date"].str[:10]))[["Open", "High", "Low", "Close"]]
    bars = frame.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return bars.dropna()


RSI_PERIOD = 14      # Wilder RSI length
RSI_MA_PERIOD = 14   # SMA of RSI (TradingView's default RSI moving average)
RSI_LEVEL = 70


def rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder's RSI (as on TradingView): RMA of gains and losses, seeded with a simple average."""
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return out
    delta = np.diff(close)
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_gain, avg_loss = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def rsi_stats(close: np.ndarray) -> tuple[float, float]:
    """(latest RSI, latest SMA of RSI); NaN when there isn't enough history."""
    values = rsi(close)
    ma = pd.Series(values).rolling(RSI_MA_PERIOD).mean().to_numpy()
    return float(values[-1]), float(ma[-1])


def sma_stats(close: pd.Series, periods: tuple[int, ...]) -> tuple[dict[int, float], dict[int, float]]:
    """(% change of each SMA on the latest bar vs the previous bar, latest SMA value)."""
    changes, levels = {}, {}
    for period in periods:
        sma = close.rolling(period).mean().to_numpy()
        changes[period] = float((sma[-1] / sma[-2] - 1) * 100) if len(sma) >= 2 else float("nan")
        levels[period] = float(sma[-1]) if len(sma) else float("nan")
    return changes, levels


# SMA pairs checked for alignment (fast above slow) by the "SMA Gaining" tab.
SMA_PAIRS = ((3, 9), (9, 20), (20, 50), (50, 100), (100, 200))


@dataclass
class MultiTimeframe:
    symbol: str
    close: float
    change_pct: float
    daily: TrendStep
    weekly: TrendStep
    monthly: TrendStep
    chart: str
    sma_change: dict[int, float]  # % change of each SMA vs the previous day (NaN if not enough bars)
    sma_level: dict[int, float]  # latest value of each SMA
    rsi: dict[str, float]  # "d"/"w"/"m": RSI on daily/weekly/monthly Heikin Ashi close; "<tf>_ma": SMA of that RSI

    @property
    def rsi_conditions(self) -> dict[str, bool]:
        """Filter key -> met: r<tf> = RSI above 70, m<tf> = RSI above its SMA (tf = d, w or m)."""
        out = {}
        for tf in ("d", "w", "m"):
            out[f"r{tf}"] = self.rsi[tf] > RSI_LEVEL  # NaN compares False
            out[f"m{tf}"] = self.rsi[tf] > self.rsi[f"{tf}_ma"]
        return out

    def sma_gap(self, fast: int, slow: int) -> float:
        """% by which the fast SMA is above (+) or below (-) the slow SMA."""
        return (self.sma_level[fast] / self.sma_level[slow] - 1) * 100

    @property
    def sma_conditions(self) -> dict[str, bool]:
        """Filter key -> met: g<p> = SMA p gaining, a<fast>_<slow> = fast SMA above slow SMA."""
        out = {f"g{p}": g for p, g in self.sma_gaining.items()}
        out.update({f"a{f}_{s}": self.sma_gap(f, s) > 0 for f, s in SMA_PAIRS})  # NaN > 0 is False
        return out

    @property
    def sma_gaining(self) -> dict[int, bool]:
        return {p: v > 0 for p, v in self.sma_change.items()}  # NaN > 0 is False

    @property
    def steps(self) -> dict[str, TrendStep]:
        return {"d": self.daily, "w": self.weekly, "m": self.monthly}


@dataclass
class Result:
    last_date: str
    bullish: bool
    above_band: bool  # bullish and Supertrend > VWAP upper band
    signal: Signal | None  # set for every bullish stock
    mtf: MultiTimeframe | None = None  # set when analyzed with multi=True
    detail: dict | None = None  # last CHART_BARS bars for the pop-up chart (multi=True)


def rounded(values) -> list[float | None]:
    return [None if v != v else round(float(v), 2) for v in values]  # v != v is NaN


def chart_detail(
    dates: np.ndarray, ha: pd.DataFrame, line: np.ndarray, direction: np.ndarray,
    sma3: np.ndarray, sma9: np.ndarray, vwap: np.ndarray, upper: np.ndarray, close: np.ndarray,
) -> dict:
    n = DETAIL_BARS
    return {
        "d": dates[-n:].tolist(),
        "o": rounded(ha["open"].to_numpy()[-n:]),
        "h": rounded(ha["high"].to_numpy()[-n:]),
        "l": rounded(ha["low"].to_numpy()[-n:]),
        "c": rounded(ha["close"].to_numpy()[-n:]),
        # Heikin Ashi closes just before the window, so the page can compute SMAs up to 200.
        "cx": rounded(ha["close"].to_numpy()[-(n + max(SMA_PERIODS)):-n]),
        "st": rounded(line[-n:]),
        "dir": direction[-n:].tolist(),
        "s3": rounded(sma3[-n:]),
        "s9": rounded(sma9[-n:]),
        "vw": rounded(vwap[-n:]),
        "up": rounded(upper[-n:]),
        "close": round(float(close[-1]), 2),
        "chg": round(float((close[-1] / close[-2] - 1) * 100), 2),
    }


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"]).dropna()


def analyze(path: Path, period: int, multiplier: float, band: float, multi: bool = False) -> Result | None:
    return analyze_frame(load(path), path.stem, period, multiplier, band, multi)


def analyze_frame(
    df: pd.DataFrame, symbol: str, period: int, multiplier: float, band: float, multi: bool = False
) -> Result | None:
    full = df
    df = df.tail(LOOKBACK_BARS)
    if len(df) < period + 2:
        return None
    dates = df["Date"].str[:10].to_numpy()
    years = df["Date"].str[:4].to_numpy()
    ha = heikin_ashi(df)
    line, direction = supertrend(ha, period, multiplier)
    vwap, upper = yearly_vwap(ha, df["Volume"].to_numpy(dtype=float), years, band)
    last_date = dates[-1]
    close = df["Close"].to_numpy()
    sma3 = ha["close"].rolling(3).mean().to_numpy()
    sma9 = ha["close"].rolling(9).mean().to_numpy()
    mtf = None
    detail = None
    if multi:
        detail = {"sym": symbol, **chart_detail(dates, ha, line, direction, sma3, sma9, vwap, upper, close)}
        steps = {"daily": trend_step(line, direction)}
        recent = full.tail(MTF_DAILY_BARS)
        tf_close = {"weekly": np.array([]), "monthly": np.array([])}
        tf_rsi = {}  # "weekly"/"monthly" -> {period label: (RSI, SMA of RSI)} for the pop-up chart
        for name, rule in (("weekly", "W-FRI"), ("monthly", "ME")):
            bars = resample(recent, rule)
            if len(bars) < period + 2:
                steps[name] = TrendStep("na")
                continue
            ha_tf = heikin_ashi(bars)
            steps[name] = trend_step(*supertrend(ha_tf, period, multiplier))
            tf_close[name] = ha_tf["close"].to_numpy()
            values = rsi(tf_close[name])
            ma = pd.Series(values).rolling(RSI_MA_PERIOD).mean().to_numpy()
            fmt = "%Y-%m-%d" if name == "weekly" else "%Y-%m"
            keep = 45 if name == "weekly" else 14  # enough periods to cover the chart window
            tf_rsi[name] = {
                label.strftime(fmt): (values[k], ma[k])
                for k, label in list(enumerate(bars.index))[-keep:]
            }
        nan2 = (float("nan"), float("nan"))
        rsi_d, rsi_d_ma = rsi_stats(ha["close"].to_numpy())
        rsi_w, rsi_w_ma = rsi_stats(tf_close["weekly"]) if len(tf_close["weekly"]) else nan2
        rsi_m, rsi_m_ma = rsi_stats(tf_close["monthly"]) if len(tf_close["monthly"]) else nan2
        daily_rsi = rsi(ha["close"].to_numpy())
        detail["rd"] = rounded(daily_rsi[-DETAIL_BARS:])
        detail["rdm"] = rounded(pd.Series(daily_rsi).rolling(RSI_MA_PERIOD).mean().to_numpy()[-DETAIL_BARS:])
        detail["rw"] = tf_rsi.get("weekly", {})
        detail["rm"] = tf_rsi.get("monthly", {})
        mtf = MultiTimeframe(
            symbol=symbol,
            close=float(close[-1]),
            change_pct=float((close[-1] / close[-2] - 1) * 100),
            chart=mini_chart(ha, line, direction, upper, years),
            **dict(zip(("sma_change", "sma_level"), sma_stats(ha["close"], SMA_PERIODS))),
            rsi={"d": rsi_d, "d_ma": rsi_d_ma, "w": rsi_w, "w_ma": rsi_w_ma, "m": rsi_m, "m_ma": rsi_m_ma},
            **steps,
        )
    if direction[-1] != 1:
        return Result(last_date, False, False, None, mtf, detail)
    # NaN (e.g. no volume yet this year) compares False, so it never passes.
    above_band = bool(line[-1] > upper[-1])

    flips = np.flatnonzero(direction[:-1] != 1)
    flip_index = flips[-1] + 1 if len(flips) else period - 1
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
    return Result(last_date, True, above_band, signal, mtf, detail)


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


def stock_cell(symbol: str, companies: dict[str, str]) -> str:
    sym = html.escape(symbol)
    name = html.escape(companies.get(symbol, ""))
    return (
        f'<td class="stock" data-v="{sym}">'
        f'<a href="{html.escape(tradingview_url(symbol))}" target="_blank" rel="noopener">{sym}</a>'
        f'<span title="{name}">{name}</span></td>'
    )


def num_cell(value: float, text: str, extra: str = "") -> str:
    return f'<td class="num{extra}" data-v="{value}">{text}</td>'


def change_cell(value: float) -> str:
    """Daily price change as a tinted pill (sign is always shown, not color alone)."""
    tone = "up" if value >= 0 else "down"
    return f'<td class="num" data-v="{value}"><span class="pill {tone}">{value:+.2f}%</span></td>'


def delta_cell(value: float) -> str:
    return f'<td class="num delta" data-v="{value}">{pct(value)}</td>'


def bars_cell(bars: int, scale: int) -> str:
    """Trend age: the number plus a small bar scaled to the longest run on the page."""
    width = max(4, round(min(bars, scale) / scale * 100)) if scale else 0
    return (
        f'<td class="num" data-v="{bars}"><span class="age">'
        f'<span class="age-bar"><i style="width:{width}%"></i></span>{bars}</span></td>'
    )


def row(
    symbol: str, companies: dict[str, str], cells: list[str], fresh: bool = False, attrs: str = ""
) -> str:
    search = html.escape(f"{symbol} {companies.get(symbol, '')}".upper())
    cls = ' class="fresh"' if fresh else ""
    return f'<tr{cls} data-search="{search}"{attrs}>{"".join(cells)}</tr>'


TREND_LABELS = {"flat": ("Flat", -1), "new": ("New bull", -2), "bear": ("Bearish", -3), "na": ("&mdash;", -4)}


def pct(value: float) -> str:
    """Signed percent; tiny non-zero moves show as <0.01% instead of a misleading 0.00%."""
    if 0 < abs(value) < 0.005:
        return ("+" if value > 0 else "-") + "<0.01%"
    return f"{value:+.2f}%"


def rsi_cell(value: float) -> str:
    """RSI value; above 70 is highlighted and marked, not just colored."""
    if value != value:
        return '<td class="num" data-v="-1"><span class="tf na">&mdash;</span></td>'
    if value > RSI_LEVEL:
        return f'<td class="num" data-v="{value}"><span class="tf up">&#9650; {value:.1f}</span></td>'
    return f'<td class="num" data-v="{value}"><span class="tf flat">{value:.1f}</span></td>'


def rsi_ma_cell(value: float, ma: float) -> str:
    """RSI versus its SMA, in RSI points: above ✓ or below ✗."""
    if value != value or ma != ma:
        return '<td class="num" data-v="-1000"><span class="tf na">&mdash;</span></td>'
    gap = value - ma
    if gap > 0:
        return f'<td class="num" data-v="{gap}"><span class="tf up">&#10003; {ma:.1f} ({gap:+.1f})</span></td>'
    return f'<td class="num" data-v="{gap}"><span class="tf bear">&#10007; {ma:.1f} ({gap:+.1f})</span></td>'


def gap_cell(gap: float) -> str:
    """Fast SMA vs slow SMA: above ✓ (green) or below ✗ (red), with the % gap."""
    if gap != gap:
        return '<td class="num" data-v="-1000"><span class="tf na">&mdash;</span></td>'
    if gap > 0:
        return f'<td class="num" data-v="{gap}"><span class="tf up">&#10003; {pct(gap)}</span></td>'
    return f'<td class="num" data-v="{gap}"><span class="tf bear">&#10007; {pct(gap)}</span></td>'


def sma_cell(change: float) -> str:
    """SMA change vs previous day: gaining ▲ (green) or falling/flat ▼ (labelled, not just colored)."""
    if change != change:  # NaN: not enough history
        return '<td class="num" data-v="-1000"><span class="tf na">&mdash;</span></td>'
    if change > 0:
        return f'<td class="num" data-v="{change}"><span class="tf up">&#9650; {pct(change)}</span></td>'
    return f'<td class="num" data-v="{change}"><span class="tf bear">&#9660; {pct(change)}</span></td>'


def trend_cell(step: TrendStep) -> str:
    """Supertrend change on one timeframe; non-rising states are labelled, not just colored."""
    if step.rising:
        return (
            f'<td class="num" data-v="{step.change_pct}">'
            f'<span class="tf up">&#9650; {pct(step.change_pct)}</span></td>'
        )
    label, rank = TREND_LABELS[step.state]
    return f'<td class="num" data-v="{rank}"><span class="tf {step.state}">{label}</span></td>'


def updated_html() -> str:
    """'Updated <time>' in US Eastern time. Must stay on one line with class="updated"
    (the GitHub workflow skips commits where only this line changed)."""
    now = datetime.now(ZoneInfo("America/New_York"))
    label = now.strftime("%b %-d, %Y, %-I:%M %p %Z")
    return f'<div class="updated">Updated <time datetime="{now.isoformat(timespec="seconds")}">{label}</time></div>'


def period_labels(dates: list[str]) -> tuple[list[str], list[int], list[str], list[int]]:
    """Week labels (the Friday ending each week, as resample("W-FRI") names them) and month
    labels ("YYYY-MM") covering the dates, plus each date's index into those lists."""
    friday = pd.offsets.Week(weekday=4)
    week_of = [friday.rollforward(pd.Timestamp(d)).strftime("%Y-%m-%d") for d in dates]
    month_of = [d[:7] for d in dates]
    weeks, months = sorted(set(week_of)), sorted(set(month_of))
    return weeks, [weeks.index(w) for w in week_of], months, [months.index(m) for m in month_of]


def pack_details(details: dict[str, dict], companies: dict[str, str]) -> dict:
    """Compact chart data for every stock: prices as integer cents, the trend direction
    as a "+"/"-" string, and one shared date list (a stock keeps its own dates only if
    they differ from the shared ones). Decoded by getDetail() in the page."""
    dates = sorted({d for det in details.values() for d in det["d"]})[-DETAIL_BARS:]
    cents = lambda values: [None if v is None else round(v * 100) for v in values]
    tenths = lambda values: [None if v is None or v != v else round(v * 10) for v in values]
    weeks, week_idx, months, month_idx = period_labels(dates)
    stocks = {}
    for sym, det in sorted(details.items()):
        packed = {key: cents(det[key]) for key in ("o", "h", "l", "c", "cx", "st", "s3", "s9", "vw", "up")}
        packed["dir"] = "".join("+" if v == 1 else "-" for v in det["dir"])
        # RSI as integer tenths: daily per candle; weekly/monthly per period (shared labels).
        packed["rd"], packed["rdm"] = tenths(det.get("rd", [])), tenths(det.get("rdm", []))
        own = det["d"] if det["d"] != dates[-len(det["d"]):] else None
        w_labels, m_labels = (period_labels(own)[0], period_labels(own)[2]) if own else (weeks, months)
        rw, rm = det.get("rw", {}), det.get("rm", {})
        packed["rw"] = tenths([rw.get(w, (None, None))[0] for w in w_labels])
        packed["rwm"] = tenths([rw.get(w, (None, None))[1] for w in w_labels])
        packed["rm"] = tenths([rm.get(m, (None, None))[0] for m in m_labels])
        packed["rmm"] = tenths([rm.get(m, (None, None))[1] for m in m_labels])
        if own:
            _, packed["wk"], _, packed["mo"] = period_labels(own)
        packed["close"] = round(det["close"] * 100)
        packed["chg"] = round(det["chg"] * 100)
        packed["n"] = companies.get(sym, "")
        if det["d"] != dates[-len(det["d"]):]:
            packed["d"] = det["d"]
        stocks[sym] = packed
    return {"dates": dates, "wk": week_idx, "mo": month_idx, "stocks": stocks}


def render(
    signals: list[Signal], rising: list[Signal], flipped: list[Flipped], mtfs: list[MultiTimeframe],
    smas: list[MultiTimeframe], rsis: list[MultiTimeframe],
    details: dict[str, dict], companies: dict[str, str],
    history: dict[str, list[dict]], watched: int, total: int, bullish: int,
    as_of: str, period: int, multiplier: float, band: float,
) -> str:
    everyone = signals + rising + [f.signal for f in flipped]
    # Runs longer than ~3 months show as a full bar.
    scale = min(max((s.bars_in_trend for s in everyone), default=1), 60)

    flipped_rows = [
        row(f.signal.symbol, companies, [
            stock_cell(f.signal.symbol, companies),
            num_cell(f.signal.close, f"{f.signal.close:,.2f}"),
            change_cell(f.signal.change_pct),
            delta_cell(f.signal.st_change_pct),
            bars_cell(f.signal.bars_in_trend, scale),
            f'<td data-v="{f.first_listed}" class="date">{f.first_listed}</td>',
            num_cell(f.days_listed, str(f.days_listed)),
            f'<td data-v="{int(f.in_today)}">'
            + ('<span class="tag">Listed</span>' if f.in_today else '<span class="dim">&mdash;</span>')
            + "</td>",
            f'<td class="spark-cell">{f.signal.chart}</td>',
        ])
        for f in flipped
    ]
    rising_rows = [
        row(s.symbol, companies, [
            stock_cell(s.symbol, companies),
            num_cell(s.close, f"{s.close:,.2f}"),
            change_cell(s.change_pct),
            delta_cell(s.st_change_pct),
            delta_cell(s.sma3_change_pct),
            delta_cell(s.sma9_change_pct),
            bars_cell(s.bars_in_trend, scale),
            f'<td class="spark-cell">{s.chart}</td>',
        ])
        for s in rising
    ]
    signal_rows = [
        row(s.symbol, companies, [
            stock_cell(s.symbol, companies),
            num_cell(s.close, f"{s.close:,.2f}"),
            change_cell(s.change_pct),
            bars_cell(s.bars_in_trend, scale),
            f'<td class="spark-cell">{s.chart}</td>',
        ], fresh=s.bars_in_trend <= 3)
        for s in signals
    ]

    mtf_rows = [
        row(m.symbol, companies, [
            stock_cell(m.symbol, companies),
            num_cell(m.close, f"{m.close:,.2f}"),
            change_cell(m.change_pct),
            trend_cell(m.daily),
            trend_cell(m.weekly),
            trend_cell(m.monthly),
            f'<td class="spark-cell">{m.chart}</td>',
        ], attrs="".join(f' data-{k}="{int(v.rising)}"' for k, v in m.steps.items()))
        for m in mtfs
    ]
    mtf_all = sum(all(v.rising for v in m.steps.values()) for m in mtfs)

    sma_rows = [
        row(m.symbol, companies, [
            stock_cell(m.symbol, companies),
            num_cell(m.close, f"{m.close:,.2f}"),
            change_cell(m.change_pct),
            *(sma_cell(m.sma_change[p]) for p in SMA_PERIODS),
            *(gap_cell(m.sma_gap(f, s)) for f, s in SMA_PAIRS),
        ], attrs="".join(f' data-{k}="{int(v)}"' for k, v in m.sma_conditions.items()))
        for m in smas
    ]
    sma_all = sum(all(m.sma_conditions.values()) for m in smas)

    rsi_rows = [
        row(m.symbol, companies, [
            stock_cell(m.symbol, companies),
            num_cell(m.close, f"{m.close:,.2f}"),
            change_cell(m.change_pct),
            rsi_cell(m.rsi["d"]),
            rsi_ma_cell(m.rsi["d"], m.rsi["d_ma"]),
            rsi_cell(m.rsi["w"]),
            rsi_ma_cell(m.rsi["w"], m.rsi["w_ma"]),
            rsi_cell(m.rsi["m"]),
            rsi_ma_cell(m.rsi["m"], m.rsi["m_ma"]),
        ], attrs="".join(f' data-{k}="{int(v)}"' for k, v in m.rsi_conditions.items()))
        for m in rsis
    ]
    rsi_all = sum(all(m.rsi_conditions.values()) for m in rsis)

    series = [{"d": day, "n": len(stocks)} for day, stocks in sorted(history.items())]
    prev = series[-2]["n"] if len(series) > 1 else None
    delta = len(signals) - prev if prev is not None else 0
    if prev is None:
        delta_text = ""
    elif delta:
        arrow, tone = ("&#9650;", "up") if delta > 0 else ("&#9660;", "down")
        delta_text = f'<span class="kpi-delta {tone}">{arrow} {abs(delta)} vs prior day</span>'
    else:
        delta_text = '<span class="kpi-delta flat">No change vs prior day</span>'
    as_of_label = pd.Timestamp(as_of).strftime("%a, %b %-d %Y")

    values = {
        "AS_OF": html.escape(as_of_label),
        "UPDATED": updated_html(),
        "COUNT": str(len(signals)),
        "DELTA": delta_text,
        "RISING": str(len(rising)),
        "FLIPPED": str(len(flipped)),
        "WATCHED": str(watched),
        "FRESH": str(sum(s.bars_in_trend <= 3 for s in signals)),
        "FLIPPED_ROWS": "\n".join(flipped_rows),
        "RISING_ROWS": "\n".join(rising_rows),
        "SIGNAL_ROWS": "\n".join(signal_rows),
        "MTF_ROWS": "\n".join(mtf_rows),
        "MTF_ALL": str(mtf_all),
        "TV_LAYOUT": TRADINGVIEW_LAYOUT,
        "SMA_ROWS": "\n".join(sma_rows),
        "SMA_ALL": str(sma_all),
        "RSI_ROWS": "\n".join(rsi_rows),
        "RSI_ALL": str(rsi_all),
        # Embedded as JSON; escape "</" so it can't close the script tag.
        "DETAILS": json.dumps(pack_details(details, companies), separators=(",", ":")).replace("</", "<\\/"),
        "SYMBOL_OPTIONS": "".join(
            f'<option value="{html.escape(sym)}">{html.escape(companies.get(sym, ""))}</option>'
            for sym in sorted(details)
        ),
    }
    page = TEMPLATE
    for key, value in values.items():
        page = page.replace(f"%%{key}%%", value)
    return page


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supertrend Monitor</title>
<style>
:root {
  --page: #f4f4f1; --surface: #fcfcfb; --surface-2: #f0efec; --ring: rgba(11,11,11,0.10);
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --series: #2a78d6; --series-fill: rgba(42,120,214,0.12); --band: #eb6834;
  --up: #0a7d32; --up-bg: rgba(12,163,12,0.12); --down: #c23434; --down-bg: rgba(208,59,59,0.12);
  --candle-up: #1a9e4b; --candle-down: #d94545; --accent: #2a78d6; --fresh: rgba(42,120,214,0.06);
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #0d0d0d; --surface: #1a1a19; --surface-2: #232322; --ring: rgba(255,255,255,0.10);
    --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --series: #3987e5; --series-fill: rgba(57,135,229,0.18); --band: #d95926;
    --up: #34c46a; --up-bg: rgba(12,163,12,0.16); --down: #ef6a6a; --down-bg: rgba(230,103,103,0.16);
    --candle-up: #2fb862; --candle-down: #e66767; --accent: #3987e5; --fresh: rgba(57,135,229,0.08);
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --surface-2: #232322; --ring: rgba(255,255,255,0.10);
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --series: #3987e5; --series-fill: rgba(57,135,229,0.18); --band: #d95926;
  --up: #34c46a; --up-bg: rgba(12,163,12,0.16); --down: #ef6a6a; --down-bg: rgba(230,103,103,0.16);
  --candle-up: #2fb862; --candle-down: #e66767; --accent: #3987e5; --fresh: rgba(57,135,229,0.08);
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 16px 40px; }

/* Top bar */
.topbar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px;
  padding: 20px 0 18px; border-bottom: 1px solid var(--grid); margin-bottom: 20px;
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 17px; letter-spacing: -0.01em; }
.brand svg { width: 22px; height: 22px; color: var(--accent); }
.asof { color: var(--ink-2); font-size: 13px; display: flex; align-items: center; gap: 6px; }
.asof::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--up); }
.updated { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.lookup { margin-left: auto; position: relative; display: flex; align-items: center; }
.lookup input {
  width: 280px; max-width: 100%; padding: 8px 34px 8px 32px; border-radius: 9px; border: 1px solid var(--ring);
  background: var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='none' stroke='%23898781' stroke-width='2'%3E%3Ccircle cx='6' cy='6' r='4.5'/%3E%3Cpath d='m9.5 9.5 3 3'/%3E%3C/svg%3E") no-repeat 11px center;
  color: var(--ink); font: inherit; font-size: 13px;
}
.lookup input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--fresh); }
.lookup kbd {
  position: absolute; right: 9px; font: 11px/1 system-ui, sans-serif; color: var(--muted);
  border: 1px solid var(--ring); border-radius: 4px; padding: 2px 5px; pointer-events: none;
}
.lookup-msg {
  position: absolute; top: calc(100% + 6px); right: 0; z-index: 5; white-space: nowrap; font-size: 12px;
  color: var(--down); background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 6px 10px;
}

/* KPI tiles */
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 20px; }
.kpi {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 14px 16px; text-align: left; font: inherit; color: inherit; min-width: 0;
}
button.kpi { cursor: pointer; transition: border-color .15s, transform .15s; }
button.kpi:hover { border-color: var(--accent); }
button.kpi[aria-pressed="true"] { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.kpi-label { color: var(--ink-2); font-size: 12px; font-weight: 600; }
.kpi-value { font-size: 30px; font-weight: 700; letter-spacing: -0.02em; margin: 4px 0 2px; }
.kpi-value small { font-size: 14px; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.kpi-foot { color: var(--muted); font-size: 12px; min-height: 16px; }
.kpi-delta.up { color: var(--up); } .kpi-delta.down { color: var(--down); } .kpi-delta.flat { color: var(--muted); }

/* History chart */
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; }

/* Tabs + table */
.main { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 20px; align-items: start; }
.tabs { display: flex; flex-direction: column; gap: 4px; position: sticky; top: 16px; }
.tab {
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
  padding: 10px 12px; border-radius: 8px; border: 1px solid transparent; background: none;
  color: var(--ink-2); font: inherit; font-weight: 600; text-align: left; cursor: pointer;
}
.tab:hover { background: var(--surface); color: var(--ink); }
.tab[aria-selected="true"] { background: var(--surface); color: var(--ink); border-color: var(--ring); box-shadow: inset 3px 0 0 var(--accent); }
.tab .count {
  font-size: 12px; font-variant-numeric: tabular-nums; color: var(--ink-2);
  background: var(--surface-2); border-radius: 999px; padding: 1px 8px;
}
.tab[aria-selected="true"] .count { background: var(--accent); color: #fff; }

.panel-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 12px 16px; border-bottom: 1px solid var(--grid); }
.panel-head h2 { font-size: 15px; margin: 0; }
.legend { display: flex; gap: 14px; color: var(--muted); font-size: 12px; margin-left: auto; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 0; border-top: 2px solid; }
.search {
  width: 220px; max-width: 100%; padding: 7px 10px 7px 30px; border-radius: 8px; border: 1px solid var(--ring);
  background: var(--page) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='none' stroke='%23898781' stroke-width='2'%3E%3Ccircle cx='6' cy='6' r='4.5'/%3E%3Cpath d='m9.5 9.5 3 3'/%3E%3C/svg%3E") no-repeat 10px center;
  color: var(--ink); font: inherit; font-size: 13px;
}
.toggle { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 12px; cursor: pointer; }

.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 640px; }
th, td { padding: 9px 14px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--grid); }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--surface-2); }
th {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); user-select: none; position: sticky; top: 0; background: var(--surface);
}
th[data-type] { cursor: pointer; }
th[data-type]:hover { color: var(--ink); }
th[aria-sort="ascending"]::after { content: " \\25B2"; font-size: 9px; }
th[aria-sort="descending"]::after { content: " \\25BC"; font-size: 9px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.stock a { color: var(--ink); font-weight: 700; text-decoration: none; display: block; }
.stock a:hover { color: var(--accent); }
.stock span { display: block; color: var(--muted); font-size: 12px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
tr.fresh td { background: var(--fresh); }
tr.fresh .stock a::after {
  content: "NEW"; margin-left: 6px; font-size: 9px; font-weight: 700; letter-spacing: .05em;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 4px; padding: 0 4px; vertical-align: 2px;
}
.pill { display: inline-block; min-width: 64px; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; }
.pill.up { color: var(--up); background: var(--up-bg); }
.pill.down { color: var(--down); background: var(--down-bg); }
.delta { color: var(--up); font-weight: 600; }
.age { display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end; }
.age-bar { width: 56px; height: 4px; border-radius: 2px; background: var(--surface-2); overflow: hidden; }
.age-bar i { display: block; height: 100%; background: var(--accent); border-radius: 2px; }
.date { color: var(--ink-2); font-variant-numeric: tabular-nums; }
.tag { font-size: 11px; font-weight: 600; color: var(--accent); background: var(--fresh); border: 1px solid var(--accent); border-radius: 4px; padding: 1px 6px; }
.dim { color: var(--muted); }
.spark-cell { padding-top: 4px; padding-bottom: 4px; }
.spark { width: 168px; height: 42px; display: block; }
.spark .up { fill: var(--candle-up); stroke: var(--candle-up); stroke-width: 1; }
.spark .down { fill: var(--candle-down); stroke: var(--candle-down); stroke-width: 1; }
.spark .st { fill: none; stroke: var(--series); stroke-width: 1.5; stroke-dasharray: 3 2; }
.spark .band { fill: none; stroke: var(--band); stroke-width: 1.5; }
.tf { display: inline-block; min-width: 84px; padding: 2px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.tf.up { color: var(--up); background: var(--up-bg); }
.tf.flat, .tf.new, .tf.na { color: var(--ink-2); background: var(--surface-2); }
.tf.bear { color: var(--down); background: var(--down-bg); }
.tf-filter { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; border: 0; padding: 0; margin: 0; }
.tf-filter legend { float: left; padding: 0; margin-right: 4px; color: var(--muted); font-size: 12px; }
.tf-filter label { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; color: var(--ink-2); cursor: pointer; }
.empty { padding: 40px 16px; text-align: center; color: var(--muted); }

/* Stock detail pop-up */
.stock a, .spark-cell { cursor: pointer; }
.spark-cell:hover .spark { opacity: .8; }
dialog.detail {
  width: min(1000px, calc(100vw - 32px)); max-width: none; padding: 0; border: 1px solid var(--ring);
  border-radius: 14px; background: var(--surface); color: var(--ink); box-shadow: 0 24px 64px rgba(0,0,0,.35);
}
dialog.detail::backdrop { background: rgba(0,0,0,.55); }
.dlg-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 16px 20px 12px; border-bottom: 1px solid var(--grid); }
.dlg-title { display: flex; flex-direction: column; min-width: 0; }
.dlg-title h3 { margin: 0; font-size: 20px; letter-spacing: -0.01em; }
.dlg-title span { color: var(--muted); font-size: 13px; }
.dlg-price { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.dlg-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.dlg-link { color: var(--accent); font-size: 13px; font-weight: 600; text-decoration: none; padding: 6px 10px; border-radius: 8px; }
.dlg-link:hover { background: var(--surface-2); }
.dlg-close {
  width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--ring); background: var(--surface);
  color: var(--ink-2); font-size: 20px; line-height: 1; cursor: pointer;
}
.dlg-close:hover { background: var(--surface-2); color: var(--ink); }
.dlg-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 12px 20px 0; color: var(--ink-2); font-size: 12px; }
.dlg-legend span { display: inline-flex; align-items: center; gap: 6px; }
.dlg-legend svg { width: 26px; height: 8px; overflow: visible; }
.dlg-legend .sep { width: 1px; height: 14px; background: var(--grid); }
.zoom { margin-left: auto; display: inline-flex; border: 1px solid var(--ring); border-radius: 8px; overflow: hidden; }
.zoom button {
  border: 0; background: none; color: var(--ink-2); font: inherit; font-size: 12px; font-weight: 600;
  padding: 4px 10px; cursor: pointer; font-variant-numeric: tabular-nums;
}
.zoom button + button { border-left: 1px solid var(--ring); }
.zoom button:hover { background: var(--surface-2); color: var(--ink); }
.zoom button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.dlg-chart { position: relative; height: 400px; padding: 4px 8px 4px; }
.dlg-scroll { overflow-x: auto; overflow-y: hidden; margin: 0 20px 14px 64px; height: 16px; scrollbar-color: var(--axis) transparent; }
.dlg-scroll > div { height: 1px; }
.dlg-scroll[hidden] { display: none; }
.dlg-chart svg { display: block; width: 100%; height: 100%; }
.dlg-chart svg:focus { outline: none; }
.dlg-chart svg:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: 6px; }
.dc .grid { stroke: var(--grid); stroke-width: 1; }
.dc .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.dc .c-up { fill: var(--candle-up); stroke: var(--candle-up); }
.dc .c-down { fill: var(--candle-down); stroke: var(--candle-down); }
.dc .candles { opacity: .55; }
.dc .band-fill { fill: var(--band); opacity: .08; }
.dc .band { fill: none; stroke: var(--band); stroke-width: 1.5; }
.dc .vwap { fill: none; stroke: var(--band); stroke-width: 1; stroke-dasharray: 2 3; }
.dc .seg { fill: none; stroke-linecap: round; stroke-linejoin: round; }
.dc .rise { stroke: var(--up); }
.dc .flat { stroke: var(--muted); }
.dc .w-st { stroke-width: 2.75; }
.dc .w-s3 { stroke-width: 1.75; stroke-dasharray: 6 4; }
.dc .w-s9 { stroke-width: 1.75; stroke-dasharray: 1 4; }
.dc .w-s20 { stroke-width: 1.5; stroke-dasharray: 10 3 2 3; }
.dc .w-s50 { stroke-width: 1.5; stroke-dasharray: 14 5; }
.dc .w-s100 { stroke-width: 1.25; }
.dc .w-s200 { stroke-width: 3.5; opacity: .45; }
.sma-toggle { display: inline-flex; align-items: center; gap: 6px; color: var(--ink); font-weight: 600; cursor: pointer; }
.long-sma[hidden], .dlg-legend .ind[hidden], .rsi-tf[hidden] { display: none; }
.zoom.rsi-tf { margin-left: 0; }
.dlg-chart.with-rsi { height: 580px; }
.dc .pane-title { font-size: 11px; font-weight: 600; fill: var(--ink-2); }
.dc .rsi-zone { fill: var(--ink); opacity: .035; }
.dc .rsi-level { stroke: var(--axis); stroke-width: 1; stroke-dasharray: 4 3; }
.dc .w-rsi { stroke-width: 2; }
.dc .w-rsima { stroke-width: 1.5; stroke-dasharray: 5 3; }
@media (max-width: 700px) { .dlg-chart.with-rsi { height: 480px; } }
.dc .end { font-size: 11px; font-weight: 600; fill: var(--ink-2); }
.mk-up { fill: var(--up); stroke: var(--surface); stroke-width: 2; }
.mk-down { fill: var(--down); stroke: var(--surface); stroke-width: 2; }
.dc .mk-halo { fill: none; stroke-width: 1.5; opacity: .45; }
.dc .mk-halo.up { stroke: var(--up); } .dc .mk-halo.down { stroke: var(--down); }
.dc .mk-label { font-size: 10px; font-weight: 700; }
.dc .mk-label.up { fill: var(--up); } .dc .mk-label.down { fill: var(--down); }
.dc .cross { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.dc .hl { fill: var(--ink); opacity: .05; }
.dlg-readout {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 13px; margin: 10px 20px 0; padding: 8px 12px;
  background: var(--surface-2); border-radius: 8px; font-size: 12px;
  min-height: 54px; align-content: center;  /* fixed height so hovering never shifts the chart */
}
.dlg-readout .r-date { font-weight: 700; color: var(--ink); min-width: 92px; }
.dlg-readout .r-item { display: inline-flex; align-items: baseline; gap: 5px; white-space: nowrap; }
.dlg-readout .r-item span { color: var(--muted); }
.dlg-readout .r-item b { font-variant-numeric: tabular-nums; color: var(--ink); }
.dlg-readout .r-item b.rise { color: var(--up); }
.dlg-readout .r-cross { font-weight: 600; }
.dlg-readout .r-cross.up { color: var(--up); } .dlg-readout .r-cross.down { color: var(--down); }
@media (max-width: 700px) { .dlg-chart { height: 320px; } .dlg-actions { margin-left: 0; } }

@media (max-width: 900px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .main { grid-template-columns: minmax(0, 1fr); }
  .tabs { flex-direction: row; overflow-x: auto; position: static; }
  .tab { flex: 1 0 auto; }
  .tab[aria-selected="true"] { box-shadow: inset 0 -3px 0 var(--accent); }
  .lookup { margin-left: 0; order: 3; width: 100%; }
  .lookup input { width: 100%; }
  .legend { margin-left: 0; }
}
</style>
</head>
<body>
<div class="wrap">
  <header class="topbar">
    <div class="brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/></svg>
      Supertrend Monitor
    </div>
    <div class="asof">%%AS_OF%%</div>
    <form class="lookup" id="lookup" role="search" autocomplete="off">
      <input id="lookup-input" list="symbols" placeholder="Chart any symbol or company" aria-label="Chart any symbol or company" spellcheck="false">
      <kbd>/</kbd>
      <span class="lookup-msg" id="lookup-msg" role="status" hidden></span>
    </form>
    <datalist id="symbols">%%SYMBOL_OPTIONS%%</datalist>
    %%UPDATED%%
  </header>

  <section class="kpis" aria-label="Summary">
    <button class="kpi" data-tab="flipped">
      <div class="kpi-label">Bullish Re-entry</div>
      <div class="kpi-value">%%FLIPPED%%</div>
      <div class="kpi-foot">of %%WATCHED%% symbols ever listed</div>
    </button>
    <button class="kpi" data-tab="rising">
      <div class="kpi-label">Above VWAP &middot; Rising</div>
      <div class="kpi-value">%%RISING%%</div>
      <div class="kpi-foot">ST, SMA 3 and SMA 9 rising</div>
    </button>
    <button class="kpi" data-tab="mtf">
      <div class="kpi-label">Multi-Timeframe Rising</div>
      <div class="kpi-value">%%MTF_ALL%%</div>
      <div class="kpi-foot">Daily, weekly &amp; monthly Supertrend rising</div>
    </button>
    <button class="kpi" data-tab="sma">
      <div class="kpi-label">SMA Gaining</div>
      <div class="kpi-value">%%SMA_ALL%%</div>
      <div class="kpi-foot">All gaining &amp; stacked 3 &gt; &hellip; &gt; 200</div>
    </button>
    <button class="kpi" data-tab="rsi">
      <div class="kpi-label">RSI 70</div>
      <div class="kpi-value">%%RSI_ALL%%</div>
      <div class="kpi-foot">D, W &amp; M RSI &gt; 70 and &gt; its SMA</div>
    </button>
    <button class="kpi" data-tab="signals">
      <div class="kpi-label">Above VWAP Band</div>
      <div class="kpi-value">%%COUNT%%</div>
      <div class="kpi-foot">%%DELTA%%</div>
    </button>
  </section>

  <div class="main">
    <nav class="tabs" role="tablist" aria-orientation="vertical" aria-label="Lists">
      <button class="tab" role="tab" id="tab-flipped" aria-controls="panel-flipped" aria-selected="true">Bullish Re-entry <span class="count">%%FLIPPED%%</span></button>
      <button class="tab" role="tab" id="tab-rising" aria-controls="panel-rising" aria-selected="false" tabindex="-1">Above VWAP &middot; Rising <span class="count">%%RISING%%</span></button>
      <button class="tab" role="tab" id="tab-mtf" aria-controls="panel-mtf" aria-selected="false" tabindex="-1">Multi-Timeframe Rising <span class="count" id="mtf-count">%%MTF_ALL%%</span></button>
      <button class="tab" role="tab" id="tab-sma" aria-controls="panel-sma" aria-selected="false" tabindex="-1">SMA Gaining <span class="count" id="sma-count">%%SMA_ALL%%</span></button>
      <button class="tab" role="tab" id="tab-rsi" aria-controls="panel-rsi" aria-selected="false" tabindex="-1">RSI 70 <span class="count" id="rsi-count">%%RSI_ALL%%</span></button>
      <button class="tab" role="tab" id="tab-signals" aria-controls="panel-signals" aria-selected="false" tabindex="-1">Above VWAP Band <span class="count">%%COUNT%%</span></button>
    </nav>

    <div>
      <section class="card panel" role="tabpanel" id="panel-flipped" aria-labelledby="tab-flipped">
        <div class="panel-head">
          <h2>Bullish Re-entry</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <div class="legend"><span><i style="color:var(--series);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP band</span></div>
        </div>
        <div class="table-wrap">
          <table data-sort="4">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="Supertrend change vs previous candle">ST &Delta;</th>
              <th class="num" data-type="num" title="Candles since the Supertrend turned bullish">Trend age</th>
              <th data-type="text">First listed</th>
              <th class="num" data-type="num" title="Days in signals.json">Days listed</th>
              <th data-type="num">Today</th>
              <th>Last 40 candles</th>
            </tr></thead>
            <tbody>
%%FLIPPED_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>

      <section class="card panel" role="tabpanel" id="panel-rising" aria-labelledby="tab-rising" hidden>
        <div class="panel-head">
          <h2>Above VWAP &middot; Rising</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <div class="legend"><span><i style="color:var(--series);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP band</span></div>
        </div>
        <div class="table-wrap">
          <table data-sort="6">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="Supertrend change vs previous candle">ST &Delta;</th>
              <th class="num" data-type="num" title="SMA 3 change vs previous candle">SMA 3 &Delta;</th>
              <th class="num" data-type="num" title="SMA 9 change vs previous candle">SMA 9 &Delta;</th>
              <th class="num" data-type="num" title="Candles since the Supertrend turned bullish">Trend age</th>
              <th>Last 40 candles</th>
            </tr></thead>
            <tbody>
%%RISING_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>


      <section class="card panel" role="tabpanel" id="panel-mtf" aria-labelledby="tab-mtf" data-count="mtf-count" hidden>
        <div class="panel-head">
          <h2>Multi-Timeframe Rising</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <fieldset class="tf-filter">
            <legend>Rising on</legend>
            <label><input type="checkbox" class="tf-check" value="d" checked> Daily</label>
            <label><input type="checkbox" class="tf-check" value="w" checked> Weekly</label>
            <label><input type="checkbox" class="tf-check" value="m" checked> Monthly</label>
          </fieldset>
          <div class="legend"><span><i style="color:var(--series);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP band</span></div>
        </div>
        <div class="table-wrap">
          <table data-sort="0">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="Daily Heikin Ashi Supertrend vs previous day">Daily ST</th>
              <th class="num" data-type="num" title="Weekly Heikin Ashi Supertrend vs previous week (current week in progress)">Weekly ST</th>
              <th class="num" data-type="num" title="Monthly Heikin Ashi Supertrend vs previous month (current month in progress)">Monthly ST</th>
              <th>Last 40 daily candles</th>
            </tr></thead>
            <tbody>
%%MTF_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>

      <section class="card panel" role="tabpanel" id="panel-sma" aria-labelledby="tab-sma" data-count="sma-count" hidden>
        <div class="panel-head">
          <h2>SMA Gaining</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <fieldset class="tf-filter">
            <legend>Gaining</legend>
            <label><input type="checkbox" class="tf-check" value="g3" checked> SMA 3</label>
            <label><input type="checkbox" class="tf-check" value="g9" checked> SMA 9</label>
            <label><input type="checkbox" class="tf-check" value="g20" checked> SMA 20</label>
            <label><input type="checkbox" class="tf-check" value="g50" checked> SMA 50</label>
            <label><input type="checkbox" class="tf-check" value="g100" checked> SMA 100</label>
            <label><input type="checkbox" class="tf-check" value="g200" checked> SMA 200</label>
          </fieldset>
          <fieldset class="tf-filter">
            <legend>Aligned</legend>
            <label><input type="checkbox" class="tf-check" value="a3_9" checked> SMA 3 &gt; SMA 9</label>
            <label><input type="checkbox" class="tf-check" value="a9_20" checked> SMA 9 &gt; SMA 20</label>
            <label><input type="checkbox" class="tf-check" value="a20_50" checked> SMA 20 &gt; SMA 50</label>
            <label><input type="checkbox" class="tf-check" value="a50_100" checked> SMA 50 &gt; SMA 100</label>
            <label><input type="checkbox" class="tf-check" value="a100_200" checked> SMA 100 &gt; SMA 200</label>
          </fieldset>
        </div>
        <div class="table-wrap">
          <table data-sort="0">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="SMA 3 of Heikin Ashi close vs previous day">SMA 3</th>
              <th class="num" data-type="num" title="SMA 9 of Heikin Ashi close vs previous day">SMA 9</th>
              <th class="num" data-type="num" title="SMA 20 of Heikin Ashi close vs previous day">SMA 20</th>
              <th class="num" data-type="num" title="SMA 50 of Heikin Ashi close vs previous day">SMA 50</th>
              <th class="num" data-type="num" title="SMA 100 of Heikin Ashi close vs previous day">SMA 100</th>
              <th class="num" data-type="num" title="SMA 200 of Heikin Ashi close vs previous day">SMA 200</th>
              <th class="num" data-type="num" title="How far SMA 3 is above (+) or below (-) SMA 9">3 &gt; 9</th>
              <th class="num" data-type="num" title="How far SMA 9 is above (+) or below (-) SMA 20">9 &gt; 20</th>
              <th class="num" data-type="num" title="How far SMA 20 is above (+) or below (-) SMA 50">20 &gt; 50</th>
              <th class="num" data-type="num" title="How far SMA 50 is above (+) or below (-) SMA 100">50 &gt; 100</th>
              <th class="num" data-type="num" title="How far SMA 100 is above (+) or below (-) SMA 200">100 &gt; 200</th>
            </tr></thead>
            <tbody>
%%SMA_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>

      <section class="card panel" role="tabpanel" id="panel-rsi" aria-labelledby="tab-rsi" data-count="rsi-count" hidden>
        <div class="panel-head">
          <h2>RSI 70</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <fieldset class="tf-filter">
            <legend>RSI &gt; 70</legend>
            <label><input type="checkbox" class="tf-check" value="rd" checked> Daily</label>
            <label><input type="checkbox" class="tf-check" value="rw" checked> Weekly</label>
            <label><input type="checkbox" class="tf-check" value="rm" checked> Monthly</label>
          </fieldset>
          <fieldset class="tf-filter">
            <legend>RSI &gt; SMA of RSI</legend>
            <label><input type="checkbox" class="tf-check" value="md" checked> Daily</label>
            <label><input type="checkbox" class="tf-check" value="mw" checked> Weekly</label>
            <label><input type="checkbox" class="tf-check" value="mm" checked> Monthly</label>
          </fieldset>
        </div>
        <div class="table-wrap">
          <table data-sort="0">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="RSI 14 of daily Heikin Ashi close">Daily RSI</th>
              <th class="num" data-type="num" title="SMA 14 of daily RSI, and RSI minus that SMA">Daily RSI SMA</th>
              <th class="num" data-type="num" title="RSI 14 of weekly Heikin Ashi close (current week in progress)">Weekly RSI</th>
              <th class="num" data-type="num" title="SMA 14 of weekly RSI, and RSI minus that SMA">Weekly RSI SMA</th>
              <th class="num" data-type="num" title="RSI 14 of monthly Heikin Ashi close (current month in progress)">Monthly RSI</th>
              <th class="num" data-type="num" title="SMA 14 of monthly RSI, and RSI minus that SMA">Monthly RSI SMA</th>
            </tr></thead>
            <tbody>
%%RSI_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>

      <section class="card panel" role="tabpanel" id="panel-signals" aria-labelledby="tab-signals" hidden>
        <div class="panel-head">
          <h2>Above VWAP Band</h2>
          <input type="search" class="search" placeholder="Search" aria-label="Search symbol or company">
          <label class="toggle"><input type="checkbox" class="fresh-only"> New (%%FRESH%%)</label>
          <div class="legend"><span><i style="color:var(--series);border-top-style:dashed"></i>Supertrend</span><span><i style="color:var(--band)"></i>VWAP band</span></div>
        </div>
        <div class="table-wrap">
          <table data-sort="3">
            <thead><tr>
              <th data-type="text">Stock</th>
              <th class="num" data-type="num">Close</th>
              <th class="num" data-type="num">Day</th>
              <th class="num" data-type="num" title="Candles since the Supertrend turned bullish">Trend age</th>
              <th>Last 40 candles</th>
            </tr></thead>
            <tbody>
%%SIGNAL_ROWS%%
            </tbody>
          </table>
          <div class="empty" hidden>No stocks</div>
        </div>
      </section>
    </div>
  </div>
</div>


<dialog class="detail" id="detail" aria-labelledby="detail-symbol">
  <div class="dlg-head">
    <div class="dlg-title"><h3 id="detail-symbol"></h3><span id="detail-name"></span></div>
    <div class="dlg-price"><span id="detail-close"></span><span class="pill" id="detail-chg"></span></div>
    <div class="dlg-actions">
      <a class="dlg-link" id="detail-link" target="_blank" rel="noopener">TradingView &#8599;</a>
      <button class="dlg-close" id="detail-close-btn" aria-label="Close" autofocus>&times;</button>
    </div>
  </div>
  <div class="dlg-legend" aria-label="Legend">
    <span class="ind"><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="2.75" stroke-linecap="round"/></svg>Supertrend</span>
    <span><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.75" stroke-dasharray="6 4" stroke-linecap="round"/></svg>SMA 3</span>
    <span><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.75" stroke-dasharray="1 4" stroke-linecap="round"/></svg>SMA 9</span>
    <span class="long-sma" hidden><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="10 3 2 3"/></svg>SMA 20</span>
    <span class="long-sma" hidden><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="14 5"/></svg>SMA 50</span>
    <span class="long-sma" hidden><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.25"/></svg>SMA 100</span>
    <span class="long-sma" hidden><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="3.5" opacity=".45"/></svg>SMA 200</span>
    <span class="ind"><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--band)" stroke-width="1.5"/></svg>VWAP band</span>
    <i class="sep"></i>
    <span><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--up)" stroke-width="2.75" stroke-linecap="round"/></svg>Rising</span>
    <span><svg viewBox="0 0 26 8"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--muted)" stroke-width="2.75" stroke-linecap="round"/></svg>Not rising</span>
    <i class="sep ind"></i>
    <span class="ind"><svg viewBox="0 0 10 8" style="width:10px"><circle cx="5" cy="4" r="4.5" class="mk-up"/></svg>Cross above</span>
    <span class="ind"><svg viewBox="0 0 10 8" style="width:10px"><circle cx="5" cy="4" r="4.5" class="mk-down"/></svg>Cross below</span>
    <label class="sma-toggle"><input type="checkbox" id="show-smas"> All SMAs</label>
    <label class="sma-toggle"><input type="checkbox" id="show-rsi"> RSI</label>
    <div class="zoom rsi-tf" role="group" aria-label="RSI timeframe" hidden>
      <button type="button" data-tf="D" aria-pressed="true">D</button><button type="button" data-tf="W">W</button><button type="button" data-tf="M">M</button>
    </div>
    <div class="zoom" role="group" aria-label="Range">
      <button type="button" data-span="40">40D</button><button type="button" data-span="90">90D</button><button type="button" data-span="180">180D</button>
    </div>
  </div>
  <div class="dlg-readout" id="detail-readout" aria-live="polite"></div>
  <div class="dlg-chart" id="detail-chart"></div>
  <div class="dlg-scroll" id="detail-scroll" aria-label="Scroll through dates"><div></div></div>
</dialog>
<script type="application/json" id="details-data">%%DETAILS%%</script>
<script>
(() => {
  /* ---------- Tabs ---------- */
  const tabs = [...document.querySelectorAll(".tab")];
  const kpis = [...document.querySelectorAll("button.kpi")];
  function select(id, focus) {
    for (const t of tabs) {
      const on = t.getAttribute("aria-controls") === "panel-" + id;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      if (on && focus) t.focus();
    }
    for (const k of kpis) k.setAttribute("aria-pressed", k.dataset.tab === id);
    history.replaceState(null, "", "#" + id);
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab.id.slice(4)));
    tab.addEventListener("keydown", e => {
      const step = {ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1}[e.key];
      if (!step) return;
      e.preventDefault();
      select(tabs[(i + step + tabs.length) % tabs.length].id.slice(4), true);
    });
  });
  kpis.forEach(k => k.addEventListener("click", () => {
    select(k.dataset.tab);
    document.querySelector(".main").scrollIntoView({behavior: "smooth", block: "start"});
  }));
  const fromHash = tabs.find(t => "#" + t.id.slice(4) === location.hash);
  select(fromHash ? fromHash.id.slice(4) : "flipped");

  /* ---------- Tables: sort + search ---------- */
  for (const panel of document.querySelectorAll(".panel")) {
    const table = panel.querySelector("table");
    const body = table.tBodies[0];
    const headers = [...table.tHead.rows[0].cells];
    const search = panel.querySelector(".search");
    const freshOnly = panel.querySelector(".fresh-only");
    const tfChecks = [...panel.querySelectorAll(".tf-check")];
    const empty = panel.querySelector(".empty");

    function sortBy(index, dir) {
      const numeric = headers[index].dataset.type === "num";
      const rows = [...body.rows].sort((a, b) => {
        const x = a.cells[index].dataset.v, y = b.cells[index].dataset.v;
        const cmp = numeric ? x - y : x.localeCompare(y);
        return dir === "ascending" ? cmp : -cmp;
      });
      headers.forEach(h => h.removeAttribute("aria-sort"));
      headers[index].setAttribute("aria-sort", dir);
      body.append(...rows);
    }
    headers.forEach((h, i) => {
      if (!h.dataset.type) return;
      h.addEventListener("click", () =>
        sortBy(i, h.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending"));
    });
    function filter() {
      const term = search.value.trim().toUpperCase();
      let shown = 0;
      for (const row of body.rows) {
        const match = row.dataset.search.includes(term) &&
          (!freshOnly || !freshOnly.checked || row.classList.contains("fresh")) &&
          tfChecks.every(c => !c.checked || row.dataset[c.value] === "1");
        row.hidden = !match;
        shown += match;
      }
      empty.hidden = shown > 0;
      if (panel.dataset.count) document.getElementById(panel.dataset.count).textContent = shown;
    }
    search.addEventListener("input", filter);
    if (freshOnly) freshOnly.addEventListener("change", filter);
    tfChecks.forEach(c => c.addEventListener("change", filter));
    sortBy(Number(table.dataset.sort), "ascending");
    filter();
  }

  /* ---------- Stock detail pop-up ---------- */
  const RAW = JSON.parse(document.getElementById("details-data").textContent);
  const DETAILS = {};
  // SMA 20/50/100/200 of Heikin Ashi close for each visible candle, using the extra
  // closes before the window ("cx"). null where there isn't enough history.
  const LONG_SMAS = [20, 50, 100, 200];
  function longSmas(before, closes) {
    const full = before.concat(closes), off = before.length, out = {};
    for (const p of LONG_SMAS) {
      out["s" + p] = closes.map((_, i) => {
        const j = off + i;
        if (j + 1 < p) return null;
        let sum = 0;
        for (let k = j - p + 1; k <= j; k++) { if (full[k] == null) return null; sum += full[k]; }
        return sum / p;
      });
    }
    return out;
  }
  let showAllSmas = false;
  let showRsi = false, rsiTf = "D";
  // RSI per candle for each timeframe: {v, ma, upV, upMa}. Weekly/monthly values repeat for
  // every day of their period; "rising" compares with the previous day (D) or period (W, M).
  function decodeRsi(r, len) {
    const t = a => (a || []).map(v => v == null ? null : v / 10);
    const rose = (a, i) => i > 0 && a[i] != null && a[i - 1] != null && a[i] > a[i - 1];
    const dv = t(r.rd), dm = t(r.rdm);
    const out = {D: {v: dv, ma: dm, upV: dv.map((_, i) => rose(dv, i)), upMa: dm.map((_, i) => rose(dm, i))}};
    for (const [tf, vals, mas, idx] of [["W", r.rw, r.rwm, r.wk || RAW.wk.slice(RAW.wk.length - len)],
                                         ["M", r.rm, r.rmm, r.mo || RAW.mo.slice(RAW.mo.length - len)]]) {
      const pv = t(vals), pm = t(mas);
      out[tf] = {
        v: idx.map(k => pv[k] ?? null), ma: idx.map(k => pm[k] ?? null),
        upV: idx.map(k => rose(pv, k)), upMa: idx.map(k => rose(pm, k)),
      };
    }
    return out;
  }
  function getDetail(sym) {
    if (DETAILS[sym]) return DETAILS[sym];
    const r = RAW.stocks[sym];
    if (!r) return null;
    const f = a => a.map(v => v == null ? null : v / 100);
    const len = r.c.length;
    return DETAILS[sym] = {
      d: r.d || RAW.dates.slice(RAW.dates.length - len),
      o: f(r.o), h: f(r.h), l: f(r.l), c: f(r.c), st: f(r.st), s3: f(r.s3), s9: f(r.s9), vw: f(r.vw), up: f(r.up),
      ...longSmas(f(r.cx || []), f(r.c)),
      rsi: decodeRsi(r, len),
      dir: [...r.dir].map(ch => ch === "+" ? 1 : -1),
      close: r.close / 100, chg: r.chg / 100, name: r.n,
    };
  }
  const dlg = document.getElementById("detail");
  const chartHost = document.getElementById("detail-chart");
  const NS = "http://www.w3.org/2000/svg";
  let openSym = null;
  let span = 40;  // visible candles
  let end = 0;     // index just past the last visible candle (n = showing the latest)
  const scroller = document.getElementById("detail-scroll");
  let syncingScroll = false;
  const MIN_SPAN = 20;
  const zoomButtons = [...document.querySelectorAll(".zoom button[data-span]")];  // not the RSI D/W/M buttons

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  const money = v => v == null ? "—" : v.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const shortDate = d => new Date(d + "T12:00:00").toLocaleDateString(undefined, {month: "short", day: "numeric"});
  const longDate = d => new Date(d + "T12:00:00").toLocaleDateString(undefined, {month: "short", day: "numeric", year: "numeric"});
  function niceStep(span, target) {
    const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * mag >= raw) return m * mag;
    return 10 * mag;
  }

  function openDetail(sym) {
    const b = getDetail(sym);
    if (!b) return false;
    openSym = sym;
    document.getElementById("detail-symbol").textContent = sym;
    document.getElementById("detail-name").textContent = b.name;
    document.getElementById("detail-close").textContent = money(b.close);
    const chg = document.getElementById("detail-chg");
    chg.textContent = (b.chg >= 0 ? "+" : "") + b.chg.toFixed(2) + "%";
    chg.className = "pill " + (b.chg >= 0 ? "up" : "down");
    document.getElementById("detail-link").href =
      "https://www.tradingview.com/chart/%%TV_LAYOUT%%/?symbol=" + encodeURIComponent(sym.replace(/-/g, "."));
    span = Math.min(40, b.d.length);
    end = b.d.length;
    dlg.showModal();
    drawDetail();
    return true;
  }

  function drawDetail() {
    const b = getDetail(openSym);
    chartHost.textContent = "";
    const n = b.d.length;
    span = Math.max(Math.min(span, n), Math.min(MIN_SPAN, n));
    end = Math.max(span, Math.min(n, end));
    const start = end - span;
    zoomButtons.forEach(z => z.setAttribute("aria-pressed", Number(z.dataset.span) === span));
    const W = chartHost.clientWidth - 16, H = chartHost.clientHeight - 16;
    const pad = {l: 56, r: 92, t: 12, b: 26};
    const ihAll = H - pad.t - pad.b, rsiGap = 26, titleH = showRsi ? 16 : 0;
    // With RSI on, the RSI pane comes first (top ~32%) and the price chart sits below it.
    const rsiTop = pad.t + titleH;
    const rsiH = showRsi ? Math.round((ihAll - titleH - rsiGap) * 0.32) : 0;
    const priceTop = showRsi ? rsiTop + rsiH + rsiGap : pad.t;
    const ih = showRsi ? ihAll - titleH - rsiH - rsiGap : ihAll;
    const iw = W - pad.l - pad.r, step = iw / span;
    const extra = showAllSmas ? LONG_SMAS.map(p => b["s" + p]) : [];
    const indicators = showAllSmas ? [] : [b.st, b.up, b.vw];
    const all = [b.h, b.l, b.s3, b.s9, ...indicators, ...extra].flatMap(a => a.slice(start, end)).filter(v => v != null);
    let lo = Math.min(...all), hi = Math.max(...all);
    const margin = (hi - lo) * 0.04 || 1; lo -= margin; hi += margin;
    const x = i => pad.l + (i - start + 0.5) * step;
    const y = v => priceTop + ((hi - v) / (hi - lo)) * ih;
    const svg = el("svg", {class: "dc", viewBox: `0 0 ${W} ${H}`, tabindex: 0, role: "img",
      "aria-label": `${openSym}: last ${span} daily Heikin Ashi candles with Supertrend, SMA 3, SMA 9 and VWAP band`}, chartHost);

    // Grid + y ticks (left), date ticks (bottom)
    const ys = niceStep(hi - lo, 5);
    for (let v = Math.ceil(lo / ys) * ys; v <= hi; v += ys) {
      el("line", {x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), class: "grid"}, svg);
      el("text", {x: pad.l - 8, y: y(v) + 4, "text-anchor": "end", class: "tick"}, svg).textContent = money(v);
    }
    for (let i = start; i < end; i += Math.ceil(span / 6)) {
      el("text", {x: x(i), y: H - 6, "text-anchor": "middle", class: "tick"}, svg).textContent = shortDate(b.d[i]);
    }
    const hl = el("rect", {y: pad.t, height: ihAll, width: step, class: "hl", visibility: "hidden"}, svg);

    // VWAP band (restarts each January, so break at a year change)
    const sameYear = i => b.d[i].slice(0, 4) === b.d[i - 1].slice(0, 4);
    let runs = [], cur = [];
    for (let i = start; i < end; i++) {
      if (b.up[i] == null || b.vw[i] == null || (i > start && !sameYear(i))) { if (cur.length) runs.push(cur); cur = []; }
      if (b.up[i] != null && b.vw[i] != null) cur.push(i);
    }
    if (cur.length) runs.push(cur);
    for (const r of showAllSmas ? [] : runs) {  // hidden in "All SMAs" mode
      const top = r.map(i => `${x(i)},${y(b.up[i])}`), bot = r.slice().reverse().map(i => `${x(i)},${y(b.vw[i])}`);
      el("polygon", {points: top.concat(bot).join(" "), class: "band-fill"}, svg);
      el("polyline", {points: top.join(" "), class: "band"}, svg);
      el("polyline", {points: r.map(i => `${x(i)},${y(b.vw[i])}`).join(" "), class: "vwap"}, svg);
    }

    // Heikin Ashi candles
    const cg = el("g", {class: "candles"}, svg);
    const bw = Math.max(step * 0.6, 1.5);
    for (let i = start; i < end; i++) {
      const cls = b.c[i] >= b.o[i] ? "c-up" : "c-down";
      const top = y(Math.max(b.o[i], b.c[i])), bottom = y(Math.min(b.o[i], b.c[i]));
      el("line", {x1: x(i), x2: x(i), y1: y(b.h[i]), y2: y(b.l[i]), class: cls, "stroke-width": 1}, cg);
      el("rect", {x: x(i) - bw / 2, y: top, width: bw, height: Math.max(bottom - top, 1), rx: 1, class: cls}, cg);
    }

    // Lines: each segment green when the value rose from the previous candle, gray otherwise.
    function segments(values, width, breakAt) {
      const paths = {rise: "", flat: ""};
      for (let i = start + 1; i < end; i++) {
        if (values[i] == null || values[i - 1] == null || (breakAt && breakAt(i))) continue;
        const kind = values[i] > values[i - 1] ? "rise" : "flat";
        paths[kind] += `M${x(i - 1)},${y(values[i - 1])}L${x(i)},${y(values[i])}`;
      }
      for (const [kind, d] of Object.entries(paths)) if (d) el("path", {d, class: `seg ${kind} ${width}`}, svg);
    }
    if (showAllSmas) for (const p of [200, 100, 50, 20]) segments(b["s" + p], "w-s" + p);
    segments(b.s9, "w-s9");
    segments(b.s3, "w-s3");
    if (!showAllSmas) segments(b.st, "w-st", i => b.dir[i] !== b.dir[i - 1]);

    // Supertrend crossing the VWAP upper band. Ignore Supertrend flips (the line jumps
    // between bands) and the January VWAP reset, which aren't real crossings.
    const crossings = {};
    for (let i = showAllSmas ? end : start + 1; i < end; i++) {  // none in "All SMAs" mode
      const s0 = b.st[i - 1], s1 = b.st[i], u0 = b.up[i - 1], u1 = b.up[i];
      if ([s0, s1, u0, u1].some(v => v == null) || b.dir[i] !== b.dir[i - 1] || !sameYear(i)) continue;
      const d0 = s0 - u0, d1 = s1 - u1;
      if ((d0 <= 0 && d1 > 0) || (d0 >= 0 && d1 < 0)) {
        const kind = d1 > 0 ? "up" : "down";
        const t = d0 === d1 ? 1 : d0 / (d0 - d1);  // where the two lines meet between the candles
        const cx = x(i - 1) + t * step, cy = y(s0 + t * (s1 - s0));
        crossings[i] = kind;
        el("circle", {cx, cy, r: 10, class: `mk-halo ${kind}`}, svg);
        el("circle", {cx, cy, r: 5.5, class: `mk-${kind}`}, svg);
        el("text", {x: cx, y: kind === "up" ? cy - 14 : cy + 22, "text-anchor": "middle", class: `mk-label ${kind}`},
          svg).textContent = kind === "up" ? "\u25B2" : "\u25BC";
      }
    }

    // Right-edge labels, nudged apart so they never overlap
    const last = end - 1;
    syncScrollbar(n, start);
    const labels = (showAllSmas
      ? [["SMA 3", b.s3[last]], ["SMA 9", b.s9[last]], ...LONG_SMAS.map(p => ["SMA " + p, b["s" + p][last]])]
      : [["Supertrend", b.st[last]], ["SMA 3", b.s3[last]], ["SMA 9", b.s9[last]], ["VWAP band", b.up[last]]])
      .filter(([, v]) => v != null).map(([t, v]) => ({t, y: y(v)})).sort((a, c) => a.y - c.y);
    for (let k = 1; k < labels.length; k++) labels[k].y = Math.max(labels[k].y, labels[k - 1].y + 13);
    for (const lab of labels) el("text", {x: W - pad.r + 8, y: lab.y + 4, class: "end"}, svg).textContent = lab.t;

    // RSI pane (0-100 scale of its own, below the price pane)
    const R = b.rsi[rsiTf];
    if (showRsi) {
      const yr = v => rsiTop + ((100 - v) / 100) * rsiH;
      el("text", {x: pad.l, y: rsiTop - 10, class: "pane-title"}, svg).textContent =
        `RSI 14 \u00b7 ${{D: "Daily", W: "Weekly", M: "Monthly"}[rsiTf]}`;
      el("rect", {x: pad.l, y: yr(70), width: iw, height: yr(30) - yr(70), class: "rsi-zone"}, svg);
      for (const v of [30, 50, 70]) {
        el("line", {x1: pad.l, x2: W - pad.r, y1: yr(v), y2: yr(v), class: v === 50 ? "grid" : "rsi-level"}, svg);
        el("text", {x: pad.l - 8, y: yr(v) + 4, "text-anchor": "end", class: "tick"}, svg).textContent = v;
      }
      const rsiLine = (values, rising, width) => {
        const paths = {rise: "", flat: ""};
        for (let i = start + 1; i < end; i++) {
          if (values[i] == null || values[i - 1] == null) continue;
          paths[rising[i] ? "rise" : "flat"] += `M${x(i - 1)},${yr(values[i - 1])}L${x(i)},${yr(values[i])}`;
        }
        for (const [kind, d] of Object.entries(paths)) if (d) el("path", {d, class: `seg ${kind} ${width}`}, svg);
      };
      rsiLine(R.ma, R.upMa, "w-rsima");
      rsiLine(R.v, R.upV, "w-rsi");
      const rl = [["RSI", R.v[last]], ["RSI SMA", R.ma[last]]].filter(([, v]) => v != null)
        .map(([t, v]) => ({t, y: yr(v)})).sort((a, c) => a.y - c.y);
      for (let k = 1; k < rl.length; k++) rl[k].y = Math.max(rl[k].y, rl[k - 1].y + 13);
      for (const lab of rl) el("text", {x: W - pad.r + 8, y: lab.y + 4, class: "end"}, svg).textContent = lab.t;
    }

    // Crosshair + readout
    const cross = el("line", {y1: pad.t, y2: pad.t + ihAll, class: "cross", visibility: "hidden"}, svg);
    const readout = document.getElementById("detail-readout");
    let idx = last;
    function items(i) {
      const up = (arr, extra) => i > 0 && arr[i] != null && arr[i - 1] != null && arr[i] > arr[i - 1] && (!extra || extra());
      return [
        ...(showRsi ? [[`RSI (${rsiTf})`, R.v[i] == null ? "\u2014" : R.v[i].toFixed(1), R.upV[i]],
                       ["RSI SMA", R.ma[i] == null ? "\u2014" : R.ma[i].toFixed(1), R.upMa[i]]] : []),
        ["O", money(b.o[i])], ["H", money(b.h[i])], ["L", money(b.l[i])], ["C", money(b.c[i])],
        ...(showAllSmas ? [] : [["Supertrend", money(b.st[i]), up(b.st, () => b.dir[i] === b.dir[i - 1])]]),
        ["SMA 3", money(b.s3[i]), up(b.s3)],
        ["SMA 9", money(b.s9[i]), up(b.s9)],
        ...(showAllSmas ? LONG_SMAS.map(p => ["SMA " + p, money(b["s" + p][i]), up(b["s" + p])]) : []),

        ...(showAllSmas ? [] : [["VWAP band", money(b.up[i])], ["VWAP", money(b.vw[i])]]),
      ];
    }
    // The strip shows the hovered candle, or the latest candle when not hovering.
    function render(i) {
      readout.textContent = "";
      const date = document.createElement("span");
      date.className = "r-date";
      date.textContent = longDate(b.d[i]);
      readout.append(date);
      for (const [label, value, rising] of items(i)) {
        const item = document.createElement("span"), l = document.createElement("span"), v = document.createElement("b");
        item.className = "r-item";
        l.textContent = label;
        v.textContent = value + (rising ? " \u25B2" : "");
        if (rising) v.className = "rise";
        item.append(l, v);
        readout.append(item);
      }
      if (crossings[i]) {
        const note = document.createElement("span");
        note.className = "r-cross " + crossings[i];
        note.textContent = crossings[i] === "up" ? "\u25B2 Crossed above band" : "\u25BC Crossed below band";
        readout.append(note);
      }
    }
    function show(i) {
      idx = Math.max(start, Math.min(last, i));
      cross.setAttribute("x1", x(idx)); cross.setAttribute("x2", x(idx)); cross.setAttribute("visibility", "visible");
      hl.setAttribute("x", x(idx) - step / 2); hl.setAttribute("visibility", "visible");
      render(idx);
    }
    function hide() { cross.setAttribute("visibility", "hidden"); hl.setAttribute("visibility", "hidden"); render(last); }
    render(last);
    svg.addEventListener("pointermove", e => {
      const r = svg.getBoundingClientRect();
      show(start + Math.floor(((e.clientX - r.left) * (W / r.width) - pad.l) / step));
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", () => show(idx));
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", e => {
      const d = {ArrowLeft: -1, ArrowRight: 1, Home: -1e9, End: 1e9}[e.key];
      if (d) { e.preventDefault(); show(idx + d); }
    });
  }

  document.addEventListener("click", e => {
    const target = e.target.closest(".stock a, .spark-cell");
    if (!target || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    const sym = target.closest("tr").cells[0].dataset.v;
    if (openDetail(sym)) e.preventDefault();
  });
  function zoomTo(next) {
    if (!openSym) return;
    const n = getDetail(openSym).d.length;
    next = Math.max(Math.min(MIN_SPAN, n), Math.min(n, Math.round(next)));
    if (next !== span) { span = next; drawDetail(); }
  }
  function panTo(nextEnd) {
    if (!openSym) return;
    const n = getDetail(openSym).d.length;
    nextEnd = Math.max(span, Math.min(n, Math.round(nextEnd)));
    if (nextEnd !== end) { end = nextEnd; drawDetail(); }
  }
  // Scrollbar width mirrors the zoom: thumb = visible part of the history.
  function syncScrollbar(n, start) {
    scroller.hidden = span >= n;
    if (scroller.hidden) return;
    scroller.firstElementChild.style.width = (scroller.clientWidth * n / span) + "px";
    syncingScroll = true;
    scroller.scrollLeft = (start / (n - span)) * (scroller.scrollWidth - scroller.clientWidth);
    requestAnimationFrame(() => { syncingScroll = false; });
  }
  let scrollFrame = 0;
  scroller.addEventListener("scroll", () => {
    if (syncingScroll || !openSym) return;
    cancelAnimationFrame(scrollFrame);
    scrollFrame = requestAnimationFrame(() => {
      const n = getDetail(openSym).d.length;
      const room = scroller.scrollWidth - scroller.clientWidth;
      panTo(span + (room > 0 ? scroller.scrollLeft / room : 1) * (n - span));
    });
  });
  zoomButtons.forEach(z => z.addEventListener("click", () => zoomTo(Number(z.dataset.span))));
  // "All SMAs" adds SMA 20/50/100/200; the choice is kept while the page is open.
  // RSI pane toggle + timeframe; both choices are kept while the page is open.
  const rsiTfGroup = document.querySelector(".rsi-tf");
  document.getElementById("show-rsi").addEventListener("change", e => {
    showRsi = e.target.checked;
    rsiTfGroup.hidden = !showRsi;
    chartHost.classList.toggle("with-rsi", showRsi);
    if (openSym) drawDetail();
  });
  rsiTfGroup.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    rsiTf = btn.dataset.tf;
    rsiTfGroup.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === btn));
    if (openSym) drawDetail();
  }));
  document.getElementById("show-smas").addEventListener("change", e => {
    showAllSmas = e.target.checked;
    document.querySelectorAll(".long-sma").forEach(el => { el.hidden = !showAllSmas; });
    // SMA mode shows only candles and SMAs: hide Supertrend, VWAP band and crossing markers.
    document.querySelectorAll(".dlg-legend .ind").forEach(el => { el.hidden = showAllSmas; });
    if (openSym) drawDetail();
  });
  // Mouse wheel up/down or trackpad pinch zooms (keeping the right edge in place).
  // Sideways swipe or Shift + wheel pans through dates.
  let panCarry = 0;
  chartHost.addEventListener("wheel", e => {
    e.preventDefault();
    const sideways = e.shiftKey ? e.deltaY || e.deltaX : e.deltaX;
    if (Math.abs(sideways) > Math.abs(e.shiftKey ? 0 : e.deltaY)) {
      const barWidth = chartHost.clientWidth / span;
      panCarry += sideways / Math.max(barWidth, 1);
      const bars = Math.trunc(panCarry);
      if (bars) { panCarry -= bars; panTo(end + bars); }
    } else {
      zoomTo(span * Math.pow(1.0015, e.deltaY * (e.ctrlKey ? 4 : 1)));
    }
  }, {passive: false});
  document.getElementById("detail-close-btn").addEventListener("click", () => dlg.close());

  // Symbol lookup: exact symbol (BRK.B or BRK-B), exact company name, or the first name match.
  const lookup = document.getElementById("lookup");
  const lookupInput = document.getElementById("lookup-input");
  const lookupMsg = document.getElementById("lookup-msg");
  const NAMES = Object.entries(RAW.stocks).map(([sym, r]) => [sym, r.n.toUpperCase()]);
  function resolve(text) {
    const q = text.trim().toUpperCase();
    if (!q) return null;
    const sym = q.replace(/\\./g, "-");
    if (RAW.stocks[sym]) return sym;
    const hit = NAMES.find(([, n]) => n === q) || NAMES.find(([, n]) => n.startsWith(q)) || NAMES.find(([, n]) => n.includes(q));
    return hit ? hit[0] : null;
  }
  function chartFromLookup() {
    const sym = resolve(lookupInput.value);
    if (sym && openDetail(sym)) {
      lookupMsg.hidden = true;
      lookupInput.value = "";
      lookupInput.blur();
    } else if (lookupInput.value.trim()) {
      lookupMsg.textContent = `No S&P 500 match for \u201c${lookupInput.value.trim()}\u201d`;
      lookupMsg.hidden = false;
    }
  }
  lookup.addEventListener("submit", e => { e.preventDefault(); chartFromLookup(); });
  lookupInput.addEventListener("input", e => {
    lookupMsg.hidden = true;
    // Picking a suggestion from the list opens the chart right away.
    if ((!e.inputType || e.inputType === "insertReplacementText") && RAW.stocks[lookupInput.value]) chartFromLookup();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "/" && !dlg.open && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      lookupInput.focus();
    }
  });
  dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });  // click on the backdrop
  dlg.addEventListener("close", () => { openSym = null; });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (openSym) drawDetail(); }, 100);
  });
})();
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


def save_signals(
    signals: list[Signal], path: Path, as_of: str, keep_days: int = 0
) -> dict[str, list[dict]]:
    """Add the day's filtered list to a JSON file of {date: [stocks...]}.

    Earlier days are kept; running again on the same day replaces that day's
    entry (e.g. the 5 PM run replaces the intraday 12 PM list).
    """
    return write_history(path, {as_of: signal_records(signals)}, keep_days)


def write_history(
    path: Path, days: dict[str, list[dict]], keep_days: int = 0
) -> dict[str, list[dict]]:
    """Merge {date: records} into the JSON file, replacing those dates.

    With keep_days > 0, dates more than keep_days calendar days before the newest
    date in the file are removed.
    """
    history: dict[str, list[dict]] = {}
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
    history.update(days)
    if keep_days > 0 and history:
        cutoff = (pd.Timestamp(max(history)) - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
        removed = [day for day in history if day < cutoff]
        for day in removed:
            del history[day]
        if removed:
            print(f"Removed {len(removed)} days older than {cutoff} from {path}")
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
    write_history(args.signals_file, filled, args.keep_days)
    print(f"Backfilled {len(filled)} trading days ({trading_days[0]} to {trading_days[-1]}) -> {args.signals_file}")


ALERT_LISTS = {
    "flipped": "Bullish Re-entry",
    "dwm": "Multi-Timeframe Rising",
    "above": "Above VWAP Band",
    "sma": "SMA Gaining",
    "rsi": "RSI 70",
}


def update_alerts(
    lists: dict[str, list[str]], state_path: Path, alert_path: Path | None,
    details: dict[str, dict], companies: dict[str, str], as_of: str,
) -> dict[str, list[str]]:
    """Save this run's lists and, if a previous run's lists exist, write an alert email
    (see write_alert_email) for symbols that weren't in them. Returns {list key: new symbols}."""
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    state_path.write_text(json.dumps({"as_of": as_of, "lists": lists}, indent=2) + "\n", encoding="utf-8")
    if previous is None:
        return {}  # first run: just record the baseline
    # A list missing from the saved state (e.g. newly added to alerts) only records its baseline.
    new = {
        key: [s for s in syms if s not in previous["lists"][key]] if key in previous["lists"] else []
        for key, syms in lists.items()
    }
    if not alert_path or not any(new.values()):
        return new

    write_alert_email(new, alert_path, details, companies, as_of)
    return new


def write_alert_email(
    new: dict[str, list[str]], path: Path, details: dict[str, dict], companies: dict[str, str], as_of: str
) -> None:
    """Write the alert email: <path> (plain text, first line = subject) and <path>.html."""
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%-I:%M %p %Z")
    counts = ", ".join(f"{len(syms)} {ALERT_LISTS[key]}" for key, syms in new.items() if syms)
    subject = f"New signals {as_of}: {counts}"
    intro = f"Data as of {as_of}, run at {now}. These stocks were not in the lists on the previous run."

    text = [subject, "", intro]
    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1d1d1b;max-width:640px">',
        f'<h2 style="font-size:18px;margin:0 0 4px">{html.escape(subject)}</h2>',
        f'<p style="color:#6b6b66;font-size:13px;margin:0 0 16px">{html.escape(intro)}</p>',
    ]
    for key, syms in new.items():
        if not syms:
            continue
        title = f"{ALERT_LISTS[key]} ({len(syms)} new)"
        text += ["", title]
        parts += [
            f'<h3 style="font-size:15px;margin:20px 0 6px">{html.escape(title)}</h3>',
            '<table cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;width:100%">',
            '<tr style="text-align:left;color:#6b6b66;border-bottom:1px solid #e4e4df">'
            '<th>Symbol</th><th>Company</th><th style="text-align:right">Close</th><th style="text-align:right">Day</th></tr>',
        ]
        for sym in syms:
            det = details.get(sym)
            name = companies.get(sym, "")
            close = f"{det['close']:,.2f}" if det else ""
            chg = f"{det['chg']:+.2f}%" if det else ""
            color = "#1f9d5c" if det and det["chg"] >= 0 else "#d0473d"
            text.append(f"  {sym:<6} {name[:32]:<32} {close:>10} {chg:>8}")
            parts.append(
                '<tr style="border-bottom:1px solid #f0efec">'
                f'<td><a href="{html.escape(tradingview_url(sym))}" '
                f'style="color:#2a78d6;font-weight:700;text-decoration:none">{html.escape(sym)}</a></td>'
                f"<td>{html.escape(name)}</td>"
                f'<td style="text-align:right">{close}</td>'
                f'<td style="text-align:right;color:{color};font-weight:600">{chg}</td></tr>'
            )
        parts.append("</table>")
    parts.append("</div>")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    path.with_suffix(".html").write_text("\n".join(parts) + "\n", encoding="utf-8")


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
        "--keep-days",
        type=int,
        default=90,
        help="remove days older than this many calendar days from signals.json (default: 90; 0 keeps everything)",
    )
    parser.add_argument(
        "--alert-file",
        type=Path,
        help="write an alert email here (plain text, plus .html) when stocks newly join Bullish Re-entry, Multi-Timeframe Rising, Above VWAP Band, SMA Gaining or RSI 70",
    )
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help="one-off: recompute signals.json for each trading day in the last DAYS calendar days",
    )
    args = parser.parse_args()
    if args.keep_days < 0:
        parser.error("--keep-days must be 0 or greater")
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
            result = analyze(path, args.period, args.multiplier, args.vwap_band, multi=True)
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if result:
            results.append(result)

    current, as_of = current_results(results)
    bullish = sum(r.bullish for r in current)
    signals = filtered(current)

    # Save today's list first so the "flipped since listed" tab also covers today's symbols.
    history = save_signals(signals, args.signals_file, as_of, args.keep_days)
    flipped = find_flipped(current, history, as_of)
    watched = len({stock["symbol"] for stocks in history.values() for stock in stocks})

    companies_path = args.data_dir / "companies.json"
    companies = json.loads(companies_path.read_text(encoding="utf-8")) if companies_path.exists() else {}
    rising = find_rising(signals)
    # Stocks whose Supertrend is rising on at least one timeframe; the page filters further.
    mtfs = sorted(
        (r.mtf for r in current if r.mtf and any(v.rising for v in r.mtf.steps.values())),
        key=lambda m: m.symbol,
    )
    # Chart data for every current stock, so the lookup box can chart any symbol.
    details = {r.detail["sym"]: r.detail for r in current if r.detail}
    # Stocks meeting at least one SMA condition; the page filters further.
    smas = sorted(
        (r.mtf for r in current if r.mtf and any(r.mtf.sma_conditions.values())), key=lambda m: m.symbol
    )
    # Stocks meeting at least one RSI condition; the page filters further.
    rsis = sorted(
        (r.mtf for r in current if r.mtf and any(r.mtf.rsi_conditions.values())), key=lambda m: m.symbol
    )
    page = render(
        signals, rising, flipped, mtfs, smas, rsis, details, companies, history, watched, len(current), bullish,
        as_of, args.period, args.multiplier, args.vwap_band,
    )
    args.output.write_text(page, encoding="utf-8")

    lists = {
        "flipped": sorted(f.signal.symbol for f in flipped),
        "dwm": [m.symbol for m in mtfs if all(v.rising for v in m.steps.values())],
        "above": sorted(s.symbol for s in signals),
        # SMA Gaining with its default filters: all six SMAs gaining and stacked 3 > 9 > 20 > 50 > 100 > 200.
        "sma": [m.symbol for m in smas if all(m.sma_conditions.values())],
        # RSI 70 with its default filters: daily, weekly and monthly RSI above 70 and above their SMA.
        "rsi": [m.symbol for m in rsis if all(m.rsi_conditions.values())],
    }
    new = update_alerts(
        lists, args.data_dir / "alerts_state.json", args.alert_file, details, companies, as_of
    )
    if any(new.values()):
        print("New: " + "; ".join(f"{ALERT_LISTS[k]}: {', '.join(v)}" for k, v in new.items() if v))
    print(
        f"{len(signals)} of {len(current)} stocks have a bullish Supertrend above the "
        f"VWAP upper band ({bullish} bullish), {len(rising)} of them increasing; {len(flipped)} of {watched} listed symbols "
        f"flipped bullish since listed with a rising Supertrend; as of {as_of} -> {args.output}, {args.signals_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
