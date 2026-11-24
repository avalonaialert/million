#!/usr/bin/env python3
"""Download maximum-available historical data for S&P 500 constituents from Yahoo Finance."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from io import StringIO
from pathlib import Path

import certifi
import pandas as pd
import yfinance as yf

# Raise download errors instead of logging them and returning an empty frame,
# so failed symbols are reported (replaces the deprecated raise_errors=True).
yf.config.debug.hide_exceptions = False

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_constituents() -> dict[str, str]:
    """Fetch the current S&P 500 constituents as {Yahoo Finance symbol: company name}."""
    # Use certifi's CA bundle: python.org macOS builds ship without system root
    # certificates, which causes CERTIFICATE_VERIFY_FAILED. Wikipedia also
    # rejects requests without a User-Agent.
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        CONSTITUENTS_URL, headers={"User-Agent": "sp500-downloader/1.0"}
    )
    with urllib.request.urlopen(request, context=context, timeout=30) as response:
        html = response.read().decode("utf-8")
    table = pd.read_html(StringIO(html))[0].dropna(subset=["Symbol"])
    # Yahoo Finance uses hyphens for share classes (e.g. BRK.B -> BRK-B).
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return dict(zip(symbols, table["Security"].astype(str)))


def get_symbols() -> list[str]:
    """Fetch the current S&P 500 constituents and return Yahoo Finance symbols."""
    return list(get_constituents())


def save_companies(companies: dict[str, str], path: Path) -> None:
    """Merge {symbol: name} into a JSON file, keeping names of past constituents."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(companies)
    path.write_text(json.dumps(dict(sorted(existing.items())), indent=2) + "\n", encoding="utf-8")


def fetch_history(symbol: str, start: str | None = None) -> pd.DataFrame:
    """Fetch price history from `start` (YYYY-MM-DD) to now, or all history if None."""
    kwargs = {"start": start} if start else {"period": "max"}
    return yf.Ticker(symbol).history(
        auto_adjust=False, actions=True, **kwargs
    )


def to_string_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Convert a history frame to the exact strings to_csv would write."""
    buffer = StringIO()
    history.to_csv(buffer, index_label="Date")
    buffer.seek(0)
    return pd.read_csv(buffer, dtype=str, keep_default_na=False)


def download_symbol(symbol: str, output_dir: Path, full: bool = False) -> bool:
    """Create or update a symbol's CSV of price history, dividends and splits.

    If the CSV exists, only rows from its last date onward are fetched; the
    last row is re-fetched because it may be a partial intraday value. If any
    new dividend or split appears, the whole history is re-downloaded because
    Yahoo recalculates "Adj Close" for every earlier row.
    """
    path = output_dir / f"{symbol.replace('/', '-')}.csv"
    try:
        if full or not path.exists():
            return save_full(symbol, path)

        # Read existing rows as raw strings so they are rewritten byte-for-byte.
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if existing.empty:
            return save_full(symbol, path)
        last_date = existing["Date"].iloc[-1][:10]

        recent = fetch_history(symbol, start=last_date)
        if recent.empty:
            print(f"No new data for {symbol}")
            return True

        new_rows = recent[recent.index.strftime("%Y-%m-%d") > last_date]
        actions = [c for c in ("Dividends", "Stock Splits") if c in new_rows]
        if (new_rows[actions] != 0).any().any():
            print(f"{symbol}: new dividend/split, refreshing full history")
            return save_full(symbol, path)

        kept = existing[existing["Date"].str[:10] < recent.index[0].strftime("%Y-%m-%d")]
        merged = pd.concat([kept, to_string_frame(recent)], ignore_index=True)
        merged.to_csv(path, index=False)
        print(f"Updated {symbol}: {len(new_rows)} new rows -> {path}")
        return True
    except Exception as exc:
        print(f"Failed {symbol}: {exc}")
        return False


def save_full(symbol: str, path: Path) -> bool:
    """Download and save the complete price history for a symbol."""
    history = fetch_history(symbol)
    if history.empty:
        print(f"No historical data returned for {symbol}")
        return False
    history.to_csv(path, index_label="Date")
    print(f"Saved {symbol}: {len(history)} rows -> {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="directory for per-symbol CSV files (default: data/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="seconds to wait between Yahoo Finance requests (default: 0.25)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="re-download the full history for every symbol instead of updating",
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be zero or greater")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        companies = get_constituents()
    except Exception as exc:
        print(f"Could not retrieve the S&P 500 constituent list: {exc}")
        return 1
    symbols = list(companies)
    save_companies(companies, args.output_dir / "companies.json")

    print(f"Downloading historical data for {len(symbols)} S&P 500 symbols...")
    succeeded = 0
    failed: list[str] = []
    for index, symbol in enumerate(symbols):
        if download_symbol(symbol, args.output_dir, full=args.full):
            succeeded += 1
        else:
            failed.append(symbol)
        if index < len(symbols) - 1 and args.delay:
            time.sleep(args.delay)

    print(f"Finished: {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed symbols: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
