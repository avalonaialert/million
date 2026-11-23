#!/usr/bin/env python3
"""Download maximum-available historical data for S&P 500 constituents from Yahoo Finance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_symbols() -> list[str]:
    """Fetch the current S&P 500 constituents and return Yahoo Finance symbols."""
    tables = pd.read_html(CONSOLIDUENTS_URL)
    symbols = tables[0]["Symbol"].dropna().astype(str)
    # Yahoo Finance uses hyphens for share classes (e.g. BRK.B -> BRK-B).
    return symbols.str.replace(".", "-", regex=False).tolist()


def download_symbol(symbol: str, output_dir: Path) -> bool:
    """Save all available price history, including dividends and splits, as CSV."""
    try:
        history = yf.Ticker(symbol).history(
            period="max",
            auto_adjust=False,
            actions=True,
            raise_errors=True,
        )
        if history.empty:
            print(f"No historical data returned for {symbol}")
            return False

        path = output_dir / f"{symbol.replace('/', '-')}.csv"
        history.to_csv(path, index_label="Date")
        print(f"Saved {symbol}: {len(history)} rows -> {path}")
        return True
    except Exception as exc:
        print(f"Failed {symbol}: {exc}")
        return False


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
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be zero or greater")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        symbols = get_symbols()
    except Exception as exc:
        print(f"Could not retrieve the S&P 500 constituent list: {exc}")
        return 1

    print(f"Downloading historical data for {len(symbols)} S&P 500 symbols...")
    succeeded = 0
    failed: list[str] = []
    for index, symbol in enumerate(symbols):
        if download_symbol(symbol, args.output_dir):
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
