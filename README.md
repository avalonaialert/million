# S&P 500 historical data downloader

Fetches the current S&P 500 constituent list and downloads each symbol's maximum available historical data from Yahoo Finance. Each symbol is saved as its own CSV file in `data/` (for example, `data/AAPL.csv`). The CSV includes price history, dividends, and stock splits.

## Run

```bash
python -m pip install -r requirements.txt
python download_sp500.py
```

To choose another output directory or adjust the delay between Yahoo Finance requests:

```bash
python download_sp500.py --output-dir data --delay 0.5
```

The script creates the output directory if needed. It reports symbols that fail and exits with a non-zero status if any download fails. Running it again refreshes the CSV files with the latest available history.

Data availability and use are subject to Yahoo Finance's terms and the source of the constituent list.
