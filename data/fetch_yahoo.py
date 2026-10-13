"""Downloads the Yahoo Finance series of the project once and saves them, so that every later step uses a fixed file.

Run from the repository root:  python data/fetch_yahoo.py
Output: data/yahoo_close.csv (one row per date and series, each series on its own trading calendar) and
data/yahoo_meta.json (download time, yfinance version, date window, tickers).

The window is the one of version 1: 2 January 2020 to 12 January 2026 (yfinance treats `end` as exclusive).
Prices are the "Close" column with yfinance's default auto_adjust=True, i.e. adjusted for splits and dividends,
which is what version 1 used (its log printed "'Adj Close' missing, using 'Close'").

Author: Muhammed Ikbal Yilmaz
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

START, END = "2020-01-01", "2026-01-13"
TICKERS = {                       # name in the project: Yahoo symbol (as in version 1)
    "ASELS": "ASELS.IS",
    "BIST100": "XU100.IS",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "USDTRY": "TRY=X",
    "BITCOIN": "BTC-USD",
    "BRENT": "BZ=F",
}
OUT = Path(__file__).resolve().parent

frames = []
for name, symbol in TICKERS.items():
    raw = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):           # newer yfinance: (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw.columns:
        sys.exit(f"No data for {symbol}; nothing was saved. Check the internet connection and run again.")
    s = raw["Close"].dropna()
    frames.append(pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "series": name, "symbol": symbol,
                                "close": s.values}))
    print(f"{name:8s} {symbol:9s} {len(s):5d} days  {s.index[0].date()} to {s.index[-1].date()}")

data = pd.concat(frames, ignore_index=True)
data.to_csv(OUT / "yahoo_close.csv", index=False, float_format="%.10g")
meta = {"downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "yfinance_version": yf.__version__, "pandas_version": pd.__version__,
        "start": START, "end_exclusive": END, "auto_adjust": True, "tickers": TICKERS, "rows": int(len(data))}
(OUT / "yahoo_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"\nSaved {len(data)} rows to data/yahoo_close.csv and data/yahoo_meta.json")
