"""Data: the frozen Yahoo Finance closes, the Turkey 5-year CDS series, and the panel on ASELSAN trading days.

Files (data/):
  yahoo_close.csv     daily closes of the eight Yahoo series, each on its own trading calendar, downloaded once by
                      data/fetch_yahoo.py for the window of version 1 (see yahoo_meta.json)
  cds_turkey_5y.xlsx  Turkey 5-year sovereign CDS spread (basis points), the file of version 1 (dataset.xlsx)

Alignment (no look-ahead): the panel has one row per ASELSAN trading day. Every other series enters with its
latest close on or before that day ("as of"). Nothing is filled backwards, and the panel ends on the last ASELSAN
trading day on or before the end of the shortest series (the CDS file ends on 8 January 2026), so that no series is
carried forward past the end of its own data.

Author: Muhammed Ikbal Yilmaz
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SERIES = ["ASELS", "BIST100", "USDTRY", "BITCOIN", "BRENT", "VIX", "DXY", "US10Y", "CDS"]
PRICE_LIKE = ["ASELS", "BIST100", "USDTRY", "BITCOIN", "BRENT"]     # 100 x log returns
RATE_LIKE = ["VIX", "DXY", "US10Y", "CDS"]                          # first differences (or levels)


def load_yahoo(path=None):
    """Wide table of closes, one column per series, on the union of all trading dates (NaN where closed)."""
    d = pd.read_csv(path or DATA / "yahoo_close.csv", parse_dates=["date"])
    return d.pivot(index="date", columns="series", values="close").sort_index()


def load_cds(path=None):
    """Turkey 5-year CDS (bp) as a Series indexed by date. The file stores dates as dd.mm.yyyy."""
    raw = pd.read_excel(path or DATA / "cds_turkey_5y.xlsx")
    s = pd.Series(pd.to_numeric(raw["CDS"], errors="raise").values,
                  index=pd.to_datetime(raw["date"], format="%d.%m.%Y"), name="CDS").sort_index()
    if s.index.duplicated().any():
        raise ValueError("duplicate dates in the CDS file")
    return s


def asof_panel(yahoo=None, cds=None):
    """Levels on ASELSAN trading days, each series as of its latest close on or before the day, together with
    the age (calendar days) of that close. Returns (levels, age)."""
    yahoo = load_yahoo() if yahoo is None else yahoo
    cds = load_cds() if cds is None else cds
    days = yahoo["ASELS"].dropna().index
    levels, age = {}, {}
    for name in SERIES:
        s = (cds if name == "CDS" else yahoo[name]).dropna()
        pos = s.index.searchsorted(days, side="right") - 1          # last observation on or before each day
        ok = pos >= 0
        v = np.full(len(days), np.nan)
        a = np.full(len(days), np.nan)
        v[ok] = s.values[pos[ok]]
        a[ok] = (days[ok] - s.index[pos[ok]]).days
        levels[name], age[name] = v, a
    levels = pd.DataFrame(levels, index=days)
    age = pd.DataFrame(age, index=days)
    # end of the panel: the last ASELSAN day on or before the end of the shortest series
    last_obs = {n: (cds if n == "CDS" else yahoo[n]).dropna().index[-1] for n in SERIES}
    end = min(last_obs.values())
    keep = levels.index <= days[days.searchsorted(end, side="right") - 1]
    keep &= levels.notna().all(axis=1)
    return levels[keep], age[keep]


def transform(levels, vix="diff", cds="diff"):
    """r_X = 100 (log X_t - log X_{t-1}) for price-like series; d_X = X_t - X_{t-1} for rate-like series, or the
    level itself (named X) when vix/cds = "level". Differences are between consecutive ASELSAN trading days."""
    out = pd.DataFrame(index=levels.index)
    for c in PRICE_LIKE:
        out[f"r_{c}"] = 100.0 * np.log(levels[c]).diff()
    for c in RATE_LIKE:
        mode = {"VIX": vix, "CDS": cds}.get(c, "diff")
        out[c if mode == "level" else f"d_{c}"] = levels[c] if mode == "level" else levels[c].diff()
    return out
