"""Analysis 1: the data, the alignment, and version 1 reproduced.

Why: version 1 downloaded the Yahoo series again at every run, up to the day of the run, so its numbers could not
be reproduced; and the alignment of series that trade on different calendars decides what "the same day" means.
Here the series are frozen once for the window of version 1, version 1's own code is run on them, and the panel of
version 2 is described.

Design:
* Table 1: for each series, its closes, the share of ASELSAN days on which its value is carried from an earlier
  date (market closed that day), and the last day available.
* Table 2: the last row of version 1 against the final closes of that day (version 1 ran during the trading day).
* Table 3: version 1's printed results against version 1's code run on the frozen data.
* Figure 1: (a) cumulative log returns of ASELSAN and BIST100; (b) the 120-day rolling market beta of version 1.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import save_table
from aselsan.data import SERIES, asof_panel, load_cds, load_yahoo, transform
from aselsan.legacy import run_v1
from aselsan.style import BLUE, INK, RED, auto_axis, end_label, free_axis, may_touch, save_figure, use_style

yahoo, cds = load_yahoo(), load_cds()
levels, age = asof_panel(yahoo, cds)
trans = transform(levels)

# Table 1
rows = []
for name in SERIES:
    s = (cds if name == "CDS" else yahoo[name]).dropna()
    rows.append({"series": name, "closes": len(s), "first": s.index[0].strftime("%Y-%m-%d"),
                 "last": s.index[-1].strftime("%Y-%m-%d"), "carried_days": int((age[name] > 0).sum()),
                 "carried_share": float((age[name] > 0).mean())})
save_table(rows, "table1_data")

# Table 2: the last row printed by version 1 (its output of 12 January 2026) against the final closes
V1_LAST = {"ASELS": 283.00, "BITCOIN": 91110.523438, "BRENT": 63.400002, "DXY": 98.897003, "USDTRY": 43.107990,
           "BIST100": 12254.830078, "US10Y": 4.179, "VIX": 15.17, "CDS": 214.44}
day = pd.Timestamp("2026-01-12")
rows = []
for name, v in V1_LAST.items():
    final = cds.get(day, np.nan) if name == "CDS" else yahoo.loc[day, name]
    rows.append({"series": name, "v1_value": v, "final_close": float(final),
                 "difference": float(v - final) if np.isfinite(final) else np.nan})
save_table(rows, "table2_v1_last_day")

# Table 3: version 1 reproduced
PRINTED = {"N": 1507, "R2": 0.4477, "beta_BIST100": 1.049422, "p_USDTRY_L1": 0.03335422,
           "ARX_RMSE": 2.4903, "Zero_RMSE": 2.7440, "AR_RMSE": 2.7099, "Lasso_RMSE": 2.4746, "RF_RMSE": 2.4982,
           "XGB_RMSE": 2.6994, "ARX_DirAcc": 64.8649, "OOS_days": 259, "DM_ARX_Zero_p": 0.0424}
ns = run_v1(yahoo, cds)
hr = ns["horse_race"]
e_arx = ns["yhat_arx"] - ns["y_true"]
e_zero = ns["yhat_zero"] - ns["y_true"]
dm_p = ns["diebold_mariano"](e_arx.values, (ns["yhat_zero"].reindex(ns["y_true"].index) - ns["y_true"]).values)[1]
rerun = {"N": int(ns["res"].nobs), "R2": float(ns["res"].rsquared), "beta_BIST100": float(ns["res"].params["r_BIST100_L0"]),
         "p_USDTRY_L1": float(ns["res"].pvalues["r_USDTRY_L1"]), "ARX_RMSE": float(hr.loc["ARX", "RMSE"]),
         "Zero_RMSE": float(hr.loc["Zero", "RMSE"]), "AR_RMSE": float(hr.loc["AR-only", "RMSE"]),
         "Lasso_RMSE": float(hr.loc["Lasso", "RMSE"]), "RF_RMSE": float(hr.loc["RandomForest", "RMSE"]),
         "XGB_RMSE": float(hr.loc["XGBoost", "RMSE"]), "ARX_DirAcc": float(hr.loc["ARX", "DirAcc(%)"]),
         "OOS_days": int(len(ns["oos"])), "DM_ARX_Zero_p": float(dm_p)}
save_table([{"quantity": k, "v1_printed": PRINTED[k], "v1_code_on_frozen_data": rerun[k]} for k in PRINTED],
           "table3_v1_reproduced")
for k in PRINTED:
    print(f"{k:14s} printed {PRINTED[k]:>10} rerun {rerun[k]:.6g}")

# Figure 1
cum = trans[["r_ASELS", "r_BIST100"]].fillna(0).cumsum()
window = 120
beta = {}
xy = trans[["r_ASELS", "r_BIST100"]].dropna()
for i in range(window, len(xy)):                      # as in version 1: window of the previous 120 days
    w = xy.iloc[i - window:i]
    beta[xy.index[i]] = float(sm.OLS(w["r_ASELS"], sm.add_constant(w["r_BIST100"])).fit().params["r_BIST100"])
beta = pd.Series(beta)
save_table([{"first_day": str(trans.index[1].date()), "last_day": str(trans.index[-1].date()),
             "days": int(trans["r_ASELS"].notna().sum()), "cum_log_ASELS": float(cum["r_ASELS"].iloc[-1]),
             "cum_log_BIST100": float(cum["r_BIST100"].iloc[-1]), "beta_min": float(beta.min()),
             "beta_max": float(beta.max()), "beta_mean": float(beta.mean()),
             "sd_ASELS": float(trans["r_ASELS"].std()), "sd_BIST100": float(trans["r_BIST100"].std()),
             "corr": float(xy.corr().iloc[0, 1])}], "table4_returns")

use_style()


def draw():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.3), constrained_layout=True)
    ax = axes[0]
    ax.plot(cum.index, cum["r_ASELS"], color=BLUE, lw=1.1)
    ax.plot(cum.index, cum["r_BIST100"], color=INK, lw=1.1)
    end_label(ax, cum.index[-1], cum["r_ASELS"].iloc[-1], "ASELSAN", BLUE)
    end_label(ax, cum.index[-1], cum["r_BIST100"].iloc[-1], "BIST100", INK)
    auto_axis(ax, "y", np.append(cum[["r_ASELS", "r_BIST100"]].to_numpy().ravel(), 0.0))
    ax.set_title("(a) Cumulative log return, %")
    ax = axes[1]
    ax.plot(beta.index, beta.values, color=BLUE, lw=1.0)
    may_touch(ax.axhline(1.0, color=RED, lw=0.8, ls=(0, (1, 1.4))))
    auto_axis(ax, "y", beta.values)
    ax.set_title(f"(b) Rolling market beta, {window} days")
    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        free_axis(a, "x")
    return fig


save_figure("fig1_data", draw, lambda: "Figure 1. ASELSAN and BIST100, 2020-2026")
