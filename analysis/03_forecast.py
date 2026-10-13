"""Analysis 3: can ASELSAN's return be forecast one day ahead?

Why: version 1's "out-of-sample forecasts" used the factors of the forecast day itself (lag 0): the BIST100 return,
the exchange rate, VIX and CDS of day t, which are known only after day t. Its machine-learning models were given
every column of the data table, again including same-day values. Their accuracy measures how well ASELSAN's return
can be explained once the day is over, not how well it can be forecast.

Design: training sample up to 31 December 2024, test sample from 2 January 2025 to the end of the panel; every
model refitted every day on all earlier days (expanding window).
* Forecasting design: AR and factor lags 1..q only (p = q = 2 as in version 1), so every input is known at the
  close of day t-1. Models: zero, prevailing mean, random walk, AR(2), ARX, Ridge, Lasso, random forest, XGBoost
  (the machine-learning models get exactly the ARX inputs).
* For comparison, the same-day design of version 1 (lags 0..q) with ARX, and a model with the same-day BIST100
  return alone.
* Table 10: RMSE, MAE, R^2_OS against the prevailing mean, directional accuracy with the Pesaran-Timmermann test,
  Diebold-Mariano (HLN) against the prevailing mean, Clark-West for the nested linear models.
* Table 11: R^2_OS of the forecasting ARX for p = q = 1, 2, 3, 5.
* Figure 3: cumulative difference of squared errors, prevailing mean minus model (rising: the model is better).
"""

import numpy as np
import pandas as pd

from common import save_table
from aselsan.data import asof_panel, transform
from aselsan.forecast import clark_west, diebold_mariano, direction, mae, oos_r2, recursive, rmse
from aselsan.models import design
from aselsan.style import AQUA, BLUE, GOLD, INK, MUTED, ORANGE, PURPLE, decimal_axis, save_figure, use_style

TEST_START = "2025-01-01"
P = Q = 2
levels, _ = asof_panel()
trans = transform(levels)

MODELS = ["Zero", "Prevailing mean", "Random walk", "AR", "ARX", "Ridge", "Lasso", "Random forest", "XGBoost"]
y, X = design(trans, P, Q, first_lag=1)
fc = recursive(y, X, TEST_START, MODELS)

y0, X0 = design(trans, P, Q, first_lag=0)
same = recursive(y0, X0, TEST_START, ["ARX"]).rename(columns={"ARX": "ARX, same-day factors"})
yb, Xb = design(trans, 0, 0, first_lag=0, factors=["r_BIST100"])
same["BIST100 same day only"] = recursive(yb, Xb, TEST_START, ["ARX"])["ARX"]
assert same.index.equals(fc.index) and np.allclose(same["Realized"], fc["Realized"])
all_fc = pd.concat([fc, same.drop(columns="Realized")], axis=1)
all_fc.to_csv("../figures/forecasts.csv", float_format="%.10g")

yt, bench = all_fc["Realized"], all_fc["Prevailing mean"]
NESTED = {"AR": "Prevailing mean", "ARX": "Prevailing mean"}
rows = []
for name in all_fc.columns[1:]:
    f = all_fc[name]
    e = f - yt
    da, pt, pt_p = direction(yt, f)
    row = {"model": name, "design": "same day (version 1)" if name in same.columns else "lagged (t-1)",
           "RMSE": rmse(e), "MAE": mae(e), "R2_OS": oos_r2(yt, f, bench), "dir_acc": da, "PT_p": pt_p}
    if name != "Prevailing mean":
        row["DM"], row["DM_p"] = diebold_mariano(e, bench - yt)
    if name in NESTED:
        row["CW"], row["CW_p"] = clark_west(yt, bench, f)
    if name == "ARX":
        row["DM_vs_AR"], row["DM_vs_AR_p"] = diebold_mariano(e, all_fc["AR"] - yt)
        row["CW_vs_AR"], row["CW_vs_AR_p"] = clark_west(yt, all_fc["AR"], f)
    rows.append(row)
save_table(rows, "table10_forecasts")
for r in rows:
    print(f"{r['model']:24s} RMSE {r['RMSE']:.4f}  R2os {r['R2_OS']:+.4f}  DA {r['dir_acc']:.3f}  DM_p {r.get('DM_p', np.nan):.3f}")

lag_rows = []
for p in (1, 2, 3, 5):
    yy, XX = design(trans, p, p, first_lag=1)
    f = recursive(yy, XX, TEST_START, ["Prevailing mean", "ARX"])
    lag_rows.append({"p=q": p, "R2_OS": oos_r2(f["Realized"], f["ARX"], f["Prevailing mean"]),
                     "CW_p": clark_west(f["Realized"], f["Prevailing mean"], f["ARX"])[1],
                     "RMSE": rmse(f["ARX"] - f["Realized"])})
save_table(lag_rows, "table11_lag_order")
save_table([{"test_first": str(yt.index[0].date()), "test_last": str(yt.index[-1].date()), "test_days": len(yt),
             "train_days_first_forecast": int((y.index < yt.index[0]).sum()),
             "sd_realized": float(yt.std()), "mean_realized": float(yt.mean())}], "table12_test_sample")

use_style()
SHOW = [("ARX", BLUE, "-"), ("AR", PURPLE, "-"), ("Lasso", AQUA, "-"), ("Random forest", GOLD, "-"),
        ("XGBoost", ORANGE, "-")]


def draw():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    base = (bench - yt) ** 2
    ax = axes[0]
    for name, col, ls in SHOW:
        ax.plot(yt.index, (base - (all_fc[name] - yt) ** 2).cumsum(), color=col, lw=1.1, ls=ls, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("(a) Forecasts with lagged information")
    ax.set_ylabel("Cumulative SSE(mean) $-$ SSE(model)")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    ax = axes[1]
    for name, col, ls in (("ARX, same-day factors", BLUE, "--"), ("BIST100 same day only", INK, ":"),
                          ("ARX", BLUE, "-")):
        ax.plot(yt.index, (base - (all_fc[name] - yt) ** 2).cumsum(), color=col, lw=1.1, ls=ls,
                label=name if name != "ARX" else "ARX, lagged factors")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("(b) Version 1's design: same-day factors")
    ax.legend(loc="upper left", fontsize=7.5)
    for a in axes:
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.tick_params(axis="x", labelsize=7.5)
        decimal_axis(a)
    return fig


save_figure("fig3_forecasts", draw,
            lambda: "Figure 3. Out-of-sample accuracy relative to the prevailing mean, 2025-2026")
