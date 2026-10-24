"""Analysis 2: how ASELSAN moves with the factors on the same day (the ARX model of version 1).

Why: version 1's ARX model includes each factor at lag 0, so it describes co-movement: how ASELSAN's return on day t
relates to the market, the exchange rate, global risk and sovereign risk on the same day. That is a legitimate and
interesting question, but a different one from forecasting (analysis 3). Version 1 stated a joint hypothesis for
each channel (all coefficients of the channel zero) but tested coefficients one at a time.

Design (version 1's specification on the panel of version 2): p = q = 2, Newey-West HAC with 5 lags.
* Table 5: coefficients, HAC standard errors and p-values.
* Table 6: for each channel, the joint Wald test of all its coefficients (lags 0..q) and the cumulative effect
  (sum over lags) with its p-value; the same for the forecasting design (lags 1..q only).
* Table 7: version 1's robustness grid: VIX and CDS in levels or differences, p = q in {1, 3, 5}, HAC 5.
* Figure 2: (a) in-sample R^2 of the contemporaneous and the forecasting design for p = q = 1, 2, 3, 5;
  (b) cumulative effect of a one-standard-deviation move of each non-market factor, with 95% HAC intervals.
"""

import numpy as np
from scipy import stats

from common import save_table
from aselsan.data import asof_panel, transform
from aselsan.models import CHANNELS, channel_columns, design, factor_names, fit_hac, joint_test, sum_test
from aselsan.style import BLUE, INK, auto_axis, free_axis, save_figure, use_style, value_axis

levels, _ = asof_panel()
trans = transform(levels)
P = Q = 2
HAC = 5

y, X = design(trans, P, Q, first_lag=0)
res = fit_hac(y, X, HAC)
save_table([{"term": n, "coef": float(res.params[n]), "se": float(res.bse[n]), "t": float(res.tvalues[n]),
             "p": float(res.pvalues[n])} for n in res.params.index]
           + [{"term": "R2", "coef": float(res.rsquared)}, {"term": "N", "coef": float(res.nobs)}],
           "table5_arx_contemporaneous")

y1, X1 = design(trans, P, Q, first_lag=1)
res1 = fit_hac(y1, X1, HAC)
rows = []
for label, base in CHANNELS.items():
    cols0, cols1 = channel_columns(X, base), channel_columns(X1, base)
    chi0, df0, p0 = joint_test(res, cols0)
    s0, ps0 = sum_test(res, cols0)
    chi1, df1, p1 = joint_test(res1, cols1)
    s1, ps1 = sum_test(res1, cols1)
    rows.append({"channel": label, "joint_chi2_same_day": chi0, "df_same_day": df0, "joint_p_same_day": p0,
                 "sum_same_day": s0, "sum_p_same_day": ps0, "joint_chi2_lagged": chi1, "df_lagged": df1,
                 "joint_p_lagged": p1, "sum_lagged": s1, "sum_p_lagged": ps1})
save_table(rows, "table6_channels")
for r in rows:
    print(f"{r['channel']:26s} same-day p {r['joint_p_same_day']:.4f}   lagged p {r['joint_p_lagged']:.4f}")

# Table 7: robustness grid of version 1
grid = []
for vix in ("diff", "level"):
    for cds in ("diff", "level"):
        f = factor_names(vix, cds)
        tr = transform(levels, vix=vix, cds=cds)
        for p in (1, 3, 5):
            yy, XX = design(tr, p, p, first_lag=0, factors=f)
            r = fit_hac(yy, XX, HAC)
            vname, cname = f[f.index("d_VIX" if vix == "diff" else "VIX")], f[f.index("d_CDS" if cds == "diff" else "CDS")]
            row = {"VIX": vix, "CDS": cds, "p=q": p, "N": int(r.nobs), "R2": float(r.rsquared),
                   "adjR2": float(r.rsquared_adj)}
            for n in ("r_BIST100_L0", "r_USDTRY_L0", "r_USDTRY_L1", f"{vname}_L0", f"{cname}_L0"):
                key = n.replace(vname, "VIX").replace(cname, "CDS")
                row[f"{key}_coef"], row[f"{key}_p"] = float(r.params[n]), float(r.pvalues[n])
            for key, name in (("VIX", vname), ("CDS", cname)):
                row[f"cum_{key}"], row[f"cum_{key}_p"] = sum_test(r, [f"{name}_L{j}" for j in range(p + 1)])
            grid.append(row)
save_table(grid, "table7_robustness")

# Figure 2 inputs
r2 = []
for p in (1, 2, 3, 5):
    a = fit_hac(*design(trans, p, p, first_lag=0), HAC)
    b = fit_hac(*design(trans, p, p, first_lag=1), HAC)
    r2.append({"p=q": p, "R2_same_day": float(a.rsquared), "R2_lagged": float(b.rsquared),
               "adjR2_same_day": float(a.rsquared_adj), "adjR2_lagged": float(b.rsquared_adj)})
save_table(r2, "table8_r2_by_lag")

sd = trans.std()
eff = []
V = res.cov_params()
for label, base in CHANNELS.items():
    if base == "BIST100":
        continue
    cols = channel_columns(X, base)
    fac = cols[0].rsplit("_L", 1)[0]
    s = float(res.params[cols].sum())
    se = float(np.sqrt(V.loc[cols, cols].values.sum()))
    eff.append({"channel": label, "sd_factor": float(sd[fac]), "effect_per_sd": s * sd[fac],
                "lo95": (s - 1.959964 * se) * sd[fac], "hi95": (s + 1.959964 * se) * sd[fac],
                "p": float(2 * stats.norm.sf(abs(s / se)))})
save_table(eff, "table9_effects_per_sd")

use_style()


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    ps = [r["p=q"] for r in r2]
    same = [r["R2_same_day"] for r in r2]
    lag = [r["R2_lagged"] for r in r2]
    ax.plot(ps, same, color=BLUE, marker="o", ms=3, lw=1.2)
    ax.plot(ps, lag, color=INK, marker="s", ms=3, lw=1.2)
    ax.annotate("with same-day factors", xy=(ps[-1], same[-1]), xytext=(0, -6), textcoords="offset points",
                color=BLUE, fontweight="bold", fontsize=7, va="top", ha="right")
    ax.annotate("lagged factors only", xy=(ps[-1], lag[-1]), xytext=(0, 6), textcoords="offset points",
                color=INK, fontweight="bold", fontsize=7, va="bottom", ha="right")
    value_axis(ax, "x", 1, 5, 1)
    value_axis(ax, "y", 0, 0.5, 0.1)
    ax.set_xlabel("$p = q$")
    ax.set_title("(a) In-sample $R^2$")
    ax = axes[1]
    yy = np.arange(len(eff))[::-1]
    for yi, e in zip(yy, eff):
        ax.plot([e["lo95"], e["hi95"]], [yi, yi], color=BLUE, lw=1.4)
        ax.plot(e["effect_per_sd"], yi, "o", color=BLUE, ms=3)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([e["channel"] for e in eff], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(yy.min() - 0.6, yy.max() + 0.6)
    free_axis(ax, "y")
    auto_axis(ax, "x", [v for e in eff for v in (e["lo95"], e["hi95"])])
    ax.set_xlabel("pp, effect of a 1 s.d. move, lags 0-2")
    ax.set_title("(b) Same-day model: cumulative effects")
    return fig


save_figure("fig2_comovement", draw, lambda: "Figure 2. Co-movement: same-day versus lagged factors")
