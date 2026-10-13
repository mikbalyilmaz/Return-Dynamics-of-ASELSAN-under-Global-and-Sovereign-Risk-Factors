"""Unit tests. Run from the repository root:  python tests/run_tests.py

Each result is compared with an independent computation: a direct loop over the raw files, a Newey-West covariance
written out by hand, OLS refitted with statsmodels on the training rows, perturbation of future data (no
look-ahead), simulated series with a known answer, and the printed output of version 1.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aselsan.data import SERIES, asof_panel, load_cds, load_yahoo, transform  # noqa: E402
from aselsan.forecast import clark_west, diebold_mariano, direction, oos_r2, recursive  # noqa: E402
from aselsan.models import design, fit_hac, joint_test  # noqa: E402

warnings.simplefilter("ignore")
failed = 0


def check(name, ok):
    global failed
    print(f"{'ok' if ok else 'FAIL':<5}{name}")
    if not ok:
        failed += 1


yahoo, cds = load_yahoo(), load_cds()
levels, age = asof_panel(yahoo, cds)
trans = transform(levels)

# 1. Panel: ASELSAN days, as-of values found by a direct search, nothing from the future, end at the CDS end
ok = levels.notna().all().all() and levels.index[-1] == pd.Timestamp("2026-01-08")
ok &= levels.index.equals(yahoo["ASELS"].dropna().index[:len(levels)])
rng = np.random.default_rng(0)
for day in rng.choice(levels.index, 60, replace=False):
    for name in SERIES:
        s = (cds if name == "CDS" else yahoo[name]).dropna()
        before = s[s.index <= day]
        ok &= before.iloc[-1] == levels.loc[day, name] and (day - before.index[-1]).days == age.loc[day, name]
check("panel: one row per ASELSAN day, each series as of its last close on or before the day (60 days checked)", ok)

# 2. Transformations
ok = np.allclose(trans["r_ASELS"].iloc[1:], 100 * np.diff(np.log(levels["ASELS"].values)))
ok &= np.allclose(trans["d_CDS"].iloc[1:], np.diff(levels["CDS"].values))
lv = transform(levels, vix="level", cds="level")
ok &= np.array_equal(lv["VIX"].values, levels["VIX"].values) and "d_VIX" not in lv
check("returns are 100 x log differences, differences between consecutive ASELSAN days, levels when asked", ok)

# 3. No look-ahead in the forecasting design: changing the data of day t and later leaves earlier rows unchanged,
#    and the row of day t unchanged when first_lag = 1
t = trans.index[800]
pert = trans.copy()
pert.loc[t:] = pert.loc[t:] * 3.0 + 1.0
y1, X1 = design(trans, 2, 2, first_lag=1)
y2, X2 = design(pert, 2, 2, first_lag=1)
ok = X1.loc[:t].equals(X2.loc[:t]) and y1.loc[:t].iloc[:-1].equals(y2.loc[:t].iloc[:-1])
_, X0a = design(trans, 2, 2, first_lag=0)
_, X0b = design(pert, 2, 2, first_lag=0)
ok &= not X0a.loc[t].equals(X0b.loc[t])            # the same-day design does use day t
ok &= not any(c.endswith("_L0") for c in X1.columns) and "r_BIST100_L0" in X0a.columns
check("forecasting design uses only data before day t; the same-day design of version 1 uses day t", ok)

# 4. HAC covariance written out: Newey-West with Bartlett weights, no small-sample correction
y, X = design(trans, 2, 2, first_lag=0)
res = fit_hac(y, X, 5)
Xa, u = X.values, (y - X @ res.params).values
S = (Xa * u[:, None]).T @ (Xa * u[:, None])
for L in range(1, 6):
    G = (Xa[L:] * u[L:, None]).T @ (Xa[:-L] * u[:-L, None])
    S += (1 - L / 6) * (G + G.T)
B = np.linalg.inv(Xa.T @ Xa)
V = B @ S @ B
check("Newey-West covariance (5 lags) equals the formula written out", np.allclose(V, res.cov_params().values, rtol=1e-9))

# 5. Joint Wald test equals beta' R' (R V R')^{-1} R beta
names = ["r_USDTRY_L0", "r_USDTRY_L1", "r_USDTRY_L2"]
idx = [list(X.columns).index(n) for n in names]
b = res.params.values[idx]
W = float(b @ np.linalg.solve(V[np.ix_(idx, idx)], b))
check("joint Wald statistic equals its quadratic form", abs(joint_test(res, names)[0] - W) < 1e-8 * max(1, W))

# 6. Recursive forecasts: ARX at day t equals statsmodels OLS on all earlier rows; zero, mean and random walk
y1, X1 = design(trans, 2, 2, first_lag=1)
fc = recursive(y1, X1, "2025-06-02", ["Zero", "Prevailing mean", "Random walk", "AR", "ARX"])
ok = True
for t in fc.index[:5].tolist() + fc.index[-3:].tolist():
    tr = y1.index < t
    ok &= abs(sm.OLS(y1[tr], X1[tr]).fit().predict(X1.loc[[t]]).iloc[0] - fc.loc[t, "ARX"]) < 1e-9
    arc = ["const", "r_ASELS_L1", "r_ASELS_L2"]
    ok &= abs(sm.OLS(y1[tr], X1.loc[tr, arc]).fit().predict(X1.loc[[t], arc]).iloc[0] - fc.loc[t, "AR"]) < 1e-9
    ok &= abs(fc.loc[t, "Prevailing mean"] - y1[tr].mean()) < 1e-12 and fc.loc[t, "Random walk"] == y1[tr].iloc[-1]
ok &= (fc["Zero"] == 0).all()
pert_y = y1.copy()
pert_y.loc[fc.index[3]:] += 50.0                    # change realized returns from the 4th test day on
fc2 = recursive(pert_y, X1, "2025-06-02", ["ARX"])
ok &= np.allclose(fc2["ARX"].iloc[:4], fc["ARX"].iloc[:4])
check("recursive forecasts equal OLS on earlier rows; a forecast does not change when later returns change", ok)

# 7. Evaluation statistics against direct computation and known cases
r = np.random.default_rng(3)
yy = r.standard_normal(300)
f1, f2 = 0.3 * r.standard_normal(300), 0.1 * r.standard_normal(300)
d = (f1 - yy) ** 2 - (f2 - yy) ** 2
dm_direct = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / 300) * np.sqrt(299 / 300)
cw_f = (yy - f2) ** 2 - ((yy - f1) ** 2 - (f2 - f1) ** 2)
ok = abs(diebold_mariano(f1 - yy, f2 - yy)[0] - dm_direct) < 1e-12
ok &= abs(clark_west(yy, f2, f1)[0] - cw_f.mean() / (cw_f.std(ddof=1) / np.sqrt(300))) < 1e-12
ok &= oos_r2(yy, yy, f1) == 1.0 and oos_r2(yy, f1, f1) == 0.0
ok &= direction(yy, yy)[0] == 1.0 and np.isnan(direction(yy, np.zeros(300))[0])
check("DM (HLN), Clark-West, R2_OS and directional accuracy equal direct computation", ok)

# 8. Known answers on simulated data: a real lagged predictor is found, an unrelated one is not
n = 1500
x = r.standard_normal(n)
idx = pd.bdate_range("2019-01-01", periods=n)
sim = pd.DataFrame({"r_ASELS": np.r_[0, 0.4 * x[:-1]] + r.standard_normal(n), "x": x}, index=idx)
ys, Xs = design(sim, 1, 1, first_lag=1, factors=["x"])
f = recursive(ys, Xs, idx[1200], ["Prevailing mean", "ARX"])
good = oos_r2(f["Realized"], f["ARX"], f["Prevailing mean"]) > 0.05 and clark_west(f["Realized"], f["Prevailing mean"], f["ARX"])[1] < 0.01
sim["r_ASELS"] = r.standard_normal(n)
ys, Xs = design(sim, 1, 1, first_lag=1, factors=["x"])
f = recursive(ys, Xs, idx[1200], ["Prevailing mean", "ARX"])
check("simulation: a true lagged predictor gives R2_OS > 0.05 (CW p < 0.01); noise gives R2_OS <= 0.01",
      good and oos_r2(f["Realized"], f["ARX"], f["Prevailing mean"]) <= 0.01)

# 9. Version 1's code on the frozen data reproduces its printed results
from aselsan.legacy import run_v1  # noqa: E402
ns = run_v1(yahoo, cds)
hr = ns["horse_race"]
ok = abs(ns["res"].rsquared - 0.4477) < 0.001 and abs(ns["res"].params["r_BIST100_L0"] - 1.049422) < 0.001
ok &= abs(hr.loc["ARX", "RMSE"] - 2.4903) < 0.01 and abs(hr.loc["Lasso", "RMSE"] - 2.4746) < 0.01
ok &= abs(hr.loc["Zero", "RMSE"] - 2.7440) < 0.01 and abs(hr.loc["AR-only", "RMSE"] - 2.7099) < 0.01
check("version 1's own code on the frozen data: R2, market beta and OOS RMSEs within 0.01 of its printout", ok)

print(f"\n{failed} check(s) failed.")
sys.exit(1 if failed else 0)
