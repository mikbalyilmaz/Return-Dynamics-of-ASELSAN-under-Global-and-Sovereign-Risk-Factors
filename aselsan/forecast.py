"""Recursive (expanding-window) one-day-ahead forecasts and their evaluation.

Design: for every test day t, each model is fitted on all rows of the design dated before t and then applied to
the row of day t. In the forecasting design (first_lag = 1) that row contains only values known at the close of
day t-1, so the forecast is a genuine one-day-ahead forecast. Every model is refitted every day.

Evaluation (e_t = forecast - realized):
  RMSE, MAE;
  out-of-sample R^2 against the prevailing (historical) mean, R2_OS = 1 - sum e_model^2 / sum e_mean^2
    (Campbell and Thompson, 2008);
  directional accuracy on days with a non-zero forecast and return, and the Pesaran-Timmermann (1992) test of
    independence between predicted and realized signs;
  Diebold-Mariano (1995) test of equal squared-error loss with the Harvey-Leybourne-Newbold (1997) correction;
  Clark-West (2007) test for nested models (does the larger model improve on the smaller one?).

Author: Muhammed Ikbal Yilmaz
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 4, 50)          # the penalty grid of version 1


def _ols(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta


def make_model(name, seed=42):
    """Model objects with fit(X, y) / predict(X). Hyperparameters of the tree models are those of version 1;
    the penalties of Ridge and Lasso are chosen by 5-fold time-series cross-validation (each fold validated
    only on later data)."""
    if name == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(5)))])
    if name == "Lasso":
        return Pipeline([("scale", StandardScaler()),
                         ("model", LassoCV(alphas=ALPHAS, cv=TimeSeriesSplit(5), max_iter=20_000))])
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=10, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=4)
    raise ValueError(name)


def recursive(y, X, test_start, models, ar_cols=None):
    """One-day-ahead forecasts for every test day. `models` is a list of names among: "Zero", "Prevailing mean",
    "Random walk", "AR", "ARX", "Ridge", "Lasso", "Random forest", "XGBoost". X includes the constant column
    "const"; the machine-learning models use X without it. Returns a DataFrame: realized and one column per model."""
    ar_cols = ar_cols or [c for c in X.columns if c.startswith("r_ASELS_L")]
    test = y.index[y.index >= pd.Timestamp(test_start)]
    out = pd.DataFrame(index=test, dtype=float)
    out["Realized"] = y.loc[test]
    Xv, yv = X.values, y.values
    feat = [i for i, c in enumerate(X.columns) if c != "const"]
    ar_idx = [list(X.columns).index("const")] + [list(X.columns).index(c) for c in ar_cols]
    for name in models:
        preds = []
        for t in test:
            n = int(y.index.searchsorted(t))            # rows 0..n-1 are dated before t
            ytr, xte = yv[:n], Xv[n]
            if name == "Zero":
                f = 0.0
            elif name == "Prevailing mean":
                f = float(ytr.mean())
            elif name == "Random walk":
                f = float(yv[n - 1])
            elif name == "AR":
                f = float(_ols(Xv[:n][:, ar_idx], ytr, xte[ar_idx]))
            elif name == "ARX":
                f = float(_ols(Xv[:n], ytr, xte))
            else:
                m = make_model(name)
                m.fit(Xv[:n][:, feat], ytr)
                f = float(m.predict(xte[feat].reshape(1, -1))[0])
            preds.append(f)
        out[name] = preds
    return out


# ------------------------------------------------------------------------------------------ evaluation
def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e):
    return float(np.mean(np.abs(e)))


def oos_r2(y, f, f_bench):
    return float(1.0 - np.sum((y - f) ** 2) / np.sum((y - f_bench) ** 2))


def direction(y, f):
    """Share of correct signs on days with non-zero forecast and return, and the Pesaran-Timmermann statistic
    and one-sided p-value (H0: predicted and realized signs independent). NaN if the forecast never changes sign."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = (y != 0) & (f != 0)
    up_y, up_f = y[m] > 0, f[m] > 0
    n = int(m.sum())
    if n == 0 or up_f.all() or (~up_f).all():
        return (float(np.mean(up_y == up_f)) if n else np.nan), np.nan, np.nan
    p_hat = np.mean(up_y == up_f)
    py, pf = up_y.mean(), up_f.mean()
    p_star = py * pf + (1 - py) * (1 - pf)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * pf * (1 - pf) / n + (2 * pf - 1) ** 2 * py * (1 - py) / n
              + 4 * py * pf * (1 - py) * (1 - pf) / n ** 2)
    pt = (p_hat - p_star) / np.sqrt(v_hat - v_star)
    return float(p_hat), float(pt), float(1 - stats.norm.cdf(pt))


def diebold_mariano(e1, e2):
    """DM test of equal mean squared error for one-step forecasts, HLN-corrected, compared with t(T-1).
    Negative statistic: model 1 has the smaller loss. Returns (statistic, two-sided p)."""
    d = np.square(np.asarray(e1, float)) - np.square(np.asarray(e2, float))
    T = d.size
    dm = d.mean() / np.sqrt(d.var(ddof=0) / T)
    hln = dm * np.sqrt((T - 1) / T)
    return float(hln), float(2 * stats.t.sf(abs(hln), df=T - 1))


def clark_west(y, f_small, f_large):
    """Clark-West test for nested models: H0 equal predictive accuracy, H1 the larger model is better.
    f_t = (y - f_small)^2 - [(y - f_large)^2 - (f_small - f_large)^2]; t-statistic of its mean, one-sided p."""
    y, a, b = (np.asarray(v, float) for v in (y, f_small, f_large))
    f = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(f.size))
    return float(t), float(stats.norm.sf(t))
