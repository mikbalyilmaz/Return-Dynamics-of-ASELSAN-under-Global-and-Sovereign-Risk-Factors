"""ARX designs and their estimation.

The ARX model of version 1, for y_t = r_t(ASELS) and the transformed factors Z_{k,t}:

    y_t = a + sum_{i=1}^{p} phi_i y_{t-i} + sum_{k=1}^{K} sum_{j=j0}^{q} beta_{k,j} Z_{k,t-j} + e_t .

j0 = 0 is the contemporaneous ("co-movement") model of version 1: it explains y_t with factor values of the same
day, which are not known before day t ends. j0 = 1 is the forecasting model: every regressor is known at the close
of day t-1. OLS with Newey-West (HAC) standard errors, as in version 1.

Author: Muhammed Ikbal Yilmaz
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["r_BIST100", "r_USDTRY", "r_BITCOIN", "r_BRENT", "d_VIX", "d_DXY", "d_US10Y", "d_CDS"]
CHANNELS = {"market (BIST100)": "BIST100", "exchange rate (USD/TRY)": "USDTRY", "global risk (VIX)": "VIX",
            "sovereign risk (CDS)": "CDS", "dollar (DXY)": "DXY", "US 10-year yield": "US10Y",
            "Bitcoin": "BITCOIN", "Brent oil": "BRENT"}


def factor_names(vix="diff", cds="diff"):
    out = list(FACTORS)
    if vix == "level":
        out[out.index("d_VIX")] = "VIX"
    if cds == "level":
        out[out.index("d_CDS")] = "CDS"
    return out


def design(trans, p, q, first_lag=0, factors=None, y_col="r_ASELS"):
    """y and X (with constant) for the ARX model; rows with any missing value dropped. Column names follow
    version 1: r_ASELS_L{i} for the AR terms and {factor}_L{j} for the factors."""
    factors = factor_names() if factors is None else factors
    y = trans[y_col]
    cols = {}
    for i in range(1, p + 1):
        cols[f"{y_col}_L{i}"] = y.shift(i)
    for c in factors:
        for j in range(first_lag, q + 1):
            cols[f"{c}_L{j}"] = trans[c].shift(j)
    X = pd.DataFrame(cols, index=trans.index)
    data = pd.concat([y.rename(y_col), X], axis=1).dropna()
    return data[y_col], sm.add_constant(data.drop(columns=y_col), has_constant="add")


def fit_hac(y, X, hac_lags=5):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})


def joint_test(res, names):
    """Wald test that all listed coefficients are zero (HAC covariance). Returns (chi2, df, p)."""
    names = [n for n in names if n in res.params.index]
    R = np.zeros((len(names), len(res.params)))
    for r, n in enumerate(names):
        R[r, list(res.params.index).index(n)] = 1.0
    w = res.wald_test(R, scalar=True)
    return float(w.statistic), len(names), float(w.pvalue)


def sum_test(res, names):
    """Sum of the listed coefficients (cumulative effect) and the p-value of H0: sum = 0."""
    names = [n for n in names if n in res.params.index]
    R = np.zeros((1, len(res.params)))
    for n in names:
        R[0, list(res.params.index).index(n)] = 1.0
    return float(res.params[names].sum()), float(res.wald_test(R, scalar=True).pvalue)


def channel_columns(X, base):
    """Columns of X that belong to one factor (any lag, differenced or level)."""
    return [c for c in X.columns if c.rsplit("_L", 1)[0] in (f"r_{base}", f"d_{base}", base)]
