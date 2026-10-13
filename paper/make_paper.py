"""Writes every number and table of the project note from ../figures/*.csv, so that the text cannot drift away
from the analysis. Run from inside paper/:  python make_paper.py

Output: numbers.tex (one macro per quoted number), tab_reproduced.tex, tab_channels.tex, tab_forecasts.tex
(table bodies).
"""

import csv
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parent


def read(name):
    with open(FIG / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def by(rows, key):
    return {r[key]: r for r in rows}


def dec(x, d):
    s = f"{float(x):.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def pval(x):
    x = float(x)
    return "$<$0.0001" if x < 0.0001 else f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


data = by(read("table1_data"), "series")
last = by(read("table2_v1_last_day"), "series")
rep = by(read("table3_v1_reproduced"), "quantity")
ret = read("table4_returns")[0]
arx = by(read("table5_arx_contemporaneous"), "term")
ch = by(read("table6_channels"), "channel")
rob = read("table7_robustness")
r2 = by(read("table8_r2_by_lag"), "p=q")
eff = by(read("table9_effects_per_sd"), "channel")
fc = by(read("table10_forecasts"), "model")
lag = by(read("table11_lag_order"), "p=q")
ts = read("table12_test_sample")[0]

# raw forecasts: facts about the signs of the forecasts
with open(FIG / "forecasts.csv", newline="") as f:
    rows = list(csv.DictReader(f))
real = [float(r["Realized"]) for r in rows]
pm = [float(r["Prevailing mean"]) for r in rows]

m = {}
# Data
m["PanelDays"] = f"{int(ret['days']) + 1:,}"
m["ReturnDays"] = f"{int(ret['days']):,}"
m["FirstDay"] = ret["first_day"]
m["LastDay"] = ret["last_day"]
for k, key in (("BRENT", "Brent"), ("VIX", "Vix"), ("DXY", "Dxy"), ("US10Y", "Usten"), ("CDS", "Cds"),
               ("BIST100", "Bist"), ("USDTRY", "Usdtry")):
    m[f"Carried{key}"] = data[k]["carried_days"]
    m[f"CarriedPct{key}"] = pct(data[k]["carried_share"])
m["CdsLast"] = data["CDS"]["last"]
m["YahooLast"] = data["ASELS"]["last"]
m["CumAsels"] = dec(ret["cum_log_ASELS"], 0)
m["CumBist"] = dec(ret["cum_log_BIST100"], 0)
m["SdAsels"] = dec(ret["sd_ASELS"], 2)
m["SdBist"] = dec(ret["sd_BIST100"], 2)
m["Corr"] = dec(ret["corr"], 2)
m["BetaMin"] = dec(ret["beta_min"], 2)
m["BetaMax"] = dec(ret["beta_max"], 2)
m["BrentLastVone"] = dec(last["BRENT"]["v1_value"], 2)
m["BrentLastFinal"] = dec(last["BRENT"]["final_close"], 2)
m["VixLastVone"] = dec(last["VIX"]["v1_value"], 2)
m["VixLastFinal"] = dec(last["VIX"]["final_close"], 2)

# Analysis 2
m["Rsq"] = dec(arx["R2"]["coef"], 3)
m["Nobs"] = f"{int(float(arx['N']['coef'])):,}"
m["BetaBist"] = dec(arx["r_BIST100_L0"]["coef"], 3)
m["BetaBistSe"] = dec(arx["r_BIST100_L0"]["se"], 3)
m["UsdLoneP"] = dec(arx["r_USDTRY_L1"]["p"], 3)
m["UsdLone"] = dec(arx["r_USDTRY_L1"]["coef"], 3)
m["CdsLoneP"] = dec(arx["d_CDS_L1"]["p"], 3)
for p, key in (("1", "A"), ("2", "B"), ("3", "C"), ("5", "D")):
    m[f"RsqSame{key}"] = dec(r2[p]["R2_same_day"], 3)
    m[f"RsqLag{key}"] = dec(r2[p]["R2_lagged"], 3)
    m[f"AdjRsqLag{key}"] = dec(r2[p]["adjR2_lagged"], 3)
VIX, CDS, USD, BRT = "global risk (VIX)", "sovereign risk (CDS)", "exchange rate (USD/TRY)", "Brent oil"
m["UsdJointP"] = dec(ch[USD]["joint_p_same_day"], 3)
m["UsdJointLagP"] = dec(ch[USD]["joint_p_lagged"], 4)
m["VixJointP"] = dec(ch[VIX]["joint_p_same_day"], 3)
m["VixSumP"] = dec(ch[VIX]["sum_p_same_day"], 3)
m["CdsJointP"] = dec(ch[CDS]["joint_p_same_day"], 3)
m["CdsSumP"] = dec(ch[CDS]["sum_p_same_day"], 3)
m["BrentJointP"] = dec(ch[BRT]["joint_p_same_day"], 3)
m["BrentSumP"] = dec(ch[BRT]["sum_p_same_day"], 3)
others_lag = [float(r["joint_p_lagged"]) for k, r in ch.items() if k != USD]
m["MinOtherLagP"] = dec(min(others_lag), 2)
for c, key in ((VIX, "Vix"), (CDS, "Cds"), (BRT, "Brent"), (USD, "Usd")):
    m[f"Eff{key}"] = dec(eff[c]["effect_per_sd"], 2)
    m[f"EffLo{key}"] = dec(eff[c]["lo95"], 2)
    m[f"EffHi{key}"] = dec(eff[c]["hi95"], 2)
    m[f"Sd{key}"] = dec(eff[c]["sd_factor"], 2)
m["SdUsten"] = dec(eff["US 10-year yield"]["sd_factor"], 3)
rob_dd = [r for r in rob if r["VIX"] == "diff" and r["CDS"] == "diff"]
rob_cl = [r for r in rob if r["CDS"] == "level"]
m["RobBetaMin"] = dec(min(float(r["r_BIST100_L0_coef"]) for r in rob), 3)
m["RobBetaMax"] = dec(max(float(r["r_BIST100_L0_coef"]) for r in rob), 3)
m["RobCumCdsPmin"] = dec(min(float(r["cum_CDS_p"]) for r in rob_dd), 3)
m["RobCumCdsPmax"] = dec(max(float(r["cum_CDS_p"]) for r in rob_dd), 3)
m["RobCdsLevelPmin"] = dec(min(float(r["CDS_L0_p"]) for r in rob_cl), 3)
m["RobCdsLevelPmax"] = dec(max(float(r["CDS_L0_p"]) for r in rob_cl), 3)
m["RobCumCdsLevelPmax"] = dec(max(float(r["cum_CDS_p"]) for r in rob_cl), 3)
m["RobCumCdsLevelPmin"] = dec(min(float(r["cum_CDS_p"]) for r in rob_cl), 3)

# Analysis 3
m["TestFirst"] = ts["test_first"]
m["TestLast"] = ts["test_last"]
m["TestDays"] = ts["test_days"]
m["TrainDays"] = f"{int(ts['train_days_first_forecast']):,}"
m["TestSd"] = dec(ts["sd_realized"], 2)
m["TestMean"] = dec(ts["mean_realized"], 2)
m["PmMin"] = dec(min(pm), 2)
m["PmMax"] = dec(max(pm), 2)
nz = [y for y in real if y != 0]
m["UpShare"] = pct(sum(y > 0 for y in nz) / len(nz))
m["NonzeroDays"] = str(len(nz))
for name, key in (("ARX", "Arx"), ("AR", "Ar"), ("Lasso", "Lasso"), ("Ridge", "Ridge"), ("Random forest", "Rf"),
                  ("XGBoost", "Xgb"), ("Zero", "Zero"), ("Random walk", "Rw"), ("Prevailing mean", "Pm"),
                  ("ARX, same-day factors", "Same"), ("BIST100 same day only", "Bist")):
    r = fc[name]
    m[f"Rmse{key}"] = dec(r["RMSE"], 3)
    m[f"Rtwo{key}"] = dec(r["R2_OS"], 3)
    m[f"Da{key}"] = pct(r["dir_acc"]) if r["dir_acc"] != "nan" else "--"
    if r.get("DM_p"):
        m[f"DmP{key}"] = pval(r["DM_p"])
    if r.get("PT_p") and r["PT_p"] != "nan":
        m[f"PtP{key}"] = pval(r["PT_p"])
m["CwPArx"] = dec(fc["ARX"]["CW_p"], 3)
m["CwPAr"] = dec(fc["AR"]["CW_p"], 3)
m["DmArxArP"] = dec(fc["ARX"]["DM_vs_AR_p"], 3)
m["CwArxArP"] = dec(fc["ARX"]["CW_vs_AR_p"], 3)
for p, key in (("1", "A"), ("2", "B"), ("3", "C"), ("5", "D")):
    m[f"LagRtwo{key}"] = dec(lag[p]["R2_OS"], 3)
    m[f"LagCwP{key}"] = dec(lag[p]["CW_p"], 3)
m["RmseVoneArx"] = dec(rep["ARX_RMSE"]["v1_printed"], 2)
m["DaVoneArx"] = dec(rep["ARX_DirAcc"]["v1_printed"], 1)
m["DmVoneP"] = dec(rep["DM_ARX_Zero_p"]["v1_printed"], 3)
m["XgbDiff"] = dec(abs(float(rep["XGB_RMSE"]["v1_printed"]) - float(rep["XGB_RMSE"]["v1_code_on_frozen_data"])), 2)
others = [k for k in rep if k.endswith("_RMSE") and k != "XGB_RMSE"]
assert max(abs(float(rep[k]["v1_printed"]) - float(rep[k]["v1_code_on_frozen_data"])) for k in others) < 0.01
LAGGED = ["Random walk", "AR", "ARX", "Ridge", "Lasso", "Random forest", "XGBoost"]
m["PtMin"] = dec(min(float(fc[k]["PT_p"]) for k in LAGGED), 2)
pos = {k: sum(float(r[k]) > 0 for r in rows) / len(rows) for k in ("AR", "Ridge", "Lasso")}
m["PosMin"] = pct(min(pos.values()), 0)
up = sum(y > 0 for y in nz) / len(nz)
assert all(abs(float(fc[k]["dir_acc"]) - up) < 1e-12 for k in ("AR", "Ridge", "Lasso"))
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8")

# Table: version 1 reproduced
LABEL = {"N": ("Observations in the ARX model", 0), "R2": ("$R^2$ of the ARX model", 4),
         "beta_BIST100": ("Coefficient of $r^{\\mathrm{BIST}}_t$", 4), "p_USDTRY_L1": ("$p$-value of $r^{\\mathrm{USD}}_{t-1}$", 4),
         "ARX_RMSE": ("RMSE, ARX", 4), "Zero_RMSE": ("RMSE, zero forecast", 4), "AR_RMSE": ("RMSE, AR", 4),
         "Lasso_RMSE": ("RMSE, Lasso", 4), "RF_RMSE": ("RMSE, random forest", 4), "XGB_RMSE": ("RMSE, XGBoost", 4),
         "ARX_DirAcc": ("Directional accuracy of ARX (\\%)", 2), "OOS_days": ("Test days", 0),
         "DM_ARX_Zero_p": ("DM $p$-value, ARX against zero", 4)}
lines = []
for k, (lab, d) in LABEL.items():
    r = rep[k]
    a, b = float(r["v1_printed"]), float(r["v1_code_on_frozen_data"])
    fa = f"{int(a):,}" if d == 0 else f"{a:.{d}f}"
    fb = f"{int(b):,}" if d == 0 else f"{b:.{d}f}"
    lines.append(f"{lab} & {fa} & {fb} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_reproduced.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table: channels
lines = []
for name, r in ch.items():
    e = eff.get(name)
    effs = f"{dec(e['effect_per_sd'], 2)} [{dec(e['lo95'], 2)}, {dec(e['hi95'], 2)}]" if e else "--"
    lines.append(f"{name} & {pval(r['joint_p_same_day'])} & {pval(r['sum_p_same_day'])} & {effs} & "
                 f"{pval(r['joint_p_lagged'])} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_channels.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table: forecasts
lines = []
for name, r in fc.items():
    if name == "ARX, same-day factors":
        lines.append("\\midrule")
        lines.append("\\multicolumn{7}{l}{\\emph{Same-day factors (design of version 1, not a forecast)}} \\\\")
    da = pct(r["dir_acc"]) if r["dir_acc"] != "nan" else "--"
    pt = pval(r["PT_p"]) if r["PT_p"] not in ("", "nan") else "--"
    dm = pval(r["DM_p"]) if r.get("DM_p") else "--"
    cw = pval(r["CW_p"]) if r.get("CW_p") else "--"
    lines.append(f"{name} & {dec(r['RMSE'], 3)} & {dec(r['R2_OS'], 3)} & {da} & {pt} & {dm} & {cw} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_forecasts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Wrote numbers.tex, tab_reproduced.tex, tab_channels.tex and tab_forecasts.tex.")
