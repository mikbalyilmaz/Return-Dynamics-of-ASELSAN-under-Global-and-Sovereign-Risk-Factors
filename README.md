# ASELSAN Return Predictability: Econometric and Machine Learning Benchmarking

This project investigates whether ASELSAN stock returns exhibit economically meaningful and empirically robust predictability using a combination of econometric modeling and modern machine learning methods. The analysis is built around a carefully specified multivariate ARX framework and a strict out-of-sample forecasting design.

The main objective is twofold:
(i) to identify whether macro-financial variables such as market returns, exchange rates, global risk, and sovereign risk contain predictive information for ASELSAN returns, and  
(ii) to assess whether more flexible machine learning models can systematically outperform a well-specified linear econometric benchmark.

---

## Data and Variables

The dataset is constructed at daily frequency and aligned to ASELSAN trading days. It includes:

- ASELSAN stock price (dependent variable)
- BIST100 index (market factor)
- USD/TRY exchange rate
- VIX (global risk)
- DXY (dollar index)
- US 10Y yield
- Turkey CDS
- Brent oil
- Bitcoin

Price-type variables are transformed into log-returns, while risk and macro variables are used in first differences or levels depending on the specification.

---

## Methodology

### 1. Baseline Econometric Model (ARX)

The core model is a distributed-lag ARX specification:

- Includes lags of ASELSAN returns
- Includes contemporaneous and lagged macro-financial predictors
- Estimated by OLS with Newey–West (HAC) standard errors
- Subjected to extensive robustness checks:
  - Lag lengths p = q = 1, 3, 5
  - VIX and CDS in levels vs differences

### 2. Out-of-Sample Forecasting

A strict recursive real-time forecasting exercise is implemented:

- Training period: 2020–2024  
- Test period: 2025–2026  
- One-step-ahead forecasts
- Evaluation metrics:
  - RMSE
  - MAE
  - Directional Accuracy
- Statistical comparison via Diebold–Mariano tests

Benchmarks:
- Zero forecast
- Random walk
- AR-only model

### 3. Machine Learning Benchmarks

Using the exact same information set and sample split:

- Ridge regression
- Lasso regression
- Random Forest
- XGBoost

All models are trained only on the training sample and evaluated strictly out-of-sample.

---

## Main Findings

- ASELSAN returns exhibit a modest but robust degree of predictability linked to:
  - Market movements (BIST100)
  - Exchange rate dynamics (USD/TRY)
  - Global risk (VIX)
- The ARX model significantly outperforms naive benchmarks in out-of-sample forecasts.
- Machine learning models do **not** deliver systematic or statistically significant improvements over the linear ARX benchmark.
- Forecasts from all models are strongly shrunk toward zero, highlighting the dominance of noise at daily frequency.
- More complex nonlinear models (e.g., XGBoost) perform worse, suggesting limited exploitable nonlinear structure in the data.

---

## Interpretation

The results support a cautious but clear conclusion:

- ASELSAN returns are not purely unpredictable, but predictability is:
  - Structurally limited
  - Economically small
  - Heavily dominated by noise
- Careful economic modeling and variable selection matter more than algorithmic complexity.
- Linear, theory-guided models remain extremely hard to beat in daily return forecasting.

---

## Project Structure

- Data loading and cleaning
- Variable transformations
- ARX estimation and robustness checks
- Out-of-sample forecasting engine
- Machine learning benchmarks
- Statistical evaluation and visualization
- Final comparison and discussion figures (PDF)

All steps are fully reproducible.

---

## How to Run

1. Install requirements:
   - numpy, pandas, statsmodels, scikit-learn, matplotlib, xgboost

2. Run the notebook / script in order:
   - Data construction
   - Estimation
   - Robustness checks
   - OOS forecasting
   - ML benchmarks
   - Figures and tables

3. Outputs (tables and figures) are saved under:
figures/


---

## Academic Positioning

This project is inspired by the large literature on return predictability and forecasting, in particular:

- Welch & Goyal (2008)
- Rapach & Zhou (2013)
- And the broader literature on the limits of predictability in financial markets.

---

## Disclaimer

This project is for academic and research purposes only. It does not constitute investment advice.

---

*— The End —*

## Citation

If you use this repository (code, workflow, or outputs) in academic work, you can cite it as:

### BibTeX
```bibtex
@misc{yilmaz_asels_predictability_2026,
  author       = {Yilmaz, Muhammed Ikbal},
  title        = {ASELSAN Return Predictability: Econometric and Machine Learning Benchmarking},
  year         = {2026},
  howpublished = {\url{https://github.com/<USERNAME>/<REPO>}},
  note         = {Version 1.0. Accessed: 2026-01-12}
}

Muhammed Ikbal Yilmaz (2026). \textit{ASELSAN Return Predictability: Econometric and Machine Learning Benchmarking}. GitHub repository, \url{https://github.com/mikbalyilmaz}. Accessed: 12 Jan 2026.
