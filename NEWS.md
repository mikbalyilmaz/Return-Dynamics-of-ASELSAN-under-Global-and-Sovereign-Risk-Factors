# Changes

## Version 2.0.0

### New

- `aselsan/` holds the data alignment, the ARX designs and their estimation, and the recursive forecasts with their
  evaluation, as a package; the variables, transformations, lag orders ($p = q = 2$), HAC lag (5), sample split and
  model hyperparameters are those of version 1.
- The Yahoo Finance series are downloaded once (`data/fetch_yahoo.py`) and stored with the repository, so every
  number can be reproduced.
- Two designs of the ARX model, kept apart: the same-day design of version 1 (factors at lags 0 to 2), which describes
  co-movement, and a forecasting design (factors at lags 1 to 2), in which every input is known at the previous close.
- Joint Wald tests for each channel (all coefficients of a factor) and cumulative effects per standard deviation of
  each factor.
- Recursive one-day-ahead forecasts, refitted every day, for zero, prevailing mean, random walk, AR, ARX, Ridge,
  Lasso, random forest and XGBoost; the machine-learning models get exactly the ARX inputs. Evaluation by the
  out-of-sample $R^2$ against the prevailing mean, Diebold–Mariano with the Harvey–Leybourne–Newbold correction,
  Clark–West for nested models and the Pesaran–Timmermann test of directional accuracy.
- Unit tests (`tests/run_tests.py`): the alignment against a direct search, the Newey–West covariance written out,
  the Wald statistic as a quadratic form, recursive forecasts against statsmodels, perturbation of future data (no
  look-ahead), simulated series with a known answer, and the code of version 1 against its printed output.
- Three analyses, three figures, CSV tables, and a project note in English with the mathematics of every step, whose
  numbers are written from the tables.
- `CITATION.cff`, `requirements.txt`, `.gitignore`, a GitHub Actions workflow that runs the tests.

### Corrections to version 1

| # | Version 1 | Version 2 |
|---|---|---|
| 1 | The "out-of-sample forecast" of the ARX model for day $t$ used the factors of day $t$ (lag 0: the BIST100 return, the exchange rate, VIX, CDS of the same day; the code comment reads "build X row for date t (including t)"). These values are known only after day $t$ ends, so the reported accuracy (RMSE 2.49, directional accuracy 64.9%) measures same-day co-movement, not forecasting. | Forecasts use lags 1 and 2 only. With this information no model beats the prevailing mean ($R^2_{OS} = -0.044$ for ARX). The same-day design is reported separately, as co-movement, and gives $R^2_{OS} = 0.154$; the same-day BIST100 return alone gives 0.164. |
| 2 | The machine-learning models were given every column of the data table except the ASELSAN return: same-day returns and price levels of all factors and the closing price of ASELSAN on day $t$, with no lags; the text described them as using "the same feature space consisting of all lagged variables used in the ARX specification". They were fitted once on the training sample. | The machine-learning models get exactly the ARX inputs and are refitted every day, like the ARX model. |
| 3 | Ridge and Lasso chose their penalties by `cv=5`, an unshuffled five-fold split in which four of the five folds are trained partly on data later than the data they are validated on. | Five-fold time-series cross-validation (`TimeSeriesSplit`): each fold is validated only on later data. The penalty grid of version 1 is kept. |
| 4 | The Yahoo series were downloaded at every run up to the day of the run, so the results could not be reproduced. The run of 12 January 2026 took its last row during the trading day (Brent 63.40 against a final close of 63.87, VIX 15.17 against 15.12). | Data downloaded once for the window of version 1 and stored with the download date and software versions. The code of version 1 on these data reproduces its printed $R^2$, market coefficient and RMSEs to within 0.01 (XGBoost to within 0.08). |
| 5 | The CDS file ends on 8 January 2026; its last value was carried forward to 9 and 12 January, so the last two CDS changes were zero. | The panel ends on 8 January 2026, the last date on which every series has its own data. |
| 6 | The channel hypotheses ($\beta_{k,j} = 0$ for all lags $j$) were stated but tested on single coefficients (and, for VIX and CDS, on their sum). | Joint Wald tests with the Newey–West covariance, together with the cumulative effects. |
| 7 | The AR-only benchmark had 5 lags, the ARX model 2; the directional accuracy of the zero forecast was reported as 0.77% (the share of days on which the return was exactly zero, since the forecast has no sign); the Diebold–Mariano test used the normal distribution without a small-sample correction. | The AR benchmark has the AR lags of the ARX model; a forecast without a sign has no directional accuracy; the prevailing mean is the main benchmark; Diebold–Mariano with the Harvey–Leybourne–Newbold correction and $t_{T-1}$, and Clark–West for nested models. |
| 8 | The conclusion stated that the Diebold–Mariano tests between ARX and the best machine-learning model were not significant, "e.g., ARX vs. Lasso, p ≈ 0.07"; the notebook's own output is $p = 0.3212$. The README concluded that ASELSAN's returns show "a modest but robust degree of predictability". | The note reports only numbers written from the tables. With information up to the previous close, no predictability is found in the test period. |
| 9 | `main.html` was empty (2 bytes); LaTeX build files (`ref.aux`, `ref.log`, `ref.out`, `ref.synctex.gz`) were committed; the citation link was a `<USERNAME>/<REPO>` placeholder; the README said outputs are saved under `figures/`, while they were under `ASELS/figures/`. | Version 1 kept in `legacy/` (notebook, README, reference list); its other files remain in the `v1.0.0` release. Build files are ignored by git; `CITATION.cff` gives the repository link. |
| 10 | The reference list had incomplete entries (for example Buluş (2025), "IEEE", and Kalyoncu et al. (2020), without a venue), and its literature was not cited in the text. | The note cites the methods it uses, with full references. |

## Version 1.0.0

First version: an ARX model of ASELSAN's daily return with eight factors, a robustness grid, an out-of-sample
comparison with simple benchmarks and four machine-learning models, figures and a reference list.
