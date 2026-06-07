# Pipeline Specification

This document describes the inputs, outputs, and contract of each pipeline step in enough detail that a reviewer can verify correctness independently.

## 1. Configuration

Each dataset has one YAML file under `configs/`. The schema is:

| Field | Type | Description |
| --- | --- | --- |
| name | str | Dataset id matching the YAML filename |
| description | str | Human-readable summary |
| hs_code | str | 10-digit HS code |
| origin_country | str | Two-letter country code used to filter the import series |
| shock_date | str | YYYY-MM-DD reference date of the supply shock |
| shock_week | str | `YYYY_WW` (ISO) for the shock week |
| train_end_week | str | Last week included in training |
| test_start_week | str | First week of the test split |
| data_start_week | str | Beginning of the weekly window |
| data_end_week | str | End of the weekly window |
| country_mention_label | str | Plot legend label for country mentions |
| item_title | str | Plot title for the case study |
| shock_label | str | Annotation label drawn on plots |
| shock_weeks | list[str] | Weeks shaded as the shock range |
| files.import_csv | str | Filename inside `data/<root>/<dataset>/` |
| files.trs | str | Filename inside `data/<root>/<dataset>/` |

## 2. Step 01 -- import_weekly

Input: `data/<root>/<dataset>/<files.import_csv>` with columns `hs_code, clearance_date, origin_country, reported_weight, taxable_value_usd`.

Processing:

1. Parse `clearance_date` as `%Y%m%d`.
2. Filter rows where `origin_country == config.origin_country`.
3. Compute `year_week` from the ISO calendar.
4. Sum `reported_weight` and `taxable_value_usd` per week.
5. Reindex against the full week range `data_start_week..data_end_week` and linearly interpolate inner gaps.

Output: `intermediate/<dataset>/import_weekly.csv` and `.png`.

## 3. Step 02 -- trs_weekly

Input: `data/<root>/<dataset>/<files.trs>` with at minimum `date, trs_score, article_count, country_mention_count, risk_signal_count`.

Processing:

1. Parse `date` as `%Y.%m.%d`.
2. Compute `year_week` and aggregate to weekly mean, max, min, std, count of `trs_score` plus sums of article-count columns.
3. Truncate to `data_end_week`.

Output: `intermediate/<dataset>/trs_weekly.csv` and `.png`.

## 4. Step 03 -- merge_features

Input: weekly import + weekly TRS.

Processing:

1. Inner-join on `year_week`.
2. Derive `import_unit_price = taxable_value_usd / reported_weight`.
3. Derive TRS features: 4- and 12-week moving averages, deviation features, percent-change features, spike/high-risk/crisis dummies, 1-4 week lags of `trs_mean`, `trs_max`, and `trs_pct_change`, a 4-week cumulative spike count, risk-signal and country-mention ratios, TRS volatility, and lag-1 versions of the dummies and ratios.
4. Drop the first four rows (lag-4 burn-in).

Output: `intermediate/<dataset>/merged.csv` plus `trs_validation.png` with a Granger-causality test and lagged correlations.

## 5. Step 04 -- forecast_model

Input: `intermediate/<dataset>/merged.csv`.

Processing:

1. Sort by `year_week`; trim any tail of repeated values that result from data-source interpolation.
2. Split: train = `year_week <= train_end_week`, test = `>= test_start_week`.
3. For each model in `AVAILABLE_MODELS`, train a BASE variant on `import_unit_price` lags only, then (except for `Chronos`) train an LLM variant that adds the TRS exogenous columns listed in `TRS_EXOG_COLS` (see [`step04_forecast_model.py`](../src/trs_pipeline/step04_forecast_model.py)).
4. One-step-ahead predictions use actual lag values, not predicted ones, to avoid data leakage.

Models:

| Name | BASE | LLM (TRS) | Notes |
| --- | --- | --- | --- |
| ARIMA | (1,1,1) | ARIMAX with `trs_mean_lag1` | statsmodels |
| RandomForest | sklearn | sklearn + TRS lags | 500 trees |
| LSTM | Keras | Keras + TRS lags | requires `tensorflow` extra |
| GBM | sklearn | sklearn + TRS lags | |
| LightGBM | LGBMRegressor | + TRS lags | |
| NGBoost | NGBRegressor | + TRS lags | |
| XGBoost | XGBRegressor | + TRS lags | |
| CatBoost | CatBoostRegressor | + TRS lags | |
| ExtraTrees | sklearn | sklearn + TRS lags | |
| Chronos | amazon/chronos-bolt-small | -- | univariate; no LLM variant |

Output: `output/<dataset>/forecast.csv`, `eval.csv`, `eval_all.csv`. Each row in `eval.csv` is one model variant. Rows whose forecasts contain NaN or that diverged (max forecast magnitude > 100x max actual) are dropped.

## 6. Step 05 -- plot_evaluation

Reads the eval and forecast tables and the merged data, then writes:

- `data_insight.png`: import unit price and TRS time series side-by-side, with lagged correlations.
- `result_insight.png`: model-wise BASE vs LLM MAE bars, improvement bars, and a forecast overlay for the best-improved model.

## 7. Step 06 -- early_warning

Input: daily TRS, weekly TRS, merged.

Processing:

1. Compute precision, recall, false-alarm rate, lift, and lead time over a sweep of TRS thresholds from 100 to 900, using a 90-day pre-shock window and the period before that as a baseline.
2. Render `overview.png`, `signal_quality.png`, `shock_trajectory.png`, `warning_simulation.png`.
3. Generate `report.md` with the threshold table at Caution(200), Alert(400), Risk(600), Critical(800).

## 8. Smoke test

A successful run on synthetic data should yield, for each dataset:

- `output/<dataset>/eval.csv` with at most 19 rows (9 BASE + 9 LLM + 1 Chronos), fewer if LSTM is skipped (the `lstm` extra is optional).
- All five early-warning artifacts plus `report.md`.
- `result_insight.png` showing visible MAE differences between BASE and LLM for at least a few models.
