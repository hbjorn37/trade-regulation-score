# Trade Regulation Score

Reproducible pipeline for forecasting import-price supply shocks using a news-derived Trade Regulation Score (TRS). Accompanies a KAIST master's thesis that benchmarks ten models (ARIMA, RandomForest, LSTM, GBM, LightGBM, NGBoost, XGBoost, CatBoost, ExtraTrees, Chronos) on two case studies:

| Case | HS code | Origin | Shock event |
| --- | --- | --- | --- |
| Semiconductor materials | 3707901010 | JP | 2019-07-01 Japan export restriction |
| Urea solution | 3102109000 | CN | 2021-10-11 China export inspection tightening |

For each case the pipeline aggregates daily import records into weekly series, joins them with weekly TRS features derived from a local LLM, and trains both a BASE model (autoregressive only) and an LLM-augmented variant (autoregressive plus TRS exogenous features). It then runs a daily-TRS-based early-warning analysis with lead-time, recall, precision, FA/year, and lift at four threshold levels.

The repository ships synthetic sample data so the entire pipeline can be run end-to-end on a fresh checkout. The thesis itself used import records downloaded from the Korean public trade-statistics portal at [tradedata.go.kr](https://tradedata.go.kr/cts/index.do); the rows themselves are not redistributed here. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for what differs between the sample run and the numbers in the thesis.

## Pipeline

```
data/<root>/<dataset>/
        |  import_monthly_2017_2022.csv    (Path A: portal snapshot)
        |  import_daily_2017_2022.csv      (Path B: your own daily extract)
        |  trs_final_2017_2022.csv         (both paths)
        v
[Path A] monthly_to_weekly.py
            -> intermediate/<dataset>/import_weekly.csv
[01] import_weekly  -> intermediate/<dataset>/import_weekly.{csv,png}    (Path B only)
[02] trs_weekly     -> intermediate/<dataset>/trs_weekly.{csv,png}
[03] merge_features -> intermediate/<dataset>/{merged.csv, trs_validation.png}
[04] forecast_model -> output/<dataset>/{forecast.csv, eval.csv, eval_all.csv}
[05] plot_evaluation-> output/<dataset>/{data_insight.png, result_insight.png}
[06] early_warning  -> output/<dataset>/early_warning/{overview.png,
                                                        signal_quality.png,
                                                        shock_trajectory.png,
                                                        warning_simulation.png,
                                                        report.md}
```

## Get Started

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) (`pipx install uv` or see the link)
- Roughly 2 GB of disk for the Chronos model checkpoint and Python deps

### 1. Clone and install

```bash
git clone git@github.com:hbjorn37/trade-regulation-score.git
cd trade-regulation-score
uv sync
```

The first sync downloads PyTorch and the Chronos foundation model the first time `step04` runs; expect a few minutes for the initial install.

### 2. Look at the sample data

```
data/sample/semi/import_monthly_2017_2022.csv
data/sample/semi/trs_final_2017_2022.csv
data/sample/urea/import_monthly_2017_2022.csv
data/sample/urea/trs_final_2017_2022.csv
```

These are not real downloads. They are deterministic synthetic files produced by [`scripts/generate_sample_data.py`](scripts/generate_sample_data.py) with a fixed seed. The import file mimics a monthly HS-code snapshot of the sort the Korean public trade-statistics portal at [tradedata.go.kr](https://tradedata.go.kr/cts/index.do) returns to a general account, including multiple partner-country rows per month. The TRS file mimics the daily output of the scoring procedure in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The trends are tuned to qualitatively match the thesis case studies (semi sees roughly a 40 percent unit-price lift around 2019-07; urea sees a ~2x spike around 2021-10), so the pipeline emits a realistic-looking lagged correlation panel and early-warning report. See [`data/sample/NOTICE.md`](data/sample/NOTICE.md) for the trend-design details and [`data/README.md`](data/README.md) for the column schema.

Regenerate the sample files with:

```bash
uv run python scripts/generate_sample_data.py
```

### 3. Convert the monthly snapshot to weekly

The pipeline's step 03 onward operates on weekly series. Convert each dataset's monthly file to the weekly drop-in at `intermediate/<dataset>/` with [`scripts/monthly_to_weekly.py`](scripts/monthly_to_weekly.py):

```bash
uv run python scripts/monthly_to_weekly.py \
    --input  data/sample/semi/import_monthly_2017_2022.csv \
    --output intermediate/semi/import_weekly.csv \
    --hs-code 3707901010 --origin-country JP \
    --start 2017-01 --end 2022-09

uv run python scripts/monthly_to_weekly.py \
    --input  data/sample/urea/import_monthly_2017_2022.csv \
    --output intermediate/urea/import_weekly.csv \
    --hs-code 3102109000 --origin-country CN \
    --start 2017-01 --end 2022-09
```

The helper uses calendar-day proration: each month's total is spread uniformly across the days of that month and then re-aggregated by ISO week. Weeks that span a month boundary get a weighted blend; the monthly grand total is preserved exactly. See [`data/README.md`](data/README.md) for the Path A vs Path B sourcing details.

### 4. Run the pipeline from step 02

Step 01 (daily-to-weekly aggregation) is bypassed because the weekly file is already in place from step 3 above.

```bash
# Semiconductor case (Japan export restriction, 2019-07-01)
uv run python scripts/run.py semi --from 02 --data-root data/sample

# Urea case (China export inspection, 2021-10-11)
uv run python scripts/run.py urea --from 02 --data-root data/sample

# Both, sequentially
uv run python scripts/run.py all --from 02 --data-root data/sample
```

Each run takes roughly 5-15 minutes on a CPU laptop. Step 04 dominates the wall-clock time because it trains nine models in BASE and LLM variants plus runs the Chronos zero-shot forecaster.

### 5. Inspect the results

After the run completes:

| Path | What to look at |
| --- | --- |
| `output/<dataset>/eval.csv` | Per-model MAE, RMSE, MAPE, Theil-U, direction accuracy, spike detection rate, and trend correlation |
| `output/<dataset>/result_insight.png` | BASE vs LLM MAE bar chart, improvement chart, best-improved model forecast |
| `output/<dataset>/data_insight.png` | Import unit price overlaid with TRS, plus lagged correlation bars |
| `output/<dataset>/early_warning/report.md` | Lead-time, recall, precision, false-alarm rate, and lift at four TRS thresholds |
| `output/<dataset>/early_warning/*.png` | Overview, signal quality, shock trajectory, warning simulation |

## Running on Your Own Data

Drop your real CSVs into `data/real/<dataset>/` (which is gitignored), then run the same flow with `--data-root data/real` instead of `data/sample`. Pick the path that matches what your portal account returns.

### Path A: portal monthly snapshot (default for tradedata.go.kr)

Place a monthly CSV at `data/real/<dataset>/import_monthly_2017_2022.csv` with columns `year_month`, `hs_code`, `origin_country`, `reported_weight`, `taxable_value_usd`, plus the daily TRS at `data/real/<dataset>/trs_final_2017_2022.csv`. Then convert and run:

```bash
uv run python scripts/monthly_to_weekly.py \
    --input  data/real/semi/import_monthly_2017_2022.csv \
    --output intermediate/semi/import_weekly.csv \
    --hs-code 3707901010 --origin-country JP \
    --start 2017-01 --end 2022-09

uv run python scripts/run.py semi --from 02 --data-root data/real
```

Repeat for `urea` with `--hs-code 3102109000 --origin-country CN`.

### Path B: you already have daily records

Place a daily CSV at `data/real/<dataset>/import_daily_2017_2022.csv` with columns `hs_code`, `clearance_date` (YYYYMMDD), `origin_country`, `reported_weight`, `taxable_value_usd`. Then run the full pipeline so step 01 aggregates daily to weekly for you:

```bash
uv run python scripts/run.py semi --data-root data/real
```

The full schemas, sourcing tips, and how to produce the daily TRS file yourself are in [`data/README.md`](data/README.md).

### Step-level execution

```bash
# Run a single step (assumes its inputs are already in place)
uv run python scripts/run.py semi --step 03 --data-root data/sample

# Run a range
uv run python scripts/run.py semi --from 04 --to 05 --data-root data/sample
```

The `--data-root` flag tells the data-reading steps (01, 02, 06) where their inputs live. The default is `data/sample`.

## Repository Layout

```
trade-regulation-score/
  configs/                       per-dataset YAML
  src/trs_pipeline/              importable package
    step01_import_weekly.py
    step02_trs_weekly.py
    step03_merge_features.py
    step04_forecast_model.py     trains the 10 models
    step05_plot_evaluation.py
    step06_early_warning.py
    models/chronos_model.py      Chronos zero-shot wrapper
  scripts/
    run.py                       pipeline entrypoint
    monthly_to_weekly.py         portal monthly -> weekly converter (Path A)
    generate_sample_data.py      deterministic synthetic data generator
  data/sample/                   synthetic CSVs (tracked)
  reference/trs_scoring/         Ollama-based TRS scoring (reference only)
  docs/
    SPEC.md                      detailed step contracts
    METHODOLOGY.md               TRS scoring methodology
    REPRODUCIBILITY.md           synthetic vs real, what to expect
```

## Citation

If you build on this work, please cite the accompanying thesis. A [`CITATION.cff`](CITATION.cff) is included; GitHub renders a "Cite this repository" button.

## License

MIT. See [`LICENSE`](LICENSE).
