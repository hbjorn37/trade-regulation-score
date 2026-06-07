# Sample data notice

The files in this directory are synthetic. They are not downloads from any primary source. They were produced by [`../../scripts/generate_sample_data.py`](../../scripts/generate_sample_data.py) with a fixed random seed so the same inputs reproduce byte-for-byte.

The intent is to give a fresh checkout enough data to run the pipeline end-to-end, and to make the pipeline's output (forecasts, leading-indicator plots, early-warning report) line up qualitatively with the thesis case studies, not just emit random noise.

## What each file mimics

`import_monthly_2017_2022.csv` mimics a monthly HS-code snapshot of the sort the Korean public trade-statistics portal at [tradedata.go.kr](https://tradedata.go.kr/cts/index.do) returns to a general account. Columns and shape match what such a snapshot looks like:

| column | example |
| --- | --- |
| year_month | `2019-07` |
| hs_code | `3707901010` |
| origin_country | `JP` |
| reported_weight | `41.20` |
| taxable_value_usd | `8820.10` |

Each month has one row per partner country (the target country plus four others), so the file resembles a multi-country portal extract.

`trs_final_2017_2022.csv` mimics the daily TRS score table the reference scoring procedure in [`../../reference/trs_scoring/`](../../reference/trs_scoring/) would emit when run over a real news corpus. See [`../../docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md) for the scoring methodology.

## How the trends were designed

The two case studies in the thesis have distinct qualitative signatures, and the generator reproduces them with deliberately simple structure so that the pipeline output (the BASE vs LLM model comparison, the lagged correlation panel, and the threshold table in `early_warning/report.md`) shows recognisable patterns rather than noise.

`semi` (HS 3707901010, photoresist from Japan, shock 2019-07-01)

- Baseline unit price around USD 180/kg before the shock.
- A smooth ramp over roughly six weeks centered on 2019-07 lifts the unit price by about 40 percent at the peak.
- After the peak the unit price decays toward a new long-run level about 15-20 percent above baseline, capturing the persistent elevation described in the thesis.
- Monthly weights are lognormal around a level appropriate for a low-volume, high-value specialty material.

`urea` (HS 3102109000, urea from China, shock 2021-10-11)

- Baseline unit price around USD 0.52/kg, much lower and higher-volume than the semi case.
- A sharper ramp around 2021-10 doubles the unit price at the peak.
- Post-shock the unit price settles around 40-50 percent above baseline with more residual volatility than the semi case.

Both cases share the same TRS structure: an AR(1) process with a sigmoid step centered roughly three weeks before the shock date, decaying back toward baseline over a few months. This is what makes the early-warning report show a Lift greater than 20x at the Alert threshold (400) with a single-digit-to-low-double-digit pre-shock lead time, matching the leading behaviour reported in the thesis.

## What this is not

These rows are not evidence about Korean imports or about any actual export-restriction event. The magnitudes are stylised to be qualitatively similar to the thesis case studies, not to reproduce the exact numbers. Reproducing the thesis numbers requires real portal downloads; see [`../README.md`](../README.md).
