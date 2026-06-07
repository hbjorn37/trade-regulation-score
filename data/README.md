# Data

The contents of this directory are not real import records. Everything under [`sample/`](sample/) is synthetic data produced by [`../scripts/generate_sample_data.py`](../scripts/generate_sample_data.py) with a fixed random seed. It exists so the pipeline can run end-to-end on a fresh checkout and so the expected file shape (filenames, column headers, value types) is documented by example. Use it to verify that the pipeline works, not to draw conclusions about Korean trade.

## Where the thesis data came from

The thesis used two inputs that this repository does not redistribute.

| Input | Public source | Why it is not shipped |
| --- | --- | --- |
| Import records by HS code | Korean trade-statistics portal at [tradedata.go.kr](https://tradedata.go.kr/cts/index.do) | The portal lets you download your own extracts; please retrieve them yourself rather than relying on a re-hosted copy |
| News corpus | Naver News (semi case), Baidu News (urea case) | Subject to source terms of service |

The first portal is open to the general public. Anyone can register an account, search by HS code, country, and time window, and download the resulting CSV. The portal publishes import aggregates by HS code and partner country at monthly resolution; that monthly snapshot is the default path documented below. If you obtained daily-resolution records through some other route (your own scrape, a research-data agreement, etc.), the pipeline accepts those directly too.

## Layout

```
data/
  sample/                              shipped synthetic data (tracked in git)
    semi/
      import_monthly_2017_2022.csv     mimics a portal monthly snapshot
      trs_final_2017_2022.csv          mimics daily TRS scores
    urea/
      import_monthly_2017_2022.csv
      trs_final_2017_2022.csv
  real/                                your real data goes here (gitignored)
    semi/...
    urea/...
```

`data/real/` and `data/private/` are excluded from git via the repo root `.gitignore`. Drop your real CSVs there following one of the two paths described below.

## Path A (default): monthly snapshot from the portal

The portal's regular export is a monthly aggregate per HS code and partner country. Download it as CSV, then convert to the weekly file the pipeline needs with [`../scripts/monthly_to_weekly.py`](../scripts/monthly_to_weekly.py). The helper uses calendar-day proration: each month's total is spread uniformly across the days of that month, then re-summed into ISO weeks. Weeks that span a month boundary get a weighted blend; the monthly grand total is preserved exactly. Within-month variation is smoothed away in the process, so the weekly series will look more piecewise-constant than one derived from true daily filings.

Prepare a monthly CSV with at least these columns:

| column | type | description |
| --- | --- | --- |
| year_month | str | `YYYY-MM` |
| reported_weight | float | monthly sum of declared weight |
| taxable_value_usd | float | monthly sum of taxable value in USD |

Optionally include `hs_code` and `origin_country` columns; the script can filter on either.

```bash
uv run python scripts/monthly_to_weekly.py \
    --input  data/real/semi/import_monthly_2017_2022.csv \
    --output intermediate/semi/import_weekly.csv \
    --hs-code 3707901010 \
    --origin-country JP \
    --start 2017-01 --end 2022-09

uv run python scripts/run.py semi --from 02 --data-root data/real
```

`scripts/run.py ... --from 02` skips step 01 because the weekly file is already in place at `intermediate/<dataset>/import_weekly.csv`. The `--data-root data/real` flag tells step 02 and step 06 where to read the daily TRS file from.

To exercise the same flow against the shipped synthetic sample, replace `data/real` with `data/sample` in both commands.

## Path B: you obtained daily records yourself

If you have daily-resolution import records from any other route (your own scrape, a separate research-data agreement, etc.), use them directly. Save the file at `data/real/<dataset>/import_daily_2017_2022.csv` matching the schema below and run the full pipeline. Step 01 will aggregate to weekly.

`import_daily_*.csv` (utf-8, no BOM)

| column | type | description |
| --- | --- | --- |
| hs_code | str | 10-digit HS code as a string |
| clearance_date | str | YYYYMMDD |
| origin_country | str | 2-letter country code |
| reported_weight | float | declared weight |
| taxable_value_usd | float | taxable value in USD |

```bash
uv run python scripts/run.py semi --data-root data/real
```

The shipped sample uses Path A, so this repo does not ship a daily example file; the column schema above is what step 01 expects.

## TRS file schema (both paths)

Both Path A and Path B need the daily TRS file under `data/real/<dataset>/trs_final_2017_2022.csv` (Path A reads it directly, even though the import side bypasses step 01).

`trs_final_*.csv` (utf-8 with BOM)

| column | type | description |
| --- | --- | --- |
| date | str | YYYY.MM.DD |
| trs_score | float | 0 to 999 |
| rationale | str | short label describing the score band |
| article_count | int | articles considered that day |
| country_mention_count | int | how many mentioned the target country |
| risk_signal_count | int | how many contained any risk keyword |
| positive_signal_count | int | how many contained a positive keyword |

## Producing the TRS scores

The thesis ran a local LLM (`qwen2.5:14b` served via Ollama) over a daily news corpus to produce `trs_final_*.csv`. Reproducing this step is optional and requires Ollama plus a news corpus; see [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) and the reference implementation at [`../reference/trs_scoring/score_trs.py`](../reference/trs_scoring/score_trs.py).
