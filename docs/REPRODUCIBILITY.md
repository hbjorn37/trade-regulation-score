# Reproducibility Notes

## Synthetic data vs the thesis

The repository ships only synthetic sample data under `data/sample/`. The thesis numbers were computed against real Korean import extracts downloaded from the public trade-statistics portal at [tradedata.go.kr](https://tradedata.go.kr/cts/index.do), plus a Korean-language news corpus. Those source rows are not re-hosted here; you are expected to retrieve your own through the portal. As a result:

- The shapes of the pipeline artifacts (CSV columns, plot layouts, the threshold table in `report.md`) match the thesis.
- The numerical values (MAE, RMSE, lift, lead time) do not match.
- On the synthetic series the Chronos zero-shot forecaster substantially outperforms the trained models. The shock signal in the synthetic data is cleaner than in real import records, which is exactly the regime in which a foundation model with strong priors does well.

## Where the synthetic numbers come from

The synthetic generator uses:

- A deterministic numpy generator seed per dataset, so every run produces identical files.
- A monthly unit-price series with a baseline AR(1) noise floor, a smooth shock ramp centered on `shock_date`, and a long-run elevated tail that decays back toward a new equilibrium. The semi case peaks at about +40 percent over baseline and settles around +15 percent; the urea case roughly doubles at the peak and settles around +40-50 percent.
- A daily TRS series driven by AR(1) noise plus a sigmoid step that rises about three weeks before the shock date and decays over a few months.

These choices make the leading-indicator pattern visible without making it trivial. The first-warning lead time at Alert (400) on the sample data sits around 11 to 22 days, which is in the same range the thesis reports for the real data.

## To reproduce the thesis numbers

1. Get import records for the two HS codes (3707901010 for semi, 3102109000 for urea) over 2017 to 2022. The default route is the monthly snapshot from the Korean public trade-statistics portal at <https://tradedata.go.kr/cts/index.do>, converted to the weekly file the pipeline needs with [`../scripts/monthly_to_weekly.py`](../scripts/monthly_to_weekly.py). If you obtained daily-resolution records yourself by some other means, use them as-is and let step 01 aggregate them. The full recipe for both paths is in [`../data/README.md`](../data/README.md).
2. Collect news for the same period from Naver News (semi case) and Baidu News (urea case), or any source that provides a date / title / body schema.
3. Score the news with [`../reference/trs_scoring/score_trs.py`](../reference/trs_scoring/score_trs.py) using a keyword list appropriate to the corpus. Expect roughly one to two days on a single GPU for the 2017-2022 corpus.
4. Place the resulting import file (`import_monthly_2017_2022.csv` for Path A or `import_daily_2017_2022.csv` for Path B) and `trs_final_2017_2022.csv` under `data/real/<dataset>/` and run the pipeline as documented in the main README.

## Optional dependencies

LSTM requires TensorFlow, which is intentionally not in the default install to keep the dependency footprint reasonable. Enable it explicitly:

```bash
uv sync --extra lstm
```

Without TensorFlow, `step04` prints a skip message and the LSTM row is absent from `eval.csv`. All other models, including Chronos, run with the default install.
