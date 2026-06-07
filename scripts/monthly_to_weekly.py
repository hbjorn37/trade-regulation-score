"""Convert monthly import aggregates into the weekly file the pipeline expects.

The Korean public trade-statistics portal at https://tradedata.go.kr/ exposes
import aggregates at HS-code level on a monthly basis to general accounts.
This pipeline operates on weekly series, so monthly extracts need to be
disaggregated. This script applies calendar-day proration: each month's
total is distributed uniformly across the days of that month, and the
resulting daily values are then re-aggregated into ISO weeks. Weeks that
span two calendar months receive a weighted contribution from each.

Input CSV (utf-8) -- one row per month, columns:

    year_month         str  YYYY-MM (e.g. 2019-07)
    reported_weight    float  monthly sum of declared weight
    taxable_value_usd  float  monthly sum of taxable value in USD

Optionally include extra columns (hs_code, origin_country, ...) to filter on.

Output CSV (utf-8) -- one row per ISO week, columns:

    year_week          str  YYYY_WW
    reported_weight    float
    taxable_value_usd  float

Place the output at intermediate/<dataset>/import_weekly.csv and run the
pipeline from step 02 (it skips step 01).

Usage::

    uv run python scripts/monthly_to_weekly.py \\
        --input  data/real/semi/import_monthly.csv \\
        --output intermediate/semi/import_weekly.csv \\
        --hs-code 3707901010 \\
        --origin-country JP \\
        --start 2017-01 --end 2022-09
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALUE_COLS = ["reported_weight", "taxable_value_usd"]


def explode_monthly_to_daily(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in monthly_df.iterrows():
        period = pd.Period(str(r["year_month"]), freq="M")
        days = pd.date_range(period.start_time, period.end_time, freq="D")
        n = len(days)
        per_day = {c: float(r[c]) / n for c in VALUE_COLS}
        for d in days:
            row = {"date": d}
            row.update(per_day)
            rows.append(row)
    return pd.DataFrame(rows)


def daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    iso = daily_df["date"].dt.isocalendar()
    daily_df = daily_df.copy()
    daily_df["year_week"] = (
        iso["year"].astype(str) + "_" + iso["week"].astype(str).str.zfill(2)
    )
    weekly = (
        daily_df.groupby("year_week", as_index=False)[VALUE_COLS].sum()
    )
    weekly["_sort_key"] = weekly["year_week"].apply(
        lambda x: tuple(int(v) for v in x.split("_"))
    )
    return weekly.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)


def filter_input(df: pd.DataFrame, hs_code: str | None, origin_country: str | None,
                 start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    if hs_code is not None and "hs_code" in out.columns:
        out = out[out["hs_code"].astype(str) == str(hs_code)]
    if origin_country is not None and "origin_country" in out.columns:
        out = out[out["origin_country"] == origin_country]
    if start is not None:
        out = out[out["year_month"].astype(str) >= start]
    if end is not None:
        out = out[out["year_month"].astype(str) <= end]
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hs-code", default=None)
    parser.add_argument("--origin-country", default=None)
    parser.add_argument("--start", default=None, help="Inclusive YYYY-MM lower bound")
    parser.add_argument("--end", default=None, help="Inclusive YYYY-MM upper bound")
    args = parser.parse_args()

    monthly = pd.read_csv(args.input, encoding="utf-8")
    missing = [c for c in ["year_month", *VALUE_COLS] if c not in monthly.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {missing}")

    monthly = filter_input(monthly, args.hs_code, args.origin_country, args.start, args.end)
    if monthly.empty:
        raise SystemExit("No rows survived filtering; check --hs-code / --origin-country / --start / --end")

    daily = explode_monthly_to_daily(monthly)
    weekly = daily_to_weekly(daily)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(weekly)} weekly rows to {args.output}")


if __name__ == "__main__":
    main()
