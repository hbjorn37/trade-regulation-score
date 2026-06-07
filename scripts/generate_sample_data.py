"""Generate the synthetic sample data shipped under data/sample/.

This script produces two CSV files per case study under data/sample/<dataset>/:

    import_monthly_2017_2022.csv   the monthly HS-code snapshot a user would
                                   download from the Korean public
                                   trade-statistics portal at
                                   https://tradedata.go.kr/cts/index.do
    trs_final_2017_2022.csv        the daily TRS scores produced by the
                                   reference scoring procedure in
                                   reference/trs_scoring/

These are not real downloads. The rows are synthesized so that the pipeline,
when fed through the documented Path A recipe (monthly_to_weekly.py then
run.py --from 02), emits forecasts and an early-warning report that
qualitatively match the thesis case studies. The synthesis is deterministic
(numpy seed 42 plus the dataset name) so re-running this script reproduces
the same CSVs byte-for-byte.

Trend design
------------

For each case study the unit price (taxable_value_usd / reported_weight)
follows a piecewise structure:

    baseline:    AR(1) around a case-specific level with mild seasonality
    shock ramp:  smooth rise over ~6 weeks centered on the shock date
    post-shock:  elevated mean that decays slowly toward a new equilibrium

Magnitudes are tuned to roughly match what the thesis reports: the
Japan-origin photoresist case (HS 3707901010) sees a ~40 percent unit-price
rise around 2019-07; the China-origin urea solution case (HS 3102109000)
sees a sharper ~2x spike around 2021-10 with greater post-shock volatility.

For each month the file contains one row per partner country so the
extract looks like a real portal download: a row for the target country
plus a few additional rows for other plausible sources of the same HS
code. The pipeline filters on origin_country, so only the target-country
rows actually drive forecasts.

The daily TRS series rises along a sigmoid centered roughly three weeks
before the shock date, so the leading-indicator pattern the thesis claims
shows up in the pipeline's lagged-correlation and early-warning artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    hs_code: str
    target_country: str
    other_countries: tuple[str, ...]
    shock_date: date
    baseline_unit_price: float
    shock_peak_multiplier: float       # peak unit price / baseline
    post_shock_multiplier: float       # long-run unit price / baseline (post recovery)
    target_weight_log_mu: float        # log-monthly weight for target country
    target_weight_log_sigma: float
    other_weight_log_mu: float         # log-monthly weight for other countries
    other_weight_log_sigma: float
    trs_baseline: float
    trs_post_shock: float
    trs_lead_days: int


SPECS = (
    CaseSpec(
        name="semi",
        hs_code="3707901010",
        target_country="JP",
        other_countries=("US", "DE", "TW", "CN"),
        shock_date=date(2019, 7, 1),
        baseline_unit_price=180.0,
        shock_peak_multiplier=1.42,
        post_shock_multiplier=1.18,
        target_weight_log_mu=3.6,
        target_weight_log_sigma=0.45,
        other_weight_log_mu=1.8,
        other_weight_log_sigma=0.7,
        trs_baseline=110.0,
        trs_post_shock=560.0,
        trs_lead_days=21,
    ),
    CaseSpec(
        name="urea",
        hs_code="3102109000",
        target_country="CN",
        other_countries=("RU", "QA", "OM", "ID"),
        shock_date=date(2021, 10, 11),
        baseline_unit_price=0.52,
        shock_peak_multiplier=2.1,
        post_shock_multiplier=1.45,
        target_weight_log_mu=14.0,
        target_weight_log_sigma=0.35,
        other_weight_log_mu=12.5,
        other_weight_log_sigma=0.6,
        trs_baseline=130.0,
        trs_post_shock=650.0,
        trs_lead_days=18,
    ),
)

DATA_START = pd.Period("2017-01", freq="M")
DATA_END = pd.Period("2022-09", freq="M")

DATA_START_DAY = date(2017, 1, 1)
DATA_END_DAY = date(2022, 9, 25)


def _shock_envelope(days_from_shock: np.ndarray, ramp_days: int = 45,
                    decay_days: int = 270) -> np.ndarray:
    """Smooth rise to ~1 at peak, slow decay; returns values in roughly [0, 1]."""
    rising = 1.0 / (1.0 + np.exp(-days_from_shock / max(1.0, ramp_days / 4)))
    decay = np.exp(-np.maximum(0, days_from_shock - 30) / decay_days)
    envelope = rising * decay
    if envelope.max() > 0:
        envelope = envelope / envelope.max()
    return envelope


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Standard AR(1) process; the initial value draws from the stationary noise."""
    eps = rng.normal(0, sigma, size=n)
    out = np.zeros(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


def _monthly_unit_price(spec: CaseSpec, months: pd.PeriodIndex,
                        rng: np.random.Generator) -> np.ndarray:
    """Piecewise baseline + shock + post-shock decay + seasonal + AR(1) noise."""
    days_from_shock = np.array(
        [(p.start_time.date() - spec.shock_date).days for p in months]
    )
    envelope = _shock_envelope(days_from_shock)

    peak_lift = spec.shock_peak_multiplier - 1.0
    long_run_lift = spec.post_shock_multiplier - 1.0
    long_run_envelope = (envelope > 0) * np.minimum(
        envelope, np.where(days_from_shock > 90, 1.0, 0.0)
    )
    shock_component = (
        peak_lift * envelope - long_run_lift * long_run_envelope + long_run_lift * (days_from_shock >= 0).astype(float) * (1 - np.exp(-np.maximum(0, days_from_shock) / 540))
    )

    month_of_year = np.array([p.month for p in months])
    seasonal = 0.025 * np.sin(2 * np.pi * (month_of_year - 1) / 12)

    noise = _ar1(len(months), rho=0.55, sigma=0.025, rng=rng)
    multiplier = 1.0 + seasonal + shock_component + noise
    multiplier = np.maximum(multiplier, 0.5)
    return spec.baseline_unit_price * multiplier


def _monthly_weights(rng: np.random.Generator, n: int, log_mu: float,
                     log_sigma: float, rho: float = 0.45) -> np.ndarray:
    """Lognormal monthly weights with mild persistence."""
    z = _ar1(n, rho=rho, sigma=log_sigma, rng=rng)
    return np.exp(log_mu + z)


def generate_monthly(spec: CaseSpec, rng: np.random.Generator) -> pd.DataFrame:
    months = pd.period_range(DATA_START, DATA_END, freq="M")
    n = len(months)

    target_unit_price = _monthly_unit_price(spec, months, rng)
    target_weight = _monthly_weights(rng, n, spec.target_weight_log_mu, spec.target_weight_log_sigma)
    target_value = target_weight * target_unit_price

    rows = []
    for i, period in enumerate(months):
        ym = f"{period.year}-{period.month:02d}"
        rows.append({
            "year_month": ym,
            "hs_code": spec.hs_code,
            "origin_country": spec.target_country,
            "reported_weight": round(float(target_weight[i]), 2),
            "taxable_value_usd": round(float(target_value[i]), 2),
        })

    # Other-country rows: not directly affected by the shock; baseline-priced with
    # mild noise. This makes the file look like a multi-country portal extract.
    for country in spec.other_countries:
        weights = _monthly_weights(rng, n, spec.other_weight_log_mu, spec.other_weight_log_sigma, rho=0.35)
        price_noise = _ar1(n, rho=0.4, sigma=0.06, rng=rng)
        prices = spec.baseline_unit_price * (1.0 + price_noise)
        prices = np.maximum(prices, spec.baseline_unit_price * 0.4)
        values = weights * prices
        for i, period in enumerate(months):
            ym = f"{period.year}-{period.month:02d}"
            rows.append({
                "year_month": ym,
                "hs_code": spec.hs_code,
                "origin_country": country,
                "reported_weight": round(float(weights[i]), 2),
                "taxable_value_usd": round(float(values[i]), 2),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["year_month", "origin_country"]).reset_index(drop=True)
    return df


RATIONALE_BANDS = [
    (0, 150, "baseline"),
    (150, 300, "low risk"),
    (300, 500, "elevated risk signals"),
    (500, 700, "regulation tightening"),
    (700, 1000, "acute spike"),
]


def _label_rationale(score: float) -> str:
    for lo, hi, label in RATIONALE_BANDS:
        if lo <= score < hi:
            return label
    return "acute spike"


def generate_daily_trs(spec: CaseSpec, rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range(DATA_START_DAY, DATA_END_DAY, freq="D")
    n = len(days)
    days_from_shock = np.array(
        [(d.date() - spec.shock_date).days for d in days]
    )

    # Sigmoid that rises trs_lead_days before the shock and stays elevated
    # for a few months before slowly returning toward baseline
    centered = days_from_shock + spec.trs_lead_days
    rising = 1.0 / (1.0 + np.exp(-centered / 4.0))
    decay = np.exp(-np.maximum(0, days_from_shock - 120) / 90.0)
    lead = rising * decay

    noise = _ar1(n, rho=0.78, sigma=35.0, rng=rng)
    level = spec.trs_baseline + (spec.trs_post_shock - spec.trs_baseline) * lead
    score = np.clip(level + noise, 0, 999)

    article_count = rng.poisson(lam=8 + 4 * lead, size=n).astype(int)
    risk_signal = rng.binomial(
        np.maximum(article_count, 1),
        np.clip(0.05 + 0.65 * lead, 0.0, 0.9),
    ).astype(int)
    country_mention = rng.binomial(
        np.maximum(article_count, 1),
        np.clip(0.25 + 0.4 * lead, 0.0, 0.95),
    ).astype(int)
    positive_signal = rng.binomial(
        np.maximum(article_count, 1),
        np.clip(0.10 - 0.05 * lead, 0.0, 0.4),
    ).astype(int)

    return pd.DataFrame({
        "date": [d.strftime("%Y.%m.%d") for d in days],
        "trs_score": np.round(score, 1),
        "rationale": [_label_rationale(s) for s in score],
        "article_count": article_count,
        "country_mention_count": country_mention,
        "risk_signal_count": risk_signal,
        "positive_signal_count": positive_signal,
    })


def write_dataset(spec: CaseSpec) -> None:
    rng = np.random.default_rng(42 + sum(ord(c) for c in spec.name))
    print(f"Generating {spec.name}: {spec.hs_code}, target {spec.target_country}, shock {spec.shock_date}")

    monthly = generate_monthly(spec, rng)
    trs = generate_daily_trs(spec, rng)

    out_dir = SAMPLE_ROOT / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = out_dir / "import_monthly_2017_2022.csv"
    trs_path = out_dir / "trs_final_2017_2022.csv"
    monthly.to_csv(monthly_path, index=False, encoding="utf-8")
    trs.to_csv(trs_path, index=False, encoding="utf-8-sig")

    print(f"  {monthly_path.relative_to(REPO_ROOT)} ({len(monthly)} rows)")
    print(f"  {trs_path.relative_to(REPO_ROOT)} ({len(trs)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=[s.name for s in SPECS] + ["all"], default="all")
    args = parser.parse_args()

    print("Writing synthetic sample CSVs (not real trade-statistics data).")
    targets = SPECS if args.dataset == "all" else [s for s in SPECS if s.name == args.dataset]
    for spec in targets:
        write_dataset(spec)


if __name__ == "__main__":
    main()
