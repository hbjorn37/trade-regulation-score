"""Merge weekly TRS with weekly imports and derive feature columns."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from trs_pipeline._config import load_config, parse_dataset_arg, week_sort_key
from trs_pipeline._paths import INTERMEDIATE_ROOT


def create_trs_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trs_pct_change"] = df["trs_mean"].pct_change()
    df["trs_max_pct_change"] = df["trs_max"].pct_change()

    df["trs_ma4"] = df["trs_mean"].rolling(window=4, min_periods=1).mean()
    df["trs_ma12"] = df["trs_mean"].rolling(window=12, min_periods=1).mean()

    df["trs_deviation_ma4"] = df["trs_mean"] - df["trs_ma4"]
    df["trs_deviation_ma12"] = df["trs_mean"] - df["trs_ma12"]
    df["trs_deviation_rate_ma4"] = df["trs_deviation_ma4"] / df["trs_ma4"].replace(0, np.nan)

    df["trs_spike"] = ((df["trs_pct_change"] > 0.5) | (df["trs_mean"] > 500)).astype(int)
    df["trs_high_risk"] = (df["trs_mean"] > 600).astype(int)
    df["trs_crisis"] = (df["trs_mean"] > 800).astype(int)

    for lag in [1, 2, 3, 4]:
        df[f"trs_mean_lag{lag}"] = df["trs_mean"].shift(lag)
        df[f"trs_max_lag{lag}"] = df["trs_max"].shift(lag)
        df[f"trs_pct_change_lag{lag}"] = df["trs_pct_change"].shift(lag)

    df["trs_spike_cum4w"] = df["trs_spike"].rolling(window=4, min_periods=1).sum()

    df["risk_signal_ratio"] = df["risk_signal_count"] / df["total_article_count"].replace(0, np.nan)
    df["country_mention_ratio"] = df["country_mention_count"] / df["total_article_count"].replace(0, np.nan)

    df["trs_volatility"] = df["trs_std"] / df["trs_mean"].replace(0, np.nan)

    for col in ["trs_spike", "trs_high_risk", "trs_spike_cum4w", "risk_signal_ratio"]:
        df[f"{col}_lag1"] = df[col].shift(1)

    return df


def merge_trs_with_import(trs_df: pd.DataFrame, import_df: pd.DataFrame) -> pd.DataFrame:
    import_df = import_df.copy()
    import_df["import_unit_price"] = (
        import_df["taxable_value_usd"] / import_df["reported_weight"]
    )
    import_df["import_unit_price"] = import_df["import_unit_price"].replace(
        [np.inf, -np.inf], np.nan
    )
    import_df["weight_log"] = np.log1p(import_df["reported_weight"])
    import_df["value_log"] = np.log1p(import_df["taxable_value_usd"])

    merged = pd.merge(trs_df, import_df, on="year_week", how="inner")
    merged["_sort_key"] = merged["year_week"].apply(week_sort_key)
    merged = (
        merged.sort_values("_sort_key")
        .drop("_sort_key", axis=1)
        .reset_index(drop=True)
    )

    merged = create_trs_derived_features(merged)

    merged["import_weight_pct_change"] = merged["reported_weight"].pct_change()
    merged["import_value_pct_change"] = merged["taxable_value_usd"].pct_change()
    merged["import_unit_price_pct_change"] = merged["import_unit_price"].pct_change()
    merged["import_weight_ma4"] = merged["reported_weight"].rolling(window=4, min_periods=1).mean()
    merged["import_value_ma4"] = merged["taxable_value_usd"].rolling(window=4, min_periods=1).mean()

    merged = merged.iloc[4:].reset_index(drop=True)
    return merged


def plot_trs_validation(
    merged: pd.DataFrame, output_path: Path, target_col: str = "import_unit_price", maxlag: int = 8
) -> None:
    if "trs_mean" not in merged.columns or target_col not in merged.columns:
        return
    test_data = merged[[target_col, "trs_mean"]].dropna()
    if len(test_data) < maxlag + 5:
        return

    granger_pvalues: dict[int, float] = {}
    try:
        gc = grangercausalitytests(test_data[[target_col, "trs_mean"]], maxlag=maxlag, verbose=False)
        for lag in range(1, maxlag + 1):
            granger_pvalues[lag] = gc[lag][0]["ssr_ftest"][1]
    except Exception:
        pass

    correlations: dict[int, float] = {}
    for lag in range(1, maxlag + 1):
        correlations[lag] = merged[target_col].corr(merged["trs_mean"].shift(lag))

    lags = list(range(1, maxlag + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("TRS Leading Indicator Validation (TRS to Import Unit Price)", fontsize=12)

    pvals = [granger_pvalues.get(l, np.nan) for l in lags]
    colors = ["#27AE60" if p < 0.05 else "#E67E22" for p in pvals]
    ax1.bar(lags, pvals, color=colors, edgecolor="white", width=0.6)
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax1.set_xlabel("Lag (weeks)")
    ax1.set_ylabel("p-value")
    ax1.set_title("(a) Granger Causality")
    ax1.set_xticks(lags)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    corrs = [correlations.get(l, np.nan) for l in lags]
    colors2 = ["#3498DB" if c > 0 else "#E74C3C" for c in corrs]
    ax2.bar(lags, corrs, color=colors2, edgecolor="white", width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Lag (weeks)")
    ax2.set_ylabel("Correlation")
    ax2.set_title("(b) Lagged TRS Correlation")
    ax2.set_xticks(lags)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def run(dataset: str, cfg: SimpleNamespace) -> None:
    inter_dir = INTERMEDIATE_ROOT / dataset
    inter_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step03_merge_features] dataset={dataset}")

    trs_file = inter_dir / "trs_weekly.csv"
    if not trs_file.exists():
        print(f"  TRS file not found: {trs_file} -- run step02 first")
        return
    trs_weekly = pd.read_csv(trs_file, encoding="utf-8-sig")
    trs_weekly["_sort_key"] = trs_weekly["year_week"].apply(week_sort_key)
    trs_weekly = (
        trs_weekly.sort_values("_sort_key")
        .drop("_sort_key", axis=1)
        .reset_index(drop=True)
    )

    import_file = inter_dir / "import_weekly.csv"
    if not import_file.exists():
        print(f"  Import file not found: {import_file} -- run step01 first")
        return
    import_df = pd.read_csv(import_file, encoding="utf-8")

    merged = merge_trs_with_import(trs_weekly, import_df)

    output_csv = inter_dir / "merged.csv"
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(
        f"  Saved: {output_csv.name} "
        f"({merged['year_week'].iloc[0]} to {merged['year_week'].iloc[-1]}, {len(merged)} rows)"
    )

    plot_trs_validation(merged, inter_dir / "trs_validation.png")

    shock_week = cfg.shock_week
    pre = merged[merged["year_week"] < shock_week]
    post = merged[merged["year_week"] >= shock_week]
    print(f"\n  Stats for {cfg.item_title}")
    print(f"    TRS mean: {merged['trs_mean'].mean():.1f}")
    print(f"    TRS max:  {merged['trs_max'].max():.1f}")
    print(f"    Unit price mean: {merged['import_unit_price'].mean():.4f}")
    print(f"    Pre-shock weeks (< {shock_week}): {len(pre)}")
    print(f"    Post-shock weeks (>= {shock_week}): {len(post)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="semi | urea | all (default)")
    parser.add_argument("--data-root", default=None, type=Path)
    args = parser.parse_args()

    for dataset in parse_dataset_arg(args):
        cfg = load_config(dataset)
        run(dataset, cfg)


if __name__ == "__main__":
    main()
