"""Plot data insight and BASE-vs-LLM result insight from eval.csv / forecast.csv."""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trs_pipeline._config import load_config, parse_dataset_arg
from trs_pipeline._paths import INTERMEDIATE_ROOT, OUTPUT_ROOT

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11})

C_BASE = "#3498DB"
C_LLM = "#E74C3C"
C_ACTUAL = "#2C3E50"
C_IMPROVE = "#27AE60"
C_WORSE = "#E67E22"
C_TRS = "#9B59B6"
C_SHOCK = "#E74C3C"


def week_to_date(week_str: str) -> datetime | None:
    try:
        year, week = week_str.split("_")
        return datetime.strptime(f"{year}-W{int(week):02d}-1", "%G-W%V-%u")
    except Exception:
        return None


def parse_metric_value(value_str):
    if pd.isna(value_str):
        return np.nan
    s = str(value_str)
    if "(" in s:
        return float(s.split("(")[0].strip())
    try:
        return float(s.replace("%", ""))
    except Exception:
        return np.nan


def parse_improvement(value_str):
    if pd.isna(value_str):
        return None
    s = str(value_str)
    m = re.search(r"\(([-+]?\d+\.?\d*)%?\)", s)
    return float(m.group(1)) if m else None


def load_eval_data(output_dir: Path) -> pd.DataFrame:
    for name in ["eval_all.csv", "eval.csv"]:
        path = output_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"eval file not found in {output_dir}")

    df = pd.read_csv(path, encoding="utf-8")
    for col in ["mae", "rmse", "mape"]:
        if col in df.columns:
            df[f"{col}_val"] = df[col].apply(parse_metric_value)
            df[f"{col}_imp"] = df[col].apply(parse_improvement)
    if "direction_accuracy" in df.columns:
        df["direction_val"] = df["direction_accuracy"].apply(
            lambda x: float(str(x).replace("%", "")) if pd.notna(x) else np.nan
        )
    return df


def load_forecast_data(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "forecast.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8")
    period_col = "period" if "period" in df.columns else "year_week"
    if period_col in df.columns:
        df["_date"] = df[period_col].apply(week_to_date)
        df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    return df


def load_merged_data(inter_dir: Path) -> pd.DataFrame | None:
    path = inter_dir / "merged.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["_date"] = df["year_week"].apply(week_to_date)
    return df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)


def build_base_llm_pairs(eval_df: pd.DataFrame) -> pd.DataFrame:
    base_df = eval_df[~eval_df["model"].str.startswith("LLM")].copy()
    rows = []
    for _, br in base_df.iterrows():
        model = br["model"]
        lr = eval_df[eval_df["model"] == f"LLM_{model}"]
        if lr.empty:
            continue
        lr = lr.iloc[0]
        imp = lr["mae_imp"] if lr["mae_imp"] is not None else 0
        rows.append({
            "model": model,
            "base_mae": br["mae_val"],
            "llm_mae": lr["mae_val"],
            "improvement": -imp,
        })
    return pd.DataFrame(rows).sort_values("base_mae")


def plot_data_insight(merged_df: pd.DataFrame, cfg: SimpleNamespace, output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"{cfg.item_title} (HS {cfg.hs_code}) Data Insight", fontsize=15, y=1.02)

    shock_date = week_to_date(cfg.shock_week)
    train_end_date = week_to_date(cfg.train_end_week)

    ax1t = ax1.twinx()
    ax1.fill_betweenx(
        [merged_df["import_unit_price"].min() * 0.8, merged_df["import_unit_price"].max() * 1.15],
        shock_date, merged_df["_date"].max(),
        color=C_SHOCK, alpha=0.04, label="_nolegend_",
    )
    ax1.axvline(shock_date, color=C_SHOCK, ls="--", lw=1.5, alpha=0.7)
    ax1.axvline(train_end_date, color="gray", ls=":", lw=1.2, alpha=0.6)

    l1 = ax1.plot(merged_df["_date"], merged_df["import_unit_price"],
                  color=C_ACTUAL, lw=1.8, label="Import Unit Price")
    l2 = ax1t.plot(merged_df["_date"], merged_df["trs_mean"],
                   color=C_TRS, lw=1.2, alpha=0.7, label="TRS Mean")

    ax1.set_ylabel("Import Unit Price (USD/kg)")
    ax1t.set_ylabel("TRS Score", color=C_TRS)
    ax1t.tick_params(axis="y", labelcolor=C_TRS)

    ax1.annotate(
        f"{cfg.shock_label}\n({cfg.shock_week})",
        xy=(shock_date, merged_df["import_unit_price"].max() * 0.95),
        fontsize=8, color=C_SHOCK, ha="center",
    )
    ax1.annotate(
        "Train | Test",
        xy=(train_end_date, merged_df["import_unit_price"].min() * 0.85),
        fontsize=8, color="gray", ha="center",
    )

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    ax1.set_title("(a) Import Unit Price vs TRS Time Series", fontsize=12)
    ax1.grid(True, alpha=0.2)

    lags = list(range(0, 9))
    lag_corrs = []
    for lag in lags:
        if lag == 0:
            c = merged_df["trs_mean"].corr(merged_df["import_unit_price"])
        else:
            c = merged_df["trs_mean"].shift(lag).corr(merged_df["import_unit_price"])
        lag_corrs.append(c)

    bar_colors = [C_IMPROVE if c > 0 else C_WORSE for c in lag_corrs]
    bars = ax2.bar(lags, lag_corrs, color=bar_colors, alpha=0.8, edgecolor="white", width=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    for bar, val in zip(bars, lag_corrs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.01 if val >= 0 else -0.025),
            f"{val:.3f}", ha="center", fontsize=7.5,
        )
    ax2.set_xlabel("Lag (weeks)")
    ax2.set_ylabel("Correlation")
    ax2.set_title("(b) Lagged Correlation: TRS to Import Unit Price", fontsize=12)
    ax2.set_xticks(lags)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = output_dir / "data_insight.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def plot_result_insight(eval_df: pd.DataFrame, forecast_df: pd.DataFrame | None,
                        cfg: SimpleNamespace, output_dir: Path) -> None:
    pairs = build_base_llm_pairs(eval_df)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    fig.suptitle(
        f"{cfg.item_title} (HS {cfg.hs_code}) Experiment Insight -- BASE vs LLM(TRS)",
        fontsize=15, y=1.01,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    models = pairs["model"].values
    x = np.arange(len(models))
    w = 0.35
    ax_a.bar(x - w / 2, pairs["base_mae"], w, label="BASE", color=C_BASE, alpha=0.85)
    ax_a.bar(x + w / 2, pairs["llm_mae"], w, label="LLM(TRS)", color=C_LLM, alpha=0.85)
    if len(pairs) > 0:
        max_mae = max(pairs["base_mae"].max(), pairs["llm_mae"].max())
        for i, (bv, lv, imp) in enumerate(zip(pairs["base_mae"], pairs["llm_mae"], pairs["improvement"])):
            y_pos = max(bv, lv) + max_mae * 0.03
            color = C_IMPROVE if imp > 0 else C_WORSE
            ax_a.text(i, y_pos, f"{imp:+.1f}%", ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax_a.set_ylabel("MAE")
    ax_a.set_title("(a) Model-wise MAE Comparison", fontsize=12)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.2, axis="y")

    ax_b = fig.add_subplot(gs[0, 1])
    sorted_pairs = pairs.sort_values("improvement")
    bar_colors = [C_IMPROVE if v > 0 else C_WORSE for v in sorted_pairs["improvement"]]
    bars_imp = ax_b.barh(sorted_pairs["model"], sorted_pairs["improvement"],
                         color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)
    ax_b.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars_imp, sorted_pairs["improvement"]):
        offset = 0.8 if val >= 0 else -0.8
        ha = "left" if val >= 0 else "right"
        ax_b.text(val + offset, bar.get_y() + bar.get_height() / 2,
                  f"{val:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax_b.set_xlabel("MAE Improvement (%)")
    ax_b.set_title("(b) MAE Improvement with LLM(TRS)", fontsize=12)
    ax_b.grid(True, alpha=0.2, axis="x")

    ax_c = fig.add_subplot(gs[1, :])
    if forecast_df is not None and "_date" in forecast_df.columns and len(pairs) > 0:
        shock_date = week_to_date(cfg.shock_week)
        ax_c.plot(forecast_df["_date"], forecast_df["actual"],
                  color=C_ACTUAL, lw=2.2, label="Actual", zorder=10)
        best = pairs.loc[pairs["improvement"].idxmax()]
        model_name = best["model"]
        col_base = f"{model_name}_pred"
        col_llm = f"LLM_{model_name}_pred"
        if col_base in forecast_df.columns:
            ax_c.plot(forecast_df["_date"], forecast_df[col_base], color=C_BASE, lw=1.4, ls="--", alpha=0.8,
                      label=f"{model_name} BASE (MAE {best['base_mae']:.2f})")
        if col_llm in forecast_df.columns:
            ax_c.plot(forecast_df["_date"], forecast_df[col_llm], color=C_LLM, lw=1.4, ls="-.", alpha=0.8,
                      label=f"{model_name} + TRS (MAE {best['llm_mae']:.2f}, {best['improvement']:+.1f}%)")
        ax_c.axvline(shock_date, color=C_SHOCK, ls="--", lw=1.5, alpha=0.6)
        ax_c.annotate(
            cfg.shock_label,
            xy=(shock_date, forecast_df["actual"].max() * 0.95),
            fontsize=8, color=C_SHOCK, ha="center",
        )
        ax_c.set_ylabel("Import Unit Price (USD/kg)")
        ax_c.set_title("(c) Best-Improved Model Forecast -- BASE vs +TRS", fontsize=12)
        ax_c.legend(loc="upper left", fontsize=9)
        ax_c.grid(True, alpha=0.2)

    plt.tight_layout()
    out = output_dir / "result_insight.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def run(dataset: str, cfg: SimpleNamespace) -> None:
    output_dir = OUTPUT_ROOT / dataset
    inter_dir = INTERMEDIATE_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step05_plot_evaluation] dataset={dataset}")

    try:
        eval_df = load_eval_data(output_dir)
        print(f"  Loaded eval ({eval_df['model'].nunique()} models)")
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    forecast_df = load_forecast_data(output_dir)
    merged_df = load_merged_data(inter_dir)

    if merged_df is not None:
        plot_data_insight(merged_df, cfg, output_dir)
    else:
        print("  merged.csv missing -- skipping data_insight")
    plot_result_insight(eval_df, forecast_df, cfg, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="semi | urea | all (default)")
    args = parser.parse_args()
    for dataset in parse_dataset_arg(args):
        cfg = load_config(dataset)
        run(dataset, cfg)


if __name__ == "__main__":
    main()
