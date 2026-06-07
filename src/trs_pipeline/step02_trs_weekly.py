"""Aggregate daily TRS records into ISO weeks and plot the result."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from trs_pipeline._config import (
    load_config,
    parse_dataset_arg,
    resolve_data_dir,
    week_sort_key,
)
from trs_pipeline._paths import INTERMEDIATE_ROOT


def aggregate_trs_to_weekly(trs_file: Path) -> pd.DataFrame | None:
    if not trs_file.exists():
        print(f"  TRS file not found: {trs_file}")
        return None

    print(f"  Reading: {trs_file.name}")
    df = pd.read_csv(trs_file, encoding="utf-8-sig")
    print(f"  Total TRS records: {len(df)}")

    df["_date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    iso = df["_date"].dt.isocalendar()
    df["year_week"] = iso["year"].astype(str) + "_" + iso["week"].astype(str).str.zfill(2)

    for col, default in (
        ("risk_signal_count", 0),
        ("country_mention_count", 0),
    ):
        if col not in df.columns:
            df[col] = default

    weekly = (
        df.groupby("year_week")
        .agg(
            trs_mean=("trs_score", "mean"),
            trs_max=("trs_score", "max"),
            trs_min=("trs_score", "min"),
            trs_std=("trs_score", "std"),
            trs_days=("trs_score", "count"),
            total_article_count=("article_count", "sum"),
            risk_signal_count=("risk_signal_count", "sum"),
            country_mention_count=("country_mention_count", "sum"),
        )
        .reset_index()
    )

    weekly["_sort_key"] = weekly["year_week"].apply(week_sort_key)
    weekly = (
        weekly.sort_values("_sort_key")
        .drop("_sort_key", axis=1)
        .reset_index(drop=True)
    )
    return weekly


def plot_trs_weekly(
    df: pd.DataFrame,
    output_png: Path,
    title_suffix: str,
    shock_weeks: list[str],
    shock_label: str,
    country_mention_label: str,
) -> None:
    x = range(len(df))
    labels = df["year_week"].tolist()
    step = max(1, len(labels) // 12)
    tick_indices = list(range(0, len(labels), step))
    if len(labels) - 1 not in tick_indices:
        tick_indices.append(len(labels) - 1)
    tick_labels = [labels[i] for i in tick_indices]

    shock_set = set(shock_weeks)
    shock_idx = [i for i, w in enumerate(labels) if w in shock_set]
    shock_start = min(shock_idx) if shock_idx else 0
    shock_end = max(shock_idx) + 1 if shock_idx else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Weekly TRS ({title_suffix})", fontsize=13)

    for ax in (ax1, ax2):
        if shock_end > shock_start:
            ax.axvspan(shock_start, shock_end, alpha=0.2, color="red", label=shock_label)
    ax1.legend(loc="upper right", fontsize=8)

    ax1.fill_between(
        x,
        df["trs_min"].fillna(df["trs_mean"]),
        df["trs_max"],
        alpha=0.25,
        color="steelblue",
    )
    ax1.plot(x, df["trs_mean"], color="steelblue", linewidth=1.2, label="TRS Mean")
    ax1.plot(x, df["trs_max"], color="coral", linewidth=0.6, alpha=0.8, label="TRS Max")
    ax1.plot(
        x,
        df["trs_min"].fillna(df["trs_mean"]),
        color="seagreen",
        linewidth=0.6,
        alpha=0.8,
        label="TRS Min",
    )
    ax1.set_ylabel("TRS Score")
    ax1.set_title("Trade Regulation Score (TRS)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(x, df["total_article_count"], color="C0", linewidth=0.9, label="Total Articles")
    ax2.plot(
        x,
        df["country_mention_count"],
        color="C1",
        linewidth=0.9,
        label=country_mention_label,
    )
    ax2.plot(
        x,
        df["risk_signal_count"],
        color="C2",
        linewidth=0.9,
        label="Risk Signal Articles",
    )
    ax2.set_ylabel("Article Count")
    ax2.set_xlabel("Year_Week")
    ax2.set_title("TRS-Related Article Counts")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.xticks(tick_indices, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def run(dataset: str, cfg: SimpleNamespace, data_root: Path | None) -> None:
    data_dir = resolve_data_dir(data_root, dataset)
    out_dir = INTERMEDIATE_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step02_trs_weekly] dataset={dataset}")

    trs_file = data_dir / cfg.files.trs
    weekly = aggregate_trs_to_weekly(trs_file)
    if weekly is None:
        return

    weekly = weekly[
        weekly["year_week"].apply(week_sort_key) <= week_sort_key(cfg.data_end_week)
    ].reset_index(drop=True)
    if weekly.empty:
        print(f"  No data in window (<= {cfg.data_end_week})")
        return

    start_week = weekly["year_week"].iloc[0]
    end_week = cfg.data_end_week

    csv_path = out_dir / "trs_weekly.csv"
    weekly.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {csv_path.name} ({start_week} to {end_week}, {len(weekly)} weeks)")

    png_path = out_dir / "trs_weekly.png"
    plot_trs_weekly(
        weekly,
        png_path,
        f"{start_week} to {end_week}",
        cfg.shock_weeks,
        cfg.shock_label,
        cfg.country_mention_label,
    )
    print(f"  Saved: {png_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="semi | urea | all (default)")
    parser.add_argument("--data-root", default=None, type=Path)
    args = parser.parse_args()

    for dataset in parse_dataset_arg(args):
        cfg = load_config(dataset)
        run(dataset, cfg, args.data_root)


if __name__ == "__main__":
    main()
