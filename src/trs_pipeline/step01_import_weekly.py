"""Aggregate daily import records into ISO weeks and linearly interpolate gaps."""

import argparse
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from trs_pipeline._config import load_config, parse_dataset_arg, resolve_data_dir
from trs_pipeline._paths import INTERMEDIATE_ROOT


def iso_weeks_in_year(year: int) -> int:
    return date(year, 12, 28).isocalendar()[1]


def parse_week(week_str: str) -> tuple[int, int]:
    y, w = week_str.split("_")
    return int(y), int(w)


def build_full_weeks(data_start: str, data_end: str) -> list[str]:
    min_yr, min_wk = parse_week(data_start)
    max_yr, max_wk = parse_week(data_end)
    full = []
    for y in range(min_yr, max_yr + 1):
        w_start = min_wk if y == min_yr else 1
        w_end = min(max_wk, iso_weeks_in_year(y)) if y == max_yr else iso_weeks_in_year(y)
        for w in range(w_start, w_end + 1):
            full.append(f"{y}_{w:02d}")
    return full


def build_weekly(df: pd.DataFrame, full_weeks: list[str], origin_country: str) -> pd.DataFrame:
    sub = cast(pd.DataFrame, df[df["origin_country"] == origin_country].copy())
    sub["year"] = sub["clearance_date"].dt.isocalendar().year
    sub["week"] = sub["clearance_date"].dt.isocalendar().week
    sub["year_week"] = sub["year"].astype(str) + "_" + sub["week"].astype(str).str.zfill(2)

    weekly = sub.groupby("year_week", as_index=False).agg(
        reported_weight=("reported_weight", "sum"),
        taxable_value_usd=("taxable_value_usd", "sum"),
    )

    full_df = pd.DataFrame({"year_week": full_weeks})
    merged = cast(pd.DataFrame, full_df.merge(weekly, on="year_week", how="left"))
    merged["reported_weight"] = merged["reported_weight"].interpolate(
        method="linear", limit_area="inside"
    )
    merged["taxable_value_usd"] = merged["taxable_value_usd"].interpolate(
        method="linear", limit_area="inside"
    )
    return merged


def plot_weekly(df: pd.DataFrame, output_png: Path, shock_weeks: list[str], shock_label: str) -> None:
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Weekly Imports ({output_png.stem})", fontsize=12)

    for ax in (ax1, ax2):
        if shock_end > shock_start:
            ax.axvspan(shock_start, shock_end, alpha=0.2, color="red", label=shock_label)
    ax1.legend(loc="upper right", fontsize=8)

    ax1.plot(x, df["reported_weight"], color="C0", linewidth=0.8)
    ax1.set_ylabel("Reported Weight")
    ax1.set_title("Reported Weight")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, df["taxable_value_usd"], color="C1", linewidth=0.8)
    ax2.set_ylabel("Taxable Value (USD)")
    ax2.set_title("Taxable Value")
    ax2.grid(True, alpha=0.3)

    plt.xticks(tick_indices, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def run(dataset: str, cfg: SimpleNamespace, data_root: Path | None) -> None:
    data_dir = resolve_data_dir(data_root, dataset)
    out_dir = INTERMEDIATE_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = data_dir / cfg.files.import_csv
    output_csv = out_dir / "import_weekly.csv"
    output_png = out_dir / "import_weekly.png"

    print(f"[step01_import_weekly] dataset={dataset}")
    print(f"  Input: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8")
    df["clearance_date"] = pd.to_datetime(df["clearance_date"].astype(str), format="%Y%m%d")

    full_weeks = build_full_weeks(cfg.data_start_week, cfg.data_end_week)
    weekly = build_weekly(df, full_weeks, cfg.origin_country)

    weekly.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"  Saved: {output_csv} ({len(weekly)} weeks)")

    plot_weekly(weekly, output_png, cfg.shock_weeks, cfg.shock_label)
    print(f"  Saved: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="semi | urea | all (default)")
    parser.add_argument("--data-root", default=None, type=Path,
                        help="Override data root (defaults to data/sample)")
    args = parser.parse_args()

    for dataset in parse_dataset_arg(args):
        cfg = load_config(dataset)
        run(dataset, cfg, args.data_root)


if __name__ == "__main__":
    main()
