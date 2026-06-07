"""Supply-shock early warning system on daily TRS scores."""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from math import erfc, sqrt
from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trs_pipeline._config import load_config, parse_dataset_arg, resolve_data_dir
from trs_pipeline._paths import INTERMEDIATE_ROOT, OUTPUT_ROOT

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


def week_to_date(week_str: str) -> datetime:
    year, week = week_str.split("_")
    return datetime.strptime(f"{year}-W{int(week):02d}-1", "%G-W%V-%u")


class EarlyWarningSystem:
    THRESHOLDS = {"Caution": 200, "Alert": 400, "Risk": 600, "Critical": 800}
    THRESHOLD_COLORS = {
        "Caution": "gold",
        "Alert": "darkorange",
        "Risk": "red",
        "Critical": "darkred",
    }

    def __init__(self, dataset: str, cfg: SimpleNamespace, data_root: Path | None):
        self.dataset = dataset
        self.cfg = cfg
        self.data_root = data_root
        self.inter_dir = INTERMEDIATE_ROOT / dataset
        self.out_dir = OUTPUT_ROOT / dataset / "early_warning"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.shock_week = cfg.shock_week
        self.shock_date = pd.Timestamp(week_to_date(self.shock_week))
        self.shock_label = cfg.shock_label
        self.item_title = cfg.item_title
        self.hs_code = cfg.hs_code
        self.target_col = "import_unit_price"
        self.score_col = "trs_score"

    def load_daily_trs(self) -> pd.DataFrame | None:
        data_dir = resolve_data_dir(self.data_root, self.dataset)
        f = data_dir / self.cfg.files.trs
        if not f.exists():
            print(f"  Daily TRS file not found: {f}")
            return None
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "date" in df.columns:
            try:
                df["_date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
            except Exception:
                df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
        print(f"  Daily TRS: {f.name} ({len(df)} days)")
        return df

    def load_weekly_trs(self) -> pd.DataFrame | None:
        path = self.inter_dir / "trs_weekly.csv"
        if not path.exists():
            print(f"  Weekly TRS file not found: {path}")
            return None
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["_date"] = pd.to_datetime(df["year_week"].apply(week_to_date))
        df = df.sort_values("_date").reset_index(drop=True)
        print(f"  Weekly TRS: {path.name} ({len(df)} weeks)")
        return df

    def load_merged_data(self) -> pd.DataFrame | None:
        path = self.inter_dir / "merged.csv"
        if not path.exists():
            print(f"  merged.csv not found: {path}")
            return None
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["_date"] = pd.to_datetime(df["year_week"].apply(week_to_date))
        df = df.sort_values("_date").reset_index(drop=True)
        if self.target_col in df.columns:
            last_valid = df[self.target_col].last_valid_index()
            if last_valid is not None and last_valid < len(df) - 1:
                df = df.iloc[: last_valid + 1].reset_index(drop=True)
        print(f"  Merged: {path.name} ({len(df)} weeks)")
        return df

    def compute_threshold_metrics(self, daily_trs: pd.DataFrame) -> pd.DataFrame:
        WINDOW_DAYS = 90
        thresholds = list(range(100, 950, 50))
        pre = daily_trs[
            (daily_trs["_date"] >= self.shock_date - pd.Timedelta(days=WINDOW_DAYS))
            & (daily_trs["_date"] < self.shock_date)
        ]
        baseline = daily_trs[daily_trs["_date"] < self.shock_date - pd.Timedelta(days=WINDOW_DAYS)]
        baseline_days = len(baseline)

        rows = []
        for thresh in thresholds:
            pre_n = int((pre[self.score_col] >= thresh).sum())
            base_n = int((baseline[self.score_col] >= thresh).sum())
            base_rate = base_n / baseline_days if baseline_days > 0 else 0.0
            pre_rate = pre_n / WINDOW_DAYS
            first = pre[pre[self.score_col] >= thresh]["_date"].min() if pre_n > 0 else None
            lead_days = int((self.shock_date - first).days) if first is not None else 0
            denom = pre_n + base_n
            precision = pre_n / denom if denom > 0 else 0.0
            recall = pre_rate
            fa_per_year = (base_n / baseline_days * 365) if baseline_days > 0 else 0.0
            if base_rate > 0:
                lift = min(pre_rate / base_rate, 20.0)
            else:
                lift = 20.0 if pre_n > 0 else 1.0
            rows.append({
                "threshold": thresh,
                "lead_days": lead_days,
                "pre_alarms": pre_n,
                "base_alarms": base_n,
                "precision": precision,
                "recall": recall,
                "fa_per_year": fa_per_year,
                "base_rate": base_rate,
                "pre_rate": pre_rate,
                "lift": lift,
            })
        return pd.DataFrame(rows)

    def plot_overview(self, daily_trs: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28, height_ratios=[1.6, 1])

        ax1 = fig.add_subplot(gs[0, :])
        ma30 = daily_trs[self.score_col].rolling(30, min_periods=1).mean()
        ax1.fill_between(daily_trs["_date"], 0, daily_trs[self.score_col],
                         alpha=0.2, color="steelblue", label="TRS (Daily)")
        ax1.plot(daily_trs["_date"], ma30, color="navy", lw=2, label="30-Day Moving Average")
        for level, thresh in self.THRESHOLDS.items():
            ax1.axhline(thresh, color=self.THRESHOLD_COLORS[level], linestyle="--",
                        alpha=0.8, lw=1.2, label=f"{level} ({thresh})")
        ax1.axvline(self.shock_date, color="crimson", lw=3, label=self.shock_label)

        rise_start = self.shock_date - pd.Timedelta(weeks=12)
        pre_12w = daily_trs[(daily_trs["_date"] >= rise_start) & (daily_trs["_date"] < self.shock_date)]
        base_period = daily_trs[daily_trs["_date"] < rise_start]
        pre_mean = pre_12w[self.score_col].mean() if len(pre_12w) > 0 else 0
        normal_mean = base_period[self.score_col].mean() if len(base_period) > 0 else 1

        ax1.axvspan(rise_start, self.shock_date, alpha=0.07, color="red")
        mid = rise_start + (self.shock_date - rise_start) / 2
        if pre_mean > normal_mean * 1.2:
            text = f"12 Weeks Before Shock\nLeading Increase Window\n(Mean {pre_mean:.0f})"
            color = "darkred"
        else:
            text = f"12 Weeks Before Shock\nLimited Leading Increase\n(Mean {pre_mean:.0f})"
            color = "gray"
        ax1.annotate(text, xy=(mid, 850), fontsize=9, color=color, ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax1.set_title("Daily TRS and Early Warning Thresholds", fontsize=14, fontweight="bold")
        ax1.set_ylabel("TRS Score")
        ax1.set_ylim(0, 1050)
        ax1.legend(loc="upper left", fontsize=9, ncol=2)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax2 = fig.add_subplot(gs[1, 0])
        zoom = daily_trs[(daily_trs["_date"] >= self.shock_date - pd.Timedelta(days=120))
                         & (daily_trs["_date"] <= self.shock_date + pd.Timedelta(days=30))]
        colors = ["crimson" if d >= self.shock_date else "steelblue" for d in zoom["_date"]]
        ax2.bar(zoom["_date"], zoom[self.score_col], color=colors, alpha=0.82, width=1)
        for thresh in [self.THRESHOLDS["Risk"], self.THRESHOLDS["Critical"]]:
            ax2.axhline(thresh, color="red", linestyle=":", lw=1.2, alpha=0.7)
        ax2.axvline(self.shock_date, color="crimson", lw=2.5, linestyle="--")
        ax2.set_title("Zoomed TRS Around the Shock (120 Days Before)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("TRS Score")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax2.legend(handles=[
            mpatches.Patch(color="steelblue", alpha=0.82, label="Pre-Shock"),
            mpatches.Patch(color="crimson", alpha=0.82, label="Post-Shock"),
        ], fontsize=9)

        ax3 = fig.add_subplot(gs[1, 1])
        daily_trs_c = daily_trs.copy()
        daily_trs_c["ym"] = daily_trs_c["_date"].dt.to_period("M")
        m_high = daily_trs_c[daily_trs_c[self.score_col] >= 600].groupby("ym").size()
        m_crit = daily_trs_c[daily_trs_c[self.score_col] >= 800].groupby("ym").size()
        months = pd.date_range(daily_trs_c["_date"].min(), daily_trs_c["_date"].max(), freq="MS")
        hc = [m_high.get(pd.Period(m, "M"), 0) for m in months]
        cc = [m_crit.get(pd.Period(m, "M"), 0) for m in months]
        ax3.bar(months, hc, width=20, alpha=0.6, color="orange", label="Risk (600+)")
        ax3.bar(months, cc, width=20, alpha=0.8, color="darkred", label="Critical (800+)")
        ax3.axvline(self.shock_date, color="crimson", lw=2.5)
        ax3.set_title("Monthly High-Risk Warning Days", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Days")
        ax3.legend(fontsize=9)
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.suptitle(f"{self.item_title} Supply Shock Early Warning -- Overview",
                     fontsize=15, fontweight="bold", y=1.01)
        out = self.out_dir / "overview.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {out.name}")

    def plot_signal_quality(self, daily_trs: pd.DataFrame) -> pd.DataFrame:
        metrics = self.compute_threshold_metrics(daily_trs)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Early Warning Performance (90-Day Pre-Shock Window)",
                     fontsize=13, fontweight="bold")

        def vlines(ax):
            for lv, th in self.THRESHOLDS.items():
                ax.axvline(th, color=self.THRESHOLD_COLORS[lv], linestyle="--",
                           alpha=0.5, lw=1, label=f"{lv}({th})")

        ax = axes[0]
        ax.plot(metrics["threshold"], metrics["precision"] * 100, "o-",
                color="green", lw=2, ms=4, label="Precision")
        ax.plot(metrics["threshold"], metrics["recall"] * 100, "s-",
                color="steelblue", lw=2, ms=4, label="Recall")
        vlines(ax)
        ax.set_xlabel("TRS Threshold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("(a) Precision and Recall")

        ax = axes[1]
        ax.plot(metrics["threshold"], metrics["fa_per_year"], "o-",
                color="crimson", lw=2, ms=4, label="False Alarm Days per Year")
        for lv, th in self.THRESHOLDS.items():
            row = metrics[metrics["threshold"] == th]
            if len(row) > 0:
                fa = row.iloc[0]["fa_per_year"]
                ax.axvline(th, color=self.THRESHOLD_COLORS[lv], linestyle="--", alpha=0.5, lw=1)
                ax.annotate(f"{lv}\n{fa:.1f} days", xy=(th, fa), xytext=(th + 20, fa + 1),
                            fontsize=7, color=self.THRESHOLD_COLORS[lv])
        ax.set_xlabel("TRS Threshold")
        ax.set_ylabel("Days/Year")
        ax.legend(fontsize=8)
        ax.set_title("(b) False Alarm Days per Year")

        ax = axes[2]
        ax.plot(metrics["threshold"], metrics["lift"], "o-", color="darkorange", lw=2, ms=5)
        ax.axhline(1.0, color="gray", linestyle=":", lw=1.5, label="Lift = 1 (No Signal)")
        for lv, th in self.THRESHOLDS.items():
            row = metrics[metrics["threshold"] == th]
            if len(row) > 0:
                lv_val = row.iloc[0]["lift"]
                ax.axvline(th, color=self.THRESHOLD_COLORS[lv], linestyle="--", alpha=0.5, lw=1)
                ax.annotate(f"{lv}\n{lv_val:.1f}x", xy=(th, lv_val), xytext=(th + 30, lv_val + 0.3),
                            fontsize=7, color=self.THRESHOLD_COLORS[lv])
        ax.set_xlabel("TRS Threshold")
        ax.set_ylabel("Lift (x)")
        ax.set_ylim(bottom=0)
        ax.set_title("(c) Lift = Pre-shock alarm rate / Baseline alarm rate")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out = self.out_dir / "signal_quality.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {out.name}")
        return metrics

    def plot_shock_trajectory(self, weekly_trs: pd.DataFrame, merged_df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("TRS Pattern Around the Shock", fontsize=13, fontweight="bold")

        ax = axes[0]
        W = 16
        shock_idx_arr = weekly_trs[weekly_trs["year_week"] == self.shock_week].index.tolist()
        if shock_idx_arr:
            si = shock_idx_arr[0]
            actual_post = min(W, len(weekly_trs) - si - 1)
            if si >= W and actual_post > 0:
                trs_arr = weekly_trs.iloc[si - W : si + actual_post + 1]["trs_mean"].values
                x = np.arange(-W, actual_post + 1)
                pre_mask = x < 0
                post_mask = x >= 0
                ax.fill_between(x[pre_mask], 0, trs_arr[pre_mask], alpha=0.25,
                                color="steelblue", label="Pre-Shock")
                ax.fill_between(x[post_mask], 0, trs_arr[post_mask], alpha=0.25,
                                color="crimson", label="Post-Shock")
                ax.plot(x, trs_arr, "o-", color="navy", lw=2, ms=4)
                ax.axvline(0, color="crimson", lw=2, linestyle="--",
                           label=f"Shock ({self.shock_week})")
                overall_mean = weekly_trs["trs_mean"].mean()
                ax.axhline(overall_mean, color="gray", linestyle=":", alpha=0.7,
                           label=f"Overall Mean ({overall_mean:.0f})")
                ax.axhline(600, color="red", linestyle=":", alpha=0.7,
                           label="Risk Threshold (600)")
                peak_x = int(x[np.argmax(trs_arr)])
                peak_v = float(np.max(trs_arr))
                direction = "before" if peak_x < 0 else ("after" if peak_x > 0 else "at")
                offset = -6 if peak_x < 0 else 2
                ax.annotate(f"Peak {peak_v:.0f}\n({abs(peak_x)} weeks {direction} shock)",
                            xy=(peak_x, peak_v), xytext=(peak_x + offset, peak_v + 50),
                            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                            fontsize=8, color="darkred")
        ax.set_xlabel(f"Weeks Relative to Shock (0 = {self.shock_week})")
        ax.set_ylabel("Mean TRS")
        ax.set_title(f"(a) Shock-Centered TRS Trajectory (±{W} Weeks)")
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[1]
        pre_13w = weekly_trs[(weekly_trs["_date"] >= self.shock_date - pd.Timedelta(weeks=13))
                             & (weekly_trs["_date"] < self.shock_date)]["trs_mean"].values
        normal = weekly_trs[weekly_trs["_date"] < self.shock_date - pd.Timedelta(weeks=13)]["trs_mean"].values

        ax.hist(normal, bins=20, alpha=0.55, color="steelblue", density=True,
                label=f"Baseline (n={len(normal)} weeks)")
        ax.hist(pre_13w, bins=10, alpha=0.7, color="crimson", density=True,
                label=f"13 Weeks Before Shock (n={len(pre_13w)} weeks)")
        if len(normal) > 0:
            mn = np.mean(normal)
            ax.axvline(mn, color="steelblue", lw=2, linestyle="--", label=f"Baseline Mean: {mn:.0f}")
        if len(pre_13w) > 0:
            mp = np.mean(pre_13w)
            ax.axvline(mp, color="crimson", lw=2, linestyle="--", label=f"Pre-shock Mean: {mp:.0f}")

        if len(normal) > 1 and len(pre_13w) > 1:
            n1, n2 = len(normal), len(pre_13w)
            m1, m2 = np.mean(normal), np.mean(pre_13w)
            s1, s2 = np.std(normal, ddof=1), np.std(pre_13w, ddof=1)
            se = sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
            t = (m2 - m1) / se if se > 0 else 0
            p = erfc(abs(t) / sqrt(2))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.annotate(f"Welch t = {t:.2f}\np-value = {p:.4f}  {sig}",
                        xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_xlabel("Mean TRS")
        ax.set_ylabel("Density")
        ax.set_title("(b) TRS Distribution: Baseline vs Pre-Shock")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out = self.out_dir / "shock_trajectory.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {out.name}")

    def plot_warning_simulation(self, merged_df: pd.DataFrame, threshold: int = 400) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Early Warning Simulation (Threshold {threshold})",
                     fontsize=13, fontweight="bold")

        dates = merged_df["_date"]
        trs = merged_df["trs_mean"]
        target = merged_df[self.target_col]
        alarm = trs >= threshold

        ax = axes[0]
        ax.plot(dates, target, "b-", lw=1.5, label="Import Unit Price")
        ax.axvline(self.shock_date, color="crimson", lw=2, linestyle="--", label="Shock Date")
        pre_mask = alarm & (dates < self.shock_date)
        if pre_mask.any():
            ax.scatter(dates[pre_mask], target[pre_mask], color="orange",
                       s=40, marker="^", zorder=5, label="Pre-Shock Warning")
        ax.set_ylabel("Import Unit Price")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"(a) Import Unit Price -- Weeks with Pre-shock TRS >= {threshold}")

        ax = axes[1]
        ax.plot(dates, trs, color="tomato", lw=1.5, alpha=0.85, label="Mean TRS")
        ax.fill_between(dates, threshold, trs, where=alarm & (dates < self.shock_date),
                        alpha=0.35, color="orange", label="Pre-Shock Warning")
        ax.fill_between(dates, threshold, trs, where=alarm & (dates >= self.shock_date),
                        alpha=0.12, color="gray", label="Post-Shock Warning")
        ax.axhline(threshold, color="orange", linestyle="--", lw=1.5, label=f"Threshold ({threshold})")
        ax.axvline(self.shock_date, color="crimson", lw=2, linestyle="--")
        pre_dates = dates[alarm & (dates < self.shock_date)]
        if len(pre_dates) > 0:
            first = pre_dates.min()
            lead = (self.shock_date - first).days
            ax.axvline(first, color="gold", lw=1.5, linestyle=":",
                       label=f"First Warning ({lead} days ahead)")
        ax.set_ylabel("TRS")
        ax.legend(fontsize=9, loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title("(b) TRS Trend -- Highlighted Warning Periods")

        plt.tight_layout()
        out = self.out_dir / "warning_simulation.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {out.name}")

    def generate_report(self, metrics: pd.DataFrame, daily_trs: pd.DataFrame) -> None:
        pre_90 = daily_trs[(daily_trs["_date"] >= self.shock_date - pd.Timedelta(days=90))
                           & (daily_trs["_date"] < self.shock_date)]
        pre_mean = pre_90[self.score_col].mean() if len(pre_90) > 0 else 0
        baseline = daily_trs[daily_trs["_date"] < self.shock_date - pd.Timedelta(days=90)]
        base_mean = baseline[self.score_col].mean() if len(baseline) > 0 else 0

        lines = [
            f"# {self.item_title} Early Warning Summary\n\n",
            f"- Shock event: {self.shock_label} ({self.shock_week})\n",
            f"- Pre-shock 90-day TRS mean: **{pre_mean:.1f}** (baseline mean: {base_mean:.1f})\n\n",
            "## Metric Definitions\n\n",
            "- **Lead time** = first alarm in the 90-day pre-shock window to shock (days)\n",
            "- **Recall** = pre-shock alarm days / 90\n",
            "- **Precision** = pre-shock alarms / (pre-shock alarms + baseline alarms)\n",
            "- **FA/year** = false alarms per year during baseline\n",
            "- **Lift** = pre-shock alarm rate / baseline alarm rate\n\n",
            "## Threshold Performance\n\n",
            "| Level (Threshold) | Lead (days) | Recall | Precision | FA/year | Lift |\n",
            "|---|---|---|---|---|---|\n",
        ]
        for level, thresh in self.THRESHOLDS.items():
            row = metrics[metrics["threshold"] == thresh]
            if len(row) > 0:
                r = row.iloc[0]
                lift_s = f"{r['lift']:.1f}x" if r["lift"] < 20 else ">20x"
                lines.append(
                    f"| {level} ({thresh}) | {r['lead_days']} | "
                    f"{r['recall']:.1%} | {r['precision']:.1%} | "
                    f"{r['fa_per_year']:.1f} | {lift_s} |\n"
                )
        lines += [
            "\n## Interpretation\n\n",
            "- Higher threshold -> lower recall, lower FA/year, higher precision.\n",
            "- Lift > 1 indicates that pre-shock TRS is significantly elevated vs baseline.\n",
            f"\n_Generated: {datetime.now().strftime('%Y-%m-%d')}_\n",
        ]
        rpt = self.out_dir / "report.md"
        with open(rpt, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  {rpt.name}")

    def run(self) -> None:
        print("=" * 60)
        print(f"{self.item_title} early warning analysis")
        print("=" * 60)

        print("\n[1/5] Loading data")
        daily = self.load_daily_trs()
        weekly = self.load_weekly_trs()
        merged = self.load_merged_data()

        if daily is None:
            return
        if merged is None:
            print("  Run step03_merge_features first")
            return
        if self.score_col not in daily.columns:
            print(f"  Missing '{self.score_col}' column in daily TRS")
            return

        print("\n[2/5] Overview")
        self.plot_overview(daily)
        print("\n[3/5] Signal quality")
        metrics = self.plot_signal_quality(daily)
        print("\n[4/5] Shock trajectory")
        if weekly is not None:
            self.plot_shock_trajectory(weekly, merged)
        print("\n[5/5] Warning simulation")
        self.plot_warning_simulation(merged)
        print("\n[report] generate")
        self.generate_report(metrics, daily)


def run(dataset: str, cfg: SimpleNamespace, data_root: Path | None) -> None:
    ews = EarlyWarningSystem(dataset, cfg, data_root)
    ews.run()


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
