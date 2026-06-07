"""Pipeline entrypoint.

Usage:
    uv run python scripts/run.py semi              # all 6 steps for semi
    uv run python scripts/run.py urea              # all 6 steps for urea
    uv run python scripts/run.py all               # semi then urea

    uv run python scripts/run.py semi --from 04    # steps 04..06
    uv run python scripts/run.py semi --from 04 --to 05
    uv run python scripts/run.py semi --step 03    # only step 03
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trs_pipeline._config import load_config  # noqa: E402
from trs_pipeline import (  # noqa: E402
    step01_import_weekly,
    step02_trs_weekly,
    step03_merge_features,
    step04_forecast_model,
    step05_plot_evaluation,
    step06_early_warning,
)

STEPS = {
    "01": ("step01_import_weekly", step01_import_weekly),
    "02": ("step02_trs_weekly", step02_trs_weekly),
    "03": ("step03_merge_features", step03_merge_features),
    "04": ("step04_forecast_model", step04_forecast_model),
    "05": ("step05_plot_evaluation", step05_plot_evaluation),
    "06": ("step06_early_warning", step06_early_warning),
}

# Steps that accept a --data-root override (read real data files)
STEPS_WITH_DATA_ROOT = {"01", "02", "06"}


def run_step(step_id: str, dataset: str, data_root: Path | None) -> bool:
    name, module = STEPS[step_id]
    cfg = load_config(dataset)
    print(f"\n{'=' * 60}")
    print(f"[{step_id}] {name} (dataset={dataset})")
    print(f"{'=' * 60}")
    try:
        if step_id in STEPS_WITH_DATA_ROOT:
            module.run(dataset, cfg, data_root)
        else:
            module.run(dataset, cfg)
        print(f"\n[{step_id}] OK")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[{step_id}] FAILED: {e}")
        return False


def run_pipeline(dataset: str, steps: list[str], data_root: Path | None) -> bool:
    print(f"\nPipeline start: dataset={dataset}, steps={steps}")
    for step_id in steps:
        if not run_step(step_id, dataset, data_root):
            print(f"\nPipeline halted at [{step_id}]")
            return False
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete: dataset={dataset}")
    print(f"{'=' * 60}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="semi, urea, or all")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=str, help="Run only this step (e.g. 03)")
    group.add_argument("--from", dest="from_step", type=str, help="Start step (e.g. 04)")
    parser.add_argument("--to", dest="to_step", type=str, help="End step (e.g. 05)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data root (default: data/sample)")
    return parser.parse_args()


def get_steps_to_run(from_step: str | None, to_step: str | None,
                     step: str | None) -> list[str]:
    all_steps = sorted(STEPS.keys())
    if step:
        sid = step.zfill(2)
        if sid not in STEPS:
            raise SystemExit(f"Unknown step: {step}. Choose from {list(STEPS.keys())}")
        return [sid]
    if from_step:
        start = from_step.zfill(2)
        end = (to_step or all_steps[-1]).zfill(2)
        if start not in STEPS or end not in STEPS:
            raise SystemExit(f"Unknown step range: {from_step}..{to_step}")
        return [s for s in all_steps if start <= s <= end]
    return all_steps


def main() -> None:
    args = parse_args()
    steps = get_steps_to_run(args.from_step, args.to_step, args.step)
    datasets = ["semi", "urea"] if args.dataset == "all" else [args.dataset]

    all_ok = True
    for ds in datasets:
        if not run_pipeline(ds, steps, args.data_root):
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
