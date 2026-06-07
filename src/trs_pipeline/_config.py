"""Config loading shared across pipeline steps."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from trs_pipeline._paths import CONFIG_DIR


def load_config(dataset: str) -> SimpleNamespace:
    path = CONFIG_DIR / f"{dataset}.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def week_sort_key(week_str: str) -> tuple[int, int]:
    year, week = week_str.split("_")
    return (int(year), int(week))


def parse_dataset_arg(parser_args) -> list[str]:
    if parser_args.dataset is None or parser_args.dataset == "all":
        return ["semi", "urea"]
    return [parser_args.dataset]


def resolve_data_dir(data_root: Path | None, dataset: str) -> Path:
    from trs_pipeline._paths import DATA_ROOT_DEFAULT

    root = data_root if data_root is not None else DATA_ROOT_DEFAULT
    return Path(root) / dataset
