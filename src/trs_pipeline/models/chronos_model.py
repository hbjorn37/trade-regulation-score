"""Zero-shot Chronos forecaster wrapper.

Uses the amazon/chronos-bolt-small checkpoint, which runs on CPU and finishes
the one-step-ahead test window in roughly the same wall-clock time as the
gradient-boosting models. Chronos is univariate, so this wrapper does not
accept exogenous TRS features: step04 reports a single Chronos row instead of
the BASE/LLM pair used for the other models.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_CHECKPOINT = "amazon/chronos-bolt-small"
DEFAULT_CONTEXT_LENGTH = 64

_pipeline_cache: dict[str, object] = {}


def _load_pipeline(checkpoint: str):
    if checkpoint in _pipeline_cache:
        return _pipeline_cache[checkpoint]
    try:
        import torch
        from chronos import ChronosBoltPipeline
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise RuntimeError("chronos-forecasting is not installed") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    pipeline = ChronosBoltPipeline.from_pretrained(
        checkpoint,
        device_map=device,
        torch_dtype=dtype,
    )
    _pipeline_cache[checkpoint] = pipeline
    return pipeline


def forecast_chronos_one_step(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> list[float]:
    """One-step-ahead forecasts over the test window, using actual lag values.

    Returns the median quantile of the Chronos forecast at each step.
    """
    pipeline = _load_pipeline(checkpoint)
    import torch

    all_y = pd.concat(
        [
            train[target_col].reset_index(drop=True),
            test[target_col].reset_index(drop=True),
        ]
    ).astype(float).values

    train_size = len(train)
    forecasts: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(test)):
            t = train_size + i
            start = max(0, t - context_length)
            context = all_y[start:t]
            if len(context) == 0:
                forecasts.append(float("nan"))
                continue
            ctx_tensor = torch.tensor(context, dtype=torch.float32)
            quantiles, _ = pipeline.predict_quantiles(
                inputs=ctx_tensor,
                prediction_length=1,
                quantile_levels=[0.5],
            )
            pred = float(quantiles[0, 0, 0].item())
            forecasts.append(pred)

    return forecasts


def is_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
