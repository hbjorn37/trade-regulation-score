"""Train and evaluate forecasting models on the merged weekly dataset."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from xgboost import XGBRegressor

from trs_pipeline._config import load_config, parse_dataset_arg
from trs_pipeline._paths import INTERMEDIATE_ROOT, OUTPUT_ROOT
from trs_pipeline.models.chronos_model import (
    forecast_chronos_one_step,
    is_available as chronos_available,
)

warnings.filterwarnings("ignore")

AVAILABLE_MODELS = [
    "ARIMA",
    "RandomForest",
    "LSTM",
    "GBM",
    "LightGBM",
    "NGBoost",
    "XGBoost",
    "CatBoost",
    "ExtraTrees",
    "Chronos",
]

ML_MODELS = {"RandomForest", "GBM", "LightGBM", "NGBoost", "XGBoost", "CatBoost", "ExtraTrees"}
TRS_ONLY_MODELS = {"Chronos"}  # univariate: no LLM variant

TARGET_COL = "import_unit_price"

TRS_EXOG_COLS = [
    "trs_mean_lag1",
    "trs_mean_lag2",
    "trs_mean_lag3",
    "trs_mean_lag4",
    "trs_max_lag1",
    "trs_pct_change_lag1",
    "trs_spike_lag1",
    "trs_high_risk_lag1",
    "trs_spike_cum4w_lag1",
    "risk_signal_ratio_lag1",
]


def validate_trs_predictive_power(df: pd.DataFrame, target_col: str, maxlag: int = 8) -> dict:
    print(f"\n  TRS leading-indicator check ({target_col})")
    results = {"granger_pvalues": {}, "correlations": {}, "is_leading_indicator": False}

    if "trs_mean" not in df.columns or target_col not in df.columns:
        return results

    test_data = df[[target_col, "trs_mean"]].dropna()
    if len(test_data) < maxlag + 5:
        return results

    try:
        gc = grangercausalitytests(test_data[[target_col, "trs_mean"]], maxlag=maxlag, verbose=False)
        significant_lags = []
        for lag in range(1, maxlag + 1):
            pv = gc[lag][0]["ssr_ftest"][1]
            results["granger_pvalues"][lag] = pv
            if pv < 0.05:
                significant_lags.append(lag)
        if significant_lags:
            print(f"    TRS significant at lags {significant_lags}")
            results["is_leading_indicator"] = True
        else:
            print("    TRS not significant at conventional levels")
    except Exception as e:
        print(f"    Granger test failed: {e}")

    for lag in range(1, maxlag + 1):
        results["correlations"][lag] = df[target_col].corr(df["trs_mean"].shift(lag))
    return results


def fit_arima_and_forecast(data, forecast_steps, exog=None, future_exog=None, order=(1, 1, 1)):
    model = ARIMA(data, exog=exog, order=order)
    fit = model.fit()
    if exog is not None:
        if future_exog is None:
            last_exog = exog.iloc[-1] if hasattr(exog, "iloc") else exog[-1]
            future_exog = np.array([[last_exog]] * forecast_steps)
        forecast = fit.forecast(steps=forecast_steps, exog=future_exog)
    else:
        forecast = fit.forecast(steps=forecast_steps)
    return fit, forecast


def create_lagged_features(data, lags=8):
    df = pd.DataFrame(data.copy())
    df.columns = ["y"]
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    return df.dropna()


def prepare_ml_features(
    train_df: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[list[str]] = None,
    lags: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lagged_df = create_lagged_features(train_df[target_col], lags=lags)
    feature_names = [f"lag_{i}" for i in range(1, lags + 1)]

    if exog_cols is not None:
        valid_cols = [c for c in exog_cols if c in train_df.columns]
        for col in valid_cols:
            values = train_df[col].values[-len(lagged_df) :]
            if np.isnan(values).any():
                values = pd.Series(values).ffill().fillna(0).values
            lagged_df[col] = values
            feature_names.append(col)

    X = lagged_df.drop("y", axis=1).values
    y = lagged_df["y"].values
    return X, y, feature_names


def _build_ml_model(model_type: str):
    if model_type == "GBM":
        return GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=3,
            min_samples_split=2, min_samples_leaf=1, subsample=0.9, random_state=42,
        )
    if model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=500, max_depth=5,
            min_samples_split=2, min_samples_leaf=1, random_state=42,
        )
    if model_type == "LightGBM":
        return LGBMRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=3,
            num_leaves=31, random_state=42, verbose=-1,
        )
    if model_type == "NGBoost":
        return NGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42, verbose=False)
    if model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=3,
            subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
    if model_type == "CatBoost":
        return CatBoostRegressor(iterations=500, learning_rate=0.01, depth=3, random_seed=42, verbose=0)
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=5,
            min_samples_split=2, min_samples_leaf=1, random_state=42,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def fit_lstm_and_forecast(
    train_df: pd.DataFrame,
    target_col: str,
    forecast_steps: int,
    exog_cols: Optional[list[str]] = None,
    lags: int = 8,
):
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
    except ImportError:
        print("    TensorFlow not available -- skipping LSTM")
        return None, [np.nan] * forecast_steps

    X, y, _ = prepare_ml_features(train_df, target_col, exog_cols, lags)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential([
        LSTM(32, input_shape=(1, X_scaled.shape[1]), recurrent_dropout=0.2, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3, clipvalue=1.0), loss="mse")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, min_delta=1e-4)
    model.fit(X_lstm, y_scaled, epochs=200, batch_size=16, validation_split=0.1, verbose=0, callbacks=[early_stop])

    forecast = []
    last_y_values = y[-lags:].tolist()
    if exog_cols:
        valid_cols = [c for c in exog_cols if c in train_df.columns]
        last_exog_values = {c: train_df[c].iloc[-1] for c in valid_cols}
    else:
        last_exog_values = {}
        valid_cols = []

    for _ in range(forecast_steps):
        feats = last_y_values[-lags:][::-1]
        for col in valid_cols:
            feats.append(last_exog_values.get(col, 0))
        input_scaled = scaler_X.transform(np.array(feats).reshape(1, -1))
        input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        pred_scaled = model.predict(input_lstm, verbose=0)[0, 0]
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        forecast.append(pred)
        last_y_values.append(pred)

    return model, forecast


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.sum() > 0 else np.nan
    )
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    theil_u = num / den if den != 0 else np.nan
    return mae, rmse, mape, theil_u


def calculate_early_warning_metrics(y_true, y_pred, threshold=0.1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {
            "direction_accuracy": np.nan,
            "spike_recall": np.nan,
            "spike_precision": np.nan,
            "trend_correlation": np.nan,
        }
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    direction_accuracy = (true_dir == pred_dir).mean()
    true_chg = np.abs(np.diff(y_true) / (y_true[:-1] + 1e-10))
    pred_chg = np.abs(np.diff(y_pred) / (y_pred[:-1] + 1e-10))
    true_spike = true_chg > threshold
    pred_spike = pred_chg > threshold
    spike_recall = (
        (true_spike & pred_spike).sum() / true_spike.sum() if true_spike.sum() > 0 else np.nan
    )
    spike_precision = (
        (true_spike & pred_spike).sum() / pred_spike.sum() if pred_spike.sum() > 0 else np.nan
    )
    trend_corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {
        "direction_accuracy": direction_accuracy,
        "spike_recall": spike_recall,
        "spike_precision": spike_precision,
        "trend_correlation": trend_corr,
    }


def forecast_ml_one_step(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    model_type: str,
    exog_cols: Optional[list[str]] = None,
    lags: int = 8,
) -> list[float]:
    X_train, y_train, _ = prepare_ml_features(train, target_col, exog_cols, lags)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    model = _build_ml_model(model_type)
    model.fit(X_scaled, y_scaled)

    all_y = pd.concat([train[target_col].reset_index(drop=True), test[target_col].reset_index(drop=True)])
    all_df = pd.concat([train.reset_index(drop=True), test.reset_index(drop=True)]).reset_index(drop=True)
    train_size = len(train)
    valid_cols = [c for c in (exog_cols or []) if c in all_df.columns]

    forecasts = []
    for i in range(len(test)):
        t = train_size + i
        if t >= lags:
            y_lags = all_y.iloc[t - lags : t].values[::-1]
        else:
            partial = all_y.iloc[:t].values[::-1]
            y_lags = np.pad(partial, (0, lags - len(partial)), mode="edge")
        trs_vals = [
            float(all_df.iloc[t][col]) if not pd.isna(all_df.iloc[t][col]) else 0.0
            for col in valid_cols
        ]
        feat = np.array(y_lags.tolist() + trs_vals).reshape(1, -1)
        if np.isnan(feat).any():
            feat = np.nan_to_num(feat, nan=0.0)
        pred_scaled = model.predict(scaler_X.transform(feat))[0]
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        forecasts.append(pred)
    return forecasts


def _run_model(model_name: str, train: pd.DataFrame, test: pd.DataFrame, target_col: str,
               forecast_periods: int, use_trs: bool) -> list[float]:
    if model_name == "ARIMA":
        if use_trs:
            exog_col = "trs_mean_lag1"
            exog = train[exog_col] if exog_col in train.columns else None
            future = (
                test[exog_col].values.reshape(-1, 1)
                if exog is not None and exog_col in test.columns else None
            )
            _, forecast = fit_arima_and_forecast(train[target_col], forecast_periods, exog=exog, future_exog=future)
        else:
            _, forecast = fit_arima_and_forecast(train[target_col], forecast_periods)
        return list(forecast.values if hasattr(forecast, "values") else forecast)

    if model_name == "LSTM":
        exog_cols = [c for c in TRS_EXOG_COLS if c in train.columns] if use_trs else None
        _, forecast = fit_lstm_and_forecast(train, target_col, forecast_periods, exog_cols=exog_cols)
        return list(forecast)

    if model_name == "Chronos":
        if not chronos_available():
            print("    chronos-forecasting not available -- skipping")
            return [np.nan] * forecast_periods
        return forecast_chronos_one_step(train, test, target_col)

    if model_name in ML_MODELS:
        exog_cols = [c for c in TRS_EXOG_COLS if c in train.columns] if use_trs else None
        return forecast_ml_one_step(train, test, target_col, model_name, exog_cols=exog_cols)

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_models(forecast_results: dict, actual: np.ndarray, models: list[str],
                    output_dir: Path, cfg: SimpleNamespace) -> list[dict]:
    print("\n  Evaluation")
    print("-" * 90)

    eval_results: list[dict] = []
    base_metrics: dict[str, tuple[float, float, float, float]] = {}

    def _metric_row(name: str, mae, rmse, mape, theil_u, ew, base_mae=None):
        improvement_mae = ((mae - base_mae) / base_mae) * 100 if base_mae else None
        return {
            "model": name,
            "mae": f"{mae:.4f}" + (f" ({improvement_mae:+.2f}%)" if base_mae else ""),
            "rmse": f"{rmse:.4f}",
            "mape": f"{mape:.4f}",
            "theil_u": f"{theil_u:.4f}",
            "direction_accuracy": f"{ew['direction_accuracy']:.2%}",
            "spike_detection_rate": (
                f"{ew['spike_recall']:.2%}" if not np.isnan(ew["spike_recall"]) else "N/A"
            ),
            "trend_corr": f"{ew['trend_correlation']:.4f}",
        }

    for model_name in models:
        is_trs_only = model_name in TRS_ONLY_MODELS
        variants = [("", False)] if is_trs_only else [("", False), ("LLM_", True)]
        for prefix, _use_trs in variants:
            full_name = f"{prefix}{model_name}" if prefix else model_name
            col_name = f"{full_name}_pred"
            if col_name not in forecast_results:
                continue
            forecast_arr = np.array(forecast_results[col_name])
            if np.isnan(forecast_arr).any():
                print(f"    {full_name:20} contains NaN -- excluded")
                continue
            if np.abs(forecast_arr).max() > np.abs(actual).max() * 100:
                print(f"    {full_name:20} diverged -- excluded")
                continue

            mae, rmse, mape, theil_u = calculate_metrics(actual, forecast_arr)
            ew = calculate_early_warning_metrics(actual, forecast_arr)

            if prefix == "":
                base_metrics[model_name] = (mae, rmse, mape, theil_u)
                eval_results.append(_metric_row(model_name, mae, rmse, mape, theil_u, ew))
                print(
                    f"    {model_name:20} MAE={mae:.4f}, RMSE={rmse:.4f}, "
                    f"MAPE={mape:.1f}%, dir={ew['direction_accuracy']:.1%}"
                )
            else:
                base_mae = base_metrics.get(model_name, (None,))[0]
                if base_mae is None:
                    continue
                eval_results.append(_metric_row(full_name, mae, rmse, mape, theil_u, ew, base_mae=base_mae))
                imp = ((mae - base_mae) / base_mae) * 100 if base_mae != 0 else 0
                status = "improved" if imp < 0 else "worse"
                print(f"    {full_name:20} MAE={mae:.4f} ({imp:+.1f}%) {status}")

    eval_df = pd.DataFrame(eval_results)
    eval_file = output_dir / "eval.csv"
    eval_df.to_csv(eval_file, index=False, encoding="utf-8")
    print(f"\n  Saved: {eval_file}")

    eval_df["item"] = cfg.item_title
    eval_df["hs_code"] = cfg.hs_code
    eval_df["target"] = TARGET_COL
    cols_order = [
        "item", "hs_code", "target", "model",
        "mae", "rmse", "mape", "theil_u",
        "direction_accuracy", "spike_detection_rate", "trend_corr",
    ]
    cols_order = [c for c in cols_order if c in eval_df.columns]
    eval_all_file = output_dir / "eval_all.csv"
    eval_df[cols_order].to_csv(eval_all_file, index=False, encoding="utf-8")
    print(f"  Saved: {eval_all_file}")

    return eval_results


def run_fixed_split(train: pd.DataFrame, test: pd.DataFrame, target_col: str,
                    models: list[str], output_dir: Path, cfg: SimpleNamespace) -> list[dict]:
    forecast_periods = len(test)
    forecast_results = {
        "period": test["year_week"].values,
        "actual": test[target_col].values,
    }

    for model_name in models:
        is_trs_only = model_name in TRS_ONLY_MODELS

        print(f"\n  {model_name} BASE")
        try:
            forecast = _run_model(model_name, train, test, target_col, forecast_periods, use_trs=False)
            forecast_results[f"{model_name}_pred"] = forecast
            print(f"    done")
        except Exception as e:
            print(f"    failed: {e}")
            forecast_results[f"{model_name}_pred"] = [np.nan] * forecast_periods

        if is_trs_only:
            continue

        print(f"  {model_name} LLM (with TRS)")
        try:
            forecast = _run_model(model_name, train, test, target_col, forecast_periods, use_trs=True)
            forecast_results[f"LLM_{model_name}_pred"] = forecast
            print(f"    done")
        except Exception as e:
            print(f"    failed: {e}")
            forecast_results[f"LLM_{model_name}_pred"] = [np.nan] * forecast_periods

    forecast_df = pd.DataFrame(forecast_results)
    forecast_file = output_dir / "forecast.csv"
    forecast_df.to_csv(forecast_file, index=False, encoding="utf-8")
    print(f"\n  Saved: {forecast_file}")

    return evaluate_models(forecast_results, test[target_col].values, models, output_dir, cfg)


def run(dataset: str, cfg: SimpleNamespace, models: Optional[list[str]] = None,
        target_col: Optional[str] = None) -> None:
    models = models or AVAILABLE_MODELS
    target_col = target_col or TARGET_COL

    print("=" * 70)
    print(f"{cfg.item_title} import forecasting")
    print("=" * 70)
    print(f"Models: {', '.join(models)}")
    print(f"Target: {target_col}")

    inter_dir = INTERMEDIATE_ROOT / dataset
    output_dir = OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = inter_dir / "merged.csv"
    if not data_file.exists():
        print(f"  merged.csv not found at {data_file} -- run step03 first")
        return

    df = pd.read_csv(data_file, encoding="utf-8-sig")
    df["_sort_key"] = df["year_week"].apply(lambda x: tuple(int(v) for v in x.split("_")))
    df = df.sort_values("_sort_key").drop("_sort_key", axis=1).reset_index(drop=True)

    # Trim a long trailing run of identical import rows, which signals an
    # interpolation tail at the end of a daily customs extract. Monthly data
    # disaggregated by calendar-day proration produces runs of up to ~5
    # identical weeks within a single month, so we only fire on runs of 8+
    # to avoid false positives on prorated input.
    import_cols = [c for c in ["reported_weight", "taxable_value_usd"] if c in df.columns]
    if import_cols and len(df) > 8:
        is_repeat = (df[import_cols].diff().fillna(0) == 0).all(axis=1)
        trail = 0
        for r in reversed(is_repeat.tolist()):
            if r:
                trail += 1
            else:
                break
        if trail >= 8:
            cutoff_idx = len(df) - trail
            cutoff = df.loc[cutoff_idx, "year_week"]
            print(f"  Trimming trailing repeated tail from {cutoff} ({trail} weeks)")
            df = df.iloc[:cutoff_idx].reset_index(drop=True)

    if target_col not in df.columns:
        print(f"  Target column '{target_col}' missing -- falling back to import_unit_price")
        target_col = "import_unit_price"

    if df[target_col].isna().any():
        df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    print(f"\n  Data: {df['year_week'].iloc[0]} to {df['year_week'].iloc[-1]} ({len(df)} weeks)")
    validate_trs_predictive_power(df, target_col)

    train = df[df["year_week"] <= cfg.train_end_week].copy()
    test = df[df["year_week"] >= cfg.test_start_week].copy()
    print(f"\n  Train: {train['year_week'].iloc[0]} to {train['year_week'].iloc[-1]} ({len(train)} weeks)")
    print(f"  Test:  {test['year_week'].iloc[0]} to {test['year_week'].iloc[-1]} ({len(test)} weeks)")
    run_fixed_split(train, test, target_col, models, output_dir, cfg)

    print("\n" + "=" * 70)
    print("Forecasting complete")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="semi | urea | all (default)")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    for dataset in parse_dataset_arg(args):
        cfg = load_config(dataset)
        run(dataset, cfg, models=args.models, target_col=args.target)


if __name__ == "__main__":
    main()
