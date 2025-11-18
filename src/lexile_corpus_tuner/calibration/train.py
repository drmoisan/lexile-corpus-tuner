# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportInvalidTypeForm=false

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

Array = Any
pd = cast(Any, pd)
np = cast(Any, np)
pearsonr = cast(Any, pearsonr)
train_test_split = cast(Any, train_test_split)

FEATURE_COLS = [
    "overall_msl",
    "overall_mlf",
    "log_num_tokens",
    "msl_std",
    "mlf_std",
    "overall_msl_sq",
    "overall_mlf_sq",
    "msl_times_mlf",
]

TARGET_COL = "lexile_official"


def compute_metrics(y_true: Array, y_pred: Array) -> dict[str, float]:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    if len(y_true) > 1:
        corr, _ = pearsonr(y_true, y_pred)
        corr_value = float(corr)
    else:
        corr_value = float("nan")
    return {"rmse": rmse, "mae": mae, "r": corr_value}


def train_regression_model(df: Any) -> tuple[ElasticNet, dict[str, float]]:
    """Train an ElasticNet regression model on calibration data."""
    filtered = df.dropna(subset=[TARGET_COL, "overall_msl", "overall_mlf"])
    filtered = filtered[filtered["num_tokens"] >= 100]
    filtered = filtered[filtered["num_slices"] >= 1]

    if filtered.empty:
        raise ValueError("Calibration dataset is empty after filtering.")

    train_df: pd.DataFrame
    val_df: pd.DataFrame

    if len(filtered) < 5:
        train_df = filtered
        val_df = filtered
    else:
        train_df, val_df = train_test_split(
            filtered,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

    X_train: Array = np.asarray(train_df[FEATURE_COLS].values, dtype=float)
    y_train: Array = np.asarray(train_df[TARGET_COL].values, dtype=float)

    X_val: Array = np.asarray(val_df[FEATURE_COLS].values, dtype=float)
    y_val: Array = np.asarray(val_df[TARGET_COL].values, dtype=float)

    model = ElasticNet(
        alpha=0.001,
        l1_ratio=0.0,
        fit_intercept=True,
        max_iter=10000,
    )
    model.fit(X_train, y_train)

    yhat_val = model.predict(X_val)
    metrics = compute_metrics(y_val, yhat_val)

    return model, metrics
