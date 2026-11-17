"""Evaluation metrics module for time series demand forecasting.

Implements:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error with explicit zero protection)
- sMAPE (Symmetric Mean Absolute Percentage Error with explicit zero protection)
- WAPE (Weighted Absolute Percentage Error / Normalized MAE)
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """Compute MAPE with explicit zero-handling protection.

    Uses epsilon floor for denominator to prevent division by zero in zero-sales intervals:
    MAPE = 100 * mean(|y - y_hat| / max(|y|, epsilon))
    """
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """Compute sMAPE with explicit zero-handling protection.

    sMAPE = 100 * mean( 2 * |y - y_hat| / (|y| + |y_hat| + epsilon) )
    Bounded between 0% and 200%.
    """
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(numerator / denominator) * 100.0)


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute WAPE (Weighted Absolute Percentage Error).

    WAPE = sum(|y - y_hat|) / sum(y) * 100
    Standard retail KPI that does not inflate on low-volume / intermittent items.
    """
    total_actual = float(np.sum(y_true))
    if total_actual == 0.0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else 100.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)


def calculate_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate all standard forecast accuracy metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)})")
    if len(yt) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    return {
        "mae": float(np.round(mean_absolute_error(yt, yp), 4)),
        "rmse": float(np.round(root_mean_squared_error(yt, yp), 4)),
        "mape": float(np.round(mean_absolute_percentage_error(yt, yp), 2)),
        "smape": float(np.round(symmetric_mean_absolute_percentage_error(yt, yp), 2)),
        "wape": float(np.round(weighted_absolute_percentage_error(yt, yp), 2)),
    }
