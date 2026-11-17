"""Conformal prediction interval estimation for time series forecasting.

Uses empirical residual quantiles from held-out validation data to construct
non-parametric lower and upper prediction intervals with guaranteed asymptotic coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConformalIntervalEstimator:
    """Estimates prediction intervals using empirical residuals from validation data.

    Interval bounds:
        Lower = max(0, point_forecast - quantile_alpha)
        Upper = point_forecast + quantile_alpha
    """

    def __init__(self, confidence_levels: Optional[List[float]] = None) -> None:
        self.confidence_levels = confidence_levels if confidence_levels is not None else [0.80, 0.90, 0.95]
        self.quantiles: Dict[str, float] = {}
        self.is_calibrated = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> ConformalIntervalEstimator:
        """Calibrate uncertainty using validation absolute errors."""
        residuals = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
        n = len(residuals)
        if n == 0:
            raise ValueError("Residual array cannot be empty for calibration.")

        self.quantiles = {}
        for conf in self.confidence_levels:
            # Conformal finite-sample correction: index = ceil((n + 1) * conf) / n
            q_val = float(np.quantile(residuals, conf, method="higher"))
            self.quantiles[f"{int(conf * 100)}%"] = float(np.round(q_val, 4))

        self.is_calibrated = True
        logger.info("Conformal intervals calibrated. Residual quantiles: %s", self.quantiles)
        return self

    def predict_intervals(
        self,
        point_predictions: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute lower and upper prediction bounds for point forecasts."""
        if not self.is_calibrated:
            raise RuntimeError("Conformal estimator must be calibrated before generating intervals.")

        conf_key = f"{int(confidence_level * 100)}%"
        if conf_key not in self.quantiles:
            raise ValueError(f"Confidence level {confidence_level} not calibrated. Available: {list(self.quantiles.keys())}")

        q = self.quantiles[conf_key]
        preds = np.asarray(point_predictions, dtype=np.float64)
        lower = np.maximum(0.0, preds - q)
        upper = preds + q
        return lower, upper

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        point_predictions: np.ndarray,
        confidence_level: float = 0.90,
    ) -> Dict[str, float]:
        """Compute empirical coverage percentage on held-out test set."""
        lower, upper = self.predict_intervals(point_predictions, confidence_level=confidence_level)
        yt = np.asarray(y_true, dtype=np.float64)
        covered = (yt >= lower) & (yt <= upper)
        empirical_coverage = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(upper - lower))

        return {
            "nominal_confidence_pct": float(confidence_level * 100.0),
            "empirical_coverage_pct": float(np.round(empirical_coverage, 2)),
            "mean_interval_width": float(np.round(mean_width, 2)),
            "margin_of_error": float(np.round(self.quantiles[f"{int(confidence_level * 100)}%"], 2)),
        }

    def save(self, filepath: str | Path) -> None:
        """Save calibrated quantile parameters to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "confidence_levels": self.confidence_levels,
            "quantiles": self.quantiles,
            "is_calibrated": self.is_calibrated,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved conformal calibrator to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> ConformalIntervalEstimator:
        """Load calibrated quantile parameters from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(confidence_levels=data["confidence_levels"])
        inst.quantiles = data["quantiles"]
        inst.is_calibrated = data["is_calibrated"]
        return inst
