"""Chronological split and rolling-origin backtesting engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from chronoscast.evaluation.metrics import calculate_all_metrics
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChronologicalSplitter:
    """Partitions time series strictly chronologically without leakage."""

    def __init__(
        self,
        train_end: str = "2016-12-31",
        val_end: str = "2017-06-30",
        test_end: str = "2017-12-31",
        date_col: str = "date",
    ) -> None:
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.test_end = pd.to_datetime(test_end)
        self.date_col = date_col

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into Train, Validation, and Test sets."""
        dates = pd.to_datetime(df[self.date_col])
        train = df[dates <= self.train_end].copy()
        val = df[(dates > self.train_end) & (dates <= self.val_end)].copy()
        test = df[(dates > self.val_end) & (dates <= self.test_end)].copy()

        logger.info(
            "Chronological split completed: Train=%d rows (%s to %s), Val=%d rows (%s to %s), Test=%d rows (%s to %s)",
            len(train), dates.min().date(), self.train_end.date(),
            len(val), (self.train_end + pd.Timedelta(days=1)).date(), self.val_end.date(),
            len(test), (self.val_end + pd.Timedelta(days=1)).date(), self.test_end.date(),
        )
        return train, val, test


class RollingOriginBacktester:
    """Evaluates forecasting models across multiple chronological forecast origins."""

    def __init__(
        self,
        origins: List[str],
        horizons: Optional[List[int]] = None,
        date_col: str = "date",
        target_col: str = "sales",
    ) -> None:
        self.origins = [pd.to_datetime(o) for o in origins]
        self.horizons = horizons if horizons is not None else [7, 14, 28]
        self.date_col = date_col
        self.target_col = target_col

    def evaluate_model(
        self,
        model_factory: Any,
        df: pd.DataFrame,
        model_name: str,
    ) -> Dict[str, Any]:
        """Run rolling-origin evaluation for a model across all configured origins and horizons."""
        results: List[Dict[str, Any]] = []
        df_sorted = df.sort_values(by=[self.date_col, "store", "item"]).copy()
        df_dates = pd.to_datetime(df_sorted[self.date_col])

        for origin in self.origins:
            train_mask = df_dates <= origin
            train_data = df_sorted[train_mask]

            # Fit model on historical data up to origin
            t0 = time.time()
            model = model_factory()
            model.fit(train_data)
            fit_time = time.time() - t0

            for h in self.horizons:
                forecast_start = origin + pd.Timedelta(days=1)
                forecast_end = origin + pd.Timedelta(days=h)
                test_mask = (df_dates >= forecast_start) & (df_dates <= forecast_end)
                actual_future = df_sorted[test_mask]

                if len(actual_future) == 0:
                    continue

                preds_df = model.predict(horizon=h)

                # Merge predictions with actuals
                merged = pd.merge(
                    actual_future,
                    preds_df,
                    on=["date", "store", "item"],
                    suffixes=("_actual", "_pred"),
                )

                metrics = calculate_all_metrics(merged[self.target_col], merged["prediction"])
                results.append({
                    "model": model_name,
                    "origin": str(origin.date()),
                    "horizon": h,
                    "eval_window_start": str(forecast_start.date()),
                    "eval_window_end": str(forecast_end.date()),
                    "fit_time_sec": np.round(fit_time, 2),
                    **metrics,
                })

        res_df = pd.DataFrame(results)
        summary = (
            res_df.groupby(["model", "horizon"])[["mae", "rmse", "mape", "smape", "wape"]]
            .mean()
            .reset_index()
        )

        return {
            "detailed_windows": res_df.to_dict(orient="records"),
            "horizon_summary": summary.to_dict(orient="records"),
        }
